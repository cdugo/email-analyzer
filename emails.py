import os
from PIL import Image
import numpy as np
from pdf2image import convert_from_path
import torch
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation
from scipy import ndimage
from PIL import ImageDraw

@dataclass
class SlideChange:
    slide_number: int
    similarity_score: float
    change_regions: List[Dict[str, any]]  # x1, y1, x2, y2, type, area
    associated_comments: List[str]

@dataclass
class VersionDiff:
    version_from: str
    version_to: str
    timestamp: str
    changes: List[SlideChange]

class PDFDiffAnalyzer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        # Initialize CLIP model for semantic image comparison
        print("Loading CLIP model...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("Model loaded successfully")
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """Convert PDF to list of images."""
        print(f"Converting PDF to images: {pdf_path}")
        return convert_from_path(pdf_path)

    def get_slide_embedding(self, image: Image.Image) -> torch.Tensor:
        """Get CLIP embedding for an image."""
        # Preprocess image for CLIP
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        # Get image features
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            
        return image_features
    
    def classify_change_type(self, region1: np.ndarray, region2: np.ndarray) -> str:
        """Classify the type of change in a region."""
        # Calculate various metrics to determine change type
        intensity_diff = np.abs(region1.mean() - region2.mean())
        texture_diff = np.abs(region1.std() - region2.std())
        size_diff = abs(region1.size - region2.size)
        
        # Adjusted thresholds for classification
        if size_diff > 2000:  # Increased threshold for resize detection
            return 'resize'
        elif texture_diff > 75:  # Increased threshold for image changes
            return 'image'
        elif intensity_diff > 50:  # Added specific threshold for text changes
            return 'text'
        return None  # Return None for insignificant changes

    def merge_nearby_regions(self, regions: List[Dict]) -> List[Dict]:
        """Merge nearby regions of the same type with more aggressive merging."""
        if not regions:
            return []
        
        # Filter out None type regions
        regions = [r for r in regions if r.get('type') is not None]
        
        merged = []
        regions = sorted(regions, key=lambda r: (r['y1'], r['x1']))
        
        while regions:
            current = regions.pop(0)
            
            # Look for nearby regions of the same type
            i = 0
            while i < len(regions):
                next_region = regions[i]
                
                # More lenient distance thresholds for merging
                distance_x = min(abs(current['x2'] - next_region['x1']), 
                            abs(current['x1'] - next_region['x2']))
                distance_y = min(abs(current['y2'] - next_region['y1']), 
                            abs(current['y1'] - next_region['y2']))
                
                # More aggressive merging conditions
                if ((distance_x < 100 and distance_y < 50) or  # Increased distance thresholds
                    (current['type'] == next_region['type'] and 
                    distance_x < 200 and distance_y < 100)):  # Even more lenient for same type
                    
                    # Merge regions
                    current = {
                        'x1': min(current['x1'], next_region['x1']),
                        'y1': min(current['y1'], next_region['y1']),
                        'x2': max(current['x2'], next_region['x2']),
                        'y2': max(current['y2'], next_region['y2']),
                        'area': current['area'] + next_region['area'],
                        'type': current['type']
                    }
                    regions.pop(i)
                else:
                    i += 1
            
            if current['area'] > 1000:  # Final area threshold check
                merged.append(current)
        
        return merged

    def detect_change_regions(self, img1: Image.Image, img2: Image.Image, 
                        threshold: float = 0.2) -> List[Dict[str, any]]:  # Increased threshold
        """Detect regions of significant change between two slides."""
        # Ensure images are the same size
        if img1.size != img2.size:
            img2 = img2.resize(img1.size)
            
        # Convert images to numpy arrays
        arr1 = np.array(img1.convert('RGB'))
        arr2 = np.array(img2.convert('RGB'))
        
        # Compute difference
        diff = np.abs(arr1 - arr2).mean(axis=2)
        
        # Apply stronger preprocessing to reduce noise
        diff_smooth = gaussian_filter(diff, sigma=2)  # Increased sigma
        
        # Create binary mask of changes
        binary_mask = diff_smooth > (threshold * 255)
        
        # More aggressive cleanup of the mask
        binary_mask = binary_erosion(binary_mask, iterations=3)  # More erosion
        binary_mask = binary_dilation(binary_mask, iterations=4)  # More dilation
        
        # Find connected components
        labeled_array, num_features = ndimage.label(binary_mask)
        
        regions = []
        min_area = 1000  # Increased minimum area threshold
        
        for i in range(1, num_features + 1):
            y_indices, x_indices = np.where(labeled_array == i)
            if len(y_indices) > 0 and len(x_indices) > 0:
                area = len(y_indices) * len(x_indices)
                if area > min_area:
                    # Calculate region properties
                    region = {
                        'x1': int(x_indices.min()),
                        'y1': int(y_indices.min()),
                        'x2': int(x_indices.max()),
                        'y2': int(y_indices.max()),
                        'area': area,
                        'type': self.classify_change_type(
                            arr1[y_indices.min():y_indices.max(), x_indices.min():x_indices.max()],
                            arr2[y_indices.min():y_indices.max(), x_indices.min():x_indices.max()]
                        )
                    }
                    regions.append(region)
        
        return self.merge_nearby_regions(regions)

    def analyze_versions(self, version1_path: str, version2_path: str, 
                        comments: List[str], similarity_threshold: float = 0.98) -> VersionDiff:
        """Analyze differences between two versions of a presentation."""
        print(f"Analyzing differences between {version1_path} and {version2_path}")
        
        # Convert both PDFs to images
        images1 = self.pdf_to_images(version1_path)
        images2 = self.pdf_to_images(version2_path)
        
        changes = []
        
        # Compare each slide
        for i in range(min(len(images1), len(images2))):
            print(f"Analyzing slide {i+1}")
            
            # Get embeddings
            embed1 = self.get_slide_embedding(images1[i])
            embed2 = self.get_slide_embedding(images2[i])
            
            # Compute similarity
            similarity = float(torch.nn.functional.cosine_similarity(embed1, embed2)[0])
            
            # Detect regions regardless of similarity score
            regions = self.detect_change_regions(images1[i], images2[i])
            
            # If we have significant regions or low similarity, record the change
            if regions or similarity < similarity_threshold:
                # Find relevant comments for this slide
                slide_comments = []
                for comment in comments:
                    comment_lower = comment.lower()
                    slide_mentions = [
                        f"slide {i+1}",
                        f"page {i+1}",
                        f"slide#{i+1}",
                        f"p{i+1}",
                        f"s{i+1}"
                    ]
                    if any(mention in comment_lower for mention in slide_mentions):
                        slide_comments.append(comment)
                
                changes.append(SlideChange(
                    slide_number=i+1,
                    similarity_score=similarity,
                    change_regions=regions,
                    associated_comments=slide_comments
                ))
        
        return VersionDiff(
            version_from=version1_path,
            version_to=version2_path,
            timestamp=datetime.now().isoformat(),
            changes=changes
        )
    
    def visualize_changes(self, diff: VersionDiff, output_dir: str):
        """Create visual representations of the changes."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert both versions to images
        images1 = self.pdf_to_images(diff.version_from)
        images2 = self.pdf_to_images(diff.version_to)
        
        # For each changed slide
        for change in diff.changes:
            slide_num = change.slide_number - 1
            
            # Create side-by-side comparison
            img1 = images1[slide_num]
            img2 = images2[slide_num].resize(img1.size)
            
            # Create comparison image
            comparison = Image.new('RGB', (img1.width * 2, img1.height))
            comparison.paste(img1, (0, 0))
            comparison.paste(img2, (img1.width, 0))
            
            # Draw boxes around changed regions
            draw = ImageDraw.Draw(comparison)
            
            colors = {
                'resize': 'blue',
                'image': 'red',
                'text': 'green'
            }
            
            for region in change.change_regions:
                color = colors.get(region.get('type', 'text'), 'red')
                
                # Draw on left image
                draw.rectangle(
                    [(region['x1'], region['y1']), 
                     (region['x2'], region['y2'])],
                    outline=color,
                    width=2
                )
                
                # Draw on right image
                draw.rectangle(
                    [(region['x1'] + img1.width, region['y1']), 
                     (region['x2'] + img1.width, region['y2'])],
                    outline=color,
                    width=2
                )
            
            # Save comparison
            output_path = os.path.join(output_dir, f'slide_{change.slide_number}_diff.png')
            comparison.save(output_path)
            
            # Save associated comments
            if change.associated_comments:
                comment_path = os.path.join(output_dir, f'slide_{change.slide_number}_comments.txt')
                with open(comment_path, 'w') as f:
                    f.write('\n'.join(change.associated_comments))

def process_revision_chain(revisions: List[Dict]) -> List[VersionDiff]:
    """Process a chain of revisions with their associated comments."""
    analyzer = PDFDiffAnalyzer()
    diffs = []
    
    # Sort revisions by date
    revisions.sort(key=lambda x: x['date'])
    
    # Process each consecutive pair of versions
    for i in range(len(revisions) - 1):
        if revisions[i].get('file_path') and revisions[i + 1].get('comments'):
            version1 = revisions[i]['file_path']
            comments = revisions[i + 1]['comments']
            
            # Look for next version
            for j in range(i + 2, len(revisions)):
                if revisions[j].get('file_path'):
                    version2 = revisions[j]['file_path']
                    diff = analyzer.analyze_versions(version1, version2, comments)
                    diffs.append(diff)
                    break
    
    return diffs

if __name__ == "__main__":
    # Example data structure
    revisions = [
        {
            'date': '2024-01-01T10:00:00',
            'file_path': 'pres_ex.pdf',
            'comments': []
        },
        {
            'date': '2024-01-01T11:00:00',
            'comments': [
                'Please make the left image smaller on slide 3',
            ],
            'file_path': None
        },
        {
            'date': '2024-01-01T12:00:00',
            'file_path': 'pres_ex_v2.pdf',
            'comments': []
        },
        {
            'date': '2024-01-01T13:00:00',
            'comments': [
                'On slide 6, update the valuation for NextEra Energy to $100',
            ],
            'file_path': None
        },
        {
            'date': '2024-01-01T14:00:00',
            'file_path': 'pres_ex_v3.pdf',
            'comments': []
        }
    ]
    
    # Process the revision chain
    diffs = process_revision_chain(revisions)
    
    # Visualize the changes
    analyzer = PDFDiffAnalyzer()
    for i, diff in enumerate(diffs):
        analyzer.visualize_changes(diff, f'diffs_version_{i+1}')
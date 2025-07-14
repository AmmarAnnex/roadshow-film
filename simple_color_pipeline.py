#!/usr/bin/env python3
"""
Simple Color Analysis Pipeline
Analyze iPhone + Sony pairs without depth estimation
"""

import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from PIL import Image
from PIL.ExifTags import TAGS

class SimpleColorPipeline:
    """Simple color analysis without MiDaS dependency"""
    
    def __init__(self):
        self.input_dir = Path("data/training_pairs")
        self.output_dir = Path("data/results/color_analysis")
        self.metadata_file = self.output_dir / "color_metadata.json"
        
        # Create directories
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata storage
        self.training_metadata = []
        print("✅ Simple color pipeline initialized (no depth estimation)")
    
    def extract_exif_metadata(self, image_path):
        """Extract EXIF metadata from image"""
        try:
            image = Image.open(image_path)
            exifdata = image.getexif()
            
            metadata = {}
            for tag_id, value in exifdata.items():
                tag = TAGS.get(tag_id, tag_id)
                metadata[tag] = value
            
            # Extract key camera settings
            camera_info = {
                "camera_make": metadata.get("Make", "Unknown"),
                "camera_model": metadata.get("Model", "Unknown"),
                "lens": metadata.get("LensModel", metadata.get("LensSpecification", "Unknown")),
                "aperture": f"f/{metadata.get('FNumber', 'Unknown')}",
                "iso": metadata.get("ISOSpeedRatings", "Unknown"),
                "focal_length": metadata.get("FocalLength", "Unknown"),
                "exposure_time": metadata.get("ExposureTime", "Unknown"),
                "datetime": metadata.get("DateTime", "Unknown")
            }
            
            return camera_info
            
        except Exception as e:
            print(f"  ⚠️ Could not extract EXIF from {image_path}: {e}")
            return {
                "camera_make": "Unknown",
                "camera_model": "Unknown", 
                "lens": "Unknown",
                "aperture": "Unknown",
                "iso": "Unknown",
                "focal_length": "Unknown",
                "exposure_time": "Unknown",
                "datetime": "Unknown"
            }
    
    def analyze_color_characteristics(self, img):
        """Analyze color characteristics of an image"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        # Calculate statistics
        bgr_means = np.mean(img, axis=(0,1))
        hsv_means = np.mean(hsv, axis=(0,1))
        lab_means = np.mean(lab, axis=(0,1))
        
        # Calculate standard deviations (color variance)
        bgr_stds = np.std(img, axis=(0,1))
        
        # Calculate brightness distribution
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness_hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Find dominant colors (simplified)
        pixels = img.reshape(-1, 3)
        dominant_color = np.mean(pixels, axis=0)
        
        return {
            "bgr_means": [float(x) for x in bgr_means],
            "hsv_means": [float(x) for x in hsv_means],
            "lab_means": [float(x) for x in lab_means],
            "bgr_stds": [float(x) for x in bgr_stds],
            "dominant_color": [float(x) for x in dominant_color],
            "brightness_mean": float(np.mean(gray)),
            "brightness_std": float(np.std(gray))
        }
    
    def analyze_image_pair(self, iphone_path, sony_path):
        """Analyze iPhone + Sony A7S3 pair with color analysis only"""
        print(f"\n🔍 Analyzing pair: {Path(iphone_path).stem} + {Path(sony_path).stem}")
        
        # Extract EXIF metadata
        iphone_exif = self.extract_exif_metadata(iphone_path)
        sony_exif = self.extract_exif_metadata(sony_path)
        
        print(f"  📱 iPhone: {iphone_exif['camera_model']}, {iphone_exif['aperture']}")
        print(f"  📷 Sony: {sony_exif['lens']}, {sony_exif['aperture']}")
        
        # Load images
        iphone_img = cv2.imread(str(iphone_path))
        sony_img = cv2.imread(str(sony_path))
        
        if iphone_img is None or sony_img is None:
            print("❌ Could not load one or both images")
            return None
        
        # Resize for comparison
        target_height = 1080
        
        # Resize iPhone image
        h, w = iphone_img.shape[:2]
        if h > target_height:
            new_width = int(w * target_height / h)
            iphone_resized = cv2.resize(iphone_img, (new_width, target_height))
        else:
            iphone_resized = iphone_img.copy()
        
        # Resize Sony image to match
        sony_resized = cv2.resize(sony_img, (iphone_resized.shape[1], iphone_resized.shape[0]))
        
        # Analyze color characteristics
        print("  🎨 Analyzing color characteristics...")
        iphone_colors = self.analyze_color_characteristics(iphone_resized)
        sony_colors = self.analyze_color_characteristics(sony_resized)
        
        # Calculate differences
        color_diff = np.mean(np.abs(iphone_resized.astype(float) - sony_resized.astype(float)))
        brightness_diff = abs(iphone_colors['brightness_mean'] - sony_colors['brightness_mean'])
        
        # Create comparison visualization
        comparison = np.hstack([iphone_resized, sony_resized])
        
        # Add labels with EXIF info
        font = cv2.FONT_HERSHEY_SIMPLEX
        iphone_label = f"iPhone: {iphone_exif['aperture']}"
        sony_label = f"Sony: {sony_exif['lens']} {sony_exif['aperture']}"
        
        cv2.putText(comparison, iphone_label, (10, 30), font, 1, (0, 255, 0), 2)
        cv2.putText(comparison, sony_label, (iphone_resized.shape[1] + 10, 30), font, 1, (0, 255, 0), 2)
        
        # Add difference info
        diff_text = f"Color Diff: {color_diff:.1f}, Brightness Diff: {brightness_diff:.1f}"
        cv2.putText(comparison, diff_text, (10, comparison.shape[0] - 20), font, 0.7, (255, 255, 255), 2)
        
        # Save analysis
        output_name = f"color_pair_{Path(iphone_path).stem}_{datetime.now().strftime('%H%M%S')}.jpg"
        output_path = self.output_dir / output_name
        cv2.imwrite(str(output_path), comparison)
        
        # Store metadata
        pair_metadata = {
            "timestamp": datetime.now().isoformat(),
            "iphone_file": str(iphone_path),
            "sony_file": str(sony_path),
            "iphone_exif": iphone_exif,
            "sony_exif": sony_exif,
            "iphone_colors": iphone_colors,
            "sony_colors": sony_colors,
            "analysis": {
                "color_difference": float(color_diff),
                "brightness_difference": float(brightness_diff),
            },
            "output_file": output_name
        }
        
        self.training_metadata.append(pair_metadata)
        
        print(f"  📊 Color difference: {color_diff:.1f}")
        print(f"  📊 Brightness difference: {brightness_diff:.1f}")
        print(f"  ✅ Saved analysis: {output_name}")
        
        return pair_metadata
    
    def process_training_folder(self):
        """Process all pairs in the training folder"""
        print("🎨 SIMPLE COLOR ANALYSIS PIPELINE")
        print("=" * 50)
        
        # Look for image pairs
        iphone_files = list(self.input_dir.glob("iphone_*.jpg")) + list(self.input_dir.glob("iphone_*.jpeg"))
        sony_files = list(self.input_dir.glob("sony_*.jpg")) + list(self.input_dir.glob("sony_*.jpeg"))
        
        if not iphone_files or not sony_files:
            print("❌ No paired images found!")
            print("📁 Expected folder structure:")
            print("   data/training_pairs/")
            print("   ├── iphone_001.jpg")
            print("   ├── sony_001.jpg")
            print("   ├── iphone_002.jpg")
            print("   └── sony_002.jpg")
            return
        
        print(f"📸 Found {len(iphone_files)} iPhone images, {len(sony_files)} Sony images")
        
        # Match pairs by number
        pairs = []
        for iphone_file in iphone_files:
            # Extract number from filename
            iphone_num = ''.join(filter(str.isdigit, iphone_file.stem))
            if not iphone_num:
                continue
                
            # Find matching Sony file
            for sony_file in sony_files:
                sony_num = ''.join(filter(str.isdigit, sony_file.stem))
                if sony_num == iphone_num:
                    pairs.append((iphone_file, sony_file, iphone_num))
                    break
        
        if not pairs:
            print("❌ No matching pairs found! Make sure files are numbered (e.g., iphone_001.jpg, sony_001.jpg)")
            return
        
        print(f"🔗 Found {len(pairs)} matching pairs")
        
        # Process each pair
        for iphone_path, sony_path, pair_num in pairs:
            self.analyze_image_pair(iphone_path, sony_path)
        
        # Save all metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(self.training_metadata, f, indent=2)
        
        print(f"\n✅ Processing complete!")
        print(f"📁 Results saved to: {self.output_dir}")
        print(f"📊 Metadata saved to: {self.metadata_file}")
        
        # Summary statistics
        if self.training_metadata:
            avg_color_diff = np.mean([m['analysis']['color_difference'] for m in self.training_metadata])
            avg_brightness_diff = np.mean([m['analysis']['brightness_difference'] for m in self.training_metadata])
            
            print(f"\n📈 SUMMARY STATISTICS:")
            print(f"   Average color difference: {avg_color_diff:.1f}")
            print(f"   Average brightness difference: {avg_brightness_diff:.1f}")
            print(f"   Total pairs processed: {len(self.training_metadata)}")


def main():
    print("🎯 SIMPLE COLOR ANALYSIS")
    print("This will analyze color differences between iPhone + Sony A7S3 pairs")
    
    pipeline = SimpleColorPipeline()
    pipeline.process_training_folder()

if __name__ == "__main__":
    main()
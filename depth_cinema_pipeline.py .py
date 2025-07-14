#!/usr/bin/env python3
"""
Paired Training Data Pipeline
Processes iPhone + Sony A7S3 pairs for training data analysis
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import sys
import os
import json
from datetime import datetime

# Add MiDaS to path
midas_path = Path("MiDaS")
if midas_path.exists():
    sys.path.append(str(midas_path))
    from midas.model_loader import default_models, load_model
else:
    print("❌ MiDaS not found. Make sure it's installed!")
    sys.exit(1)

class PairedTrainingPipeline:
    """Process iPhone + Sony A7S3 pairs for training data"""
    
    def __init__(self):
        self.input_dir = Path("data/training_pairs")
        self.output_dir = Path("data/results/training_analysis")
        self.metadata_file = self.output_dir / "training_metadata.json"
        
        # Create directories
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize MiDaS
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Using device: {self.device}")
        
        # Load MiDaS model
        model_path = "MiDaS/dpt_swin2_tiny_256.pt"
        if not Path(model_path).exists():
            print("❌ Download MiDaS weights first!")
            sys.exit(1)
            
        self.model, self.transform, self.net_w, self.net_h = load_model(
            self.device, model_path, "dpt_swin2_tiny_256", optimize=False, height=None, square=False
        )
        print("✅ MiDaS model loaded!")
        
        # Initialize metadata storage
        self.training_metadata = []
    
    def estimate_depth(self, image):
        """Estimate depth map from image"""
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform({"image": img_rgb})["image"]
        
        with torch.no_grad():
            sample = torch.from_numpy(input_tensor).to(self.device).unsqueeze(0)
            prediction = self.model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )
        
        depth_map = cv2.normalize(prediction, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        return depth_map
    
    def analyze_image_pair(self, iphone_path, sony_path, metadata):
        """Analyze iPhone + Sony A7S3 pair"""
        print(f"\n🔍 Analyzing pair: {Path(iphone_path).stem} + {Path(sony_path).stem}")
        
        # Load images
        iphone_img = cv2.imread(str(iphone_path))
        sony_img = cv2.imread(str(sony_path))
        
        if iphone_img is None or sony_img is None:
            print("❌ Could not load one or both images")
            return None
        
        # Resize for comparison (maintain aspect ratio)
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
        
        # Estimate depth for both
        print("  📏 Estimating depth for iPhone...")
        iphone_depth = self.estimate_depth(iphone_resized)
        
        print("  📏 Estimating depth for Sony...")
        sony_depth = self.estimate_depth(sony_resized)
        
        # Calculate differences
        color_diff = np.mean(np.abs(iphone_resized.astype(float) - sony_resized.astype(float)))
        depth_diff = np.mean(np.abs(iphone_depth.astype(float) - sony_depth.astype(float)))
        
        # Analyze color characteristics
        iphone_mean = np.mean(iphone_resized, axis=(0,1))
        sony_mean = np.mean(sony_resized, axis=(0,1))
        
        # Create analysis visualization
        depth_iphone_colored = cv2.applyColorMap(iphone_depth, cv2.COLORMAP_PLASMA)
        depth_sony_colored = cv2.applyColorMap(sony_depth, cv2.COLORMAP_PLASMA)
        
        # Arrange comparison
        top_row = np.hstack([iphone_resized, sony_resized])
        bottom_row = np.hstack([depth_iphone_colored, depth_sony_colored])
        comparison = np.vstack([top_row, bottom_row])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "iPhone 12 Pro Max", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, f"Sony A7S3 + {metadata.get('lens', 'Unknown')}", 
                   (iphone_resized.shape[1] + 10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "iPhone Depth", (10, iphone_resized.shape[0] + 30), 
                   font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Sony Depth", (iphone_resized.shape[1] + 10, iphone_resized.shape[0] + 30), 
                   font, 1, (255, 255, 255), 2)
        
        # Save analysis
        output_name = f"pair_{Path(iphone_path).stem}_{datetime.now().strftime('%H%M%S')}.jpg"
        output_path = self.output_dir / output_name
        cv2.imwrite(str(output_path), comparison)
        
        # Store metadata
        pair_metadata = {
            "timestamp": datetime.now().isoformat(),
            "iphone_file": str(iphone_path),
            "sony_file": str(sony_path),
            "lens": metadata.get('lens', 'Unknown'),
            "aperture": metadata.get('aperture', 'Unknown'),
            "subject": metadata.get('subject', 'Unknown'),
            "lighting": metadata.get('lighting', 'Unknown'),
            "analysis": {
                "color_difference": float(color_diff),
                "depth_difference": float(depth_diff),
                "iphone_color_mean": [float(x) for x in iphone_mean],
                "sony_color_mean": [float(x) for x in sony_mean],
            },
            "output_file": output_name
        }
        
        self.training_metadata.append(pair_metadata)
        
        print(f"  📊 Color difference: {color_diff:.1f}")
        print(f"  📊 Depth difference: {depth_diff:.1f}")
        print(f"  ✅ Saved analysis: {output_name}")
        
        return pair_metadata
    
    def process_training_folder(self):
        """Process all pairs in the training folder"""
        print("🎬 PAIRED TRAINING DATA PIPELINE")
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
            # Get metadata for this pair (you'll input this manually)
            print(f"\n📝 Enter metadata for pair {pair_num}:")
            lens = input("Lens used (e.g., 'Zeiss 50mm f/1.4'): ") or "Unknown"
            aperture = input("Aperture (e.g., 'f/1.4'): ") or "Unknown"
            subject = input("Subject (e.g., 'Plant with bokeh'): ") or "Unknown"
            lighting = input("Lighting (e.g., 'Natural patio light'): ") or "Unknown"
            
            metadata = {
                "lens": lens,
                "aperture": aperture,
                "subject": subject,
                "lighting": lighting
            }
            
            self.analyze_image_pair(iphone_path, sony_path, metadata)
        
        # Save all metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(self.training_metadata, f, indent=2)
        
        print(f"\n✅ Processing complete!")
        print(f"📁 Results saved to: {self.output_dir}")
        print(f"📊 Metadata saved to: {self.metadata_file}")
        
        # Summary statistics
        if self.training_metadata:
            avg_color_diff = np.mean([m['analysis']['color_difference'] for m in self.training_metadata])
            avg_depth_diff = np.mean([m['analysis']['depth_difference'] for m in self.training_metadata])
            
            print(f"\n📈 SUMMARY STATISTICS:")
            print(f"   Average color difference: {avg_color_diff:.1f}")
            print(f"   Average depth difference: {avg_depth_diff:.1f}")
            print(f"   Total pairs processed: {len(self.training_metadata)}")


def main():
    print("🎯 WEEKEND TRAINING DATA COLLECTION")
    print("This will analyze iPhone + Sony A7S3 pairs for training data")
    print("\n📋 SHOOTING INSTRUCTIONS:")
    print("1. Take iPhone photo (save as iphone_001.jpg, iphone_002.jpg, etc.)")
    print("2. Take Sony photo of SAME subject (save as sony_001.jpg, sony_002.jpg, etc.)")
    print("3. Try different: subjects, lenses, apertures, lighting")
    print("4. Put all photos in data/training_pairs/")
    print("5. Run this script to analyze pairs")
    
    pipeline = PairedTrainingPipeline()
    pipeline.process_training_folder()

if __name__ == "__main__":
    main()
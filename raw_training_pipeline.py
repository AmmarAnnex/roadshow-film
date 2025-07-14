#!/usr/bin/env python3
"""
RAW Training Data Pipeline
Proper analysis of DNG/ARW files with full metadata extraction
"""

import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import rawpy
import exifread
from PIL import Image
import os

class RAWTrainingPipeline:
    """Process RAW DNG/ARW files with proper metadata extraction"""
    
    def __init__(self):
        self.input_dir = Path("data/training_pairs")
        self.output_dir = Path("data/results/raw_analysis")
        self.metadata_file = self.output_dir / "raw_metadata.json"
        
        # Create directories
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata storage
        self.training_metadata = []
        print("✅ RAW pipeline initialized - preserving all sensor data")
    
    def extract_raw_metadata(self, raw_path):
        """Extract comprehensive metadata from RAW files"""
        try:
            print(f"    📊 Extracting metadata from {Path(raw_path).name}")
            
            # Open RAW file for metadata
            with open(raw_path, 'rb') as f:
                tags = exifread.process_file(f, details=True)
            
            # Extract key metadata
            metadata = {}
            
            # Basic camera info
            metadata['make'] = str(tags.get('Image Make', 'Unknown')).strip()
            metadata['model'] = str(tags.get('Image Model', 'Unknown')).strip()
            metadata['datetime'] = str(tags.get('EXIF DateTimeOriginal', 'Unknown'))
            
            # Lens information
            metadata['lens_make'] = str(tags.get('EXIF LensMake', 'Unknown')).strip()
            metadata['lens_model'] = str(tags.get('EXIF LensModel', 'Unknown')).strip()
            metadata['lens_spec'] = str(tags.get('EXIF LensSpecification', 'Unknown'))
            
            # Camera settings
            fnumber = tags.get('EXIF FNumber')
            if fnumber:
                metadata['aperture'] = f"f/{float(fnumber.values[0]):.1f}"
            else:
                metadata['aperture'] = 'Unknown'
            
            focal_length = tags.get('EXIF FocalLength')
            if focal_length:
                metadata['focal_length'] = f"{float(focal_length.values[0])}mm"
            else:
                metadata['focal_length'] = 'Unknown'
            
            iso = tags.get('EXIF ISOSpeedRatings')
            if iso:
                metadata['iso'] = int(iso.values[0])
            else:
                metadata['iso'] = 'Unknown'
            
            exposure_time = tags.get('EXIF ExposureTime')
            if exposure_time:
                metadata['shutter_speed'] = f"1/{int(1/float(exposure_time.values[0]))}"
            else:
                metadata['shutter_speed'] = 'Unknown'
            
            # Sony specific metadata
            if 'sony' in raw_path.lower() or metadata['make'].lower() == 'sony':
                metadata['camera_type'] = 'Sony A7S3'
                # Sony specific tags
                wb_mode = tags.get('EXIF WhiteBalance')
                if wb_mode:
                    metadata['white_balance'] = str(wb_mode)
                    
            # iPhone specific metadata  
            elif 'iphone' in raw_path.lower() or 'apple' in metadata['make'].lower():
                metadata['camera_type'] = 'iPhone 12 Pro Max'
                # iPhone DNG often has different tags
                
            return metadata
            
        except Exception as e:
            print(f"    ⚠️ Metadata extraction error for {raw_path}: {e}")
            return {
                'make': 'Unknown',
                'model': 'Unknown', 
                'camera_type': 'Unknown',
                'lens_model': 'Unknown',
                'aperture': 'Unknown',
                'focal_length': 'Unknown',
                'iso': 'Unknown',
                'shutter_speed': 'Unknown'
            }
    
    def process_raw_image(self, raw_path, target_size=(2048, 1365)):
        """Process RAW file to RGB array while preserving color depth"""
        try:
            print(f"    🎨 Processing RAW: {Path(raw_path).name}")
            
            # Load RAW file
            with rawpy.imread(str(raw_path)) as raw:
                # Process with minimal manipulation to preserve sensor characteristics
                rgb = raw.postprocess(
                    use_camera_wb=True,           # Use camera white balance
                    use_auto_wb=False,           # Don't auto-adjust
                    output_color=rawpy.ColorSpace.sRGB,  # sRGB output
                    output_bps=16,               # 16-bit output (preserve depth)
                    demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,  # High quality
                    no_auto_bright=True,         # Preserve exposure
                    gamma=(1, 1),               # Linear gamma (no curve)
                    brightness=1.0,              # No brightness adjustment
                )
            
            # Convert from 16-bit to 8-bit for analysis (but keep 16-bit data)
            rgb_8bit = (rgb / 256).astype(np.uint8)
            
            # Resize for comparison while maintaining aspect ratio
            h, w = rgb_8bit.shape[:2]
            if h > target_size[1]:
                scale = target_size[1] / h
                new_w = int(w * scale)
                new_h = target_size[1]
                rgb_8bit = cv2.resize(rgb_8bit, (new_w, new_h))
            
            # Convert RGB to BGR for OpenCV
            bgr_8bit = cv2.cvtColor(rgb_8bit, cv2.COLOR_RGB2BGR)
            
            return bgr_8bit, rgb  # Return both 8-bit for display and 16-bit for analysis
            
        except Exception as e:
            print(f"    ❌ RAW processing error for {raw_path}: {e}")
            return None, None
    
    def analyze_raw_characteristics(self, rgb_16bit):
        """Analyze RAW characteristics preserving bit depth"""
        # Convert to float for analysis
        img_float = rgb_16bit.astype(np.float64) / 65535.0  # Normalize 16-bit
        
        # Calculate statistics in linear space
        channel_means = np.mean(img_float, axis=(0,1))
        channel_stds = np.std(img_float, axis=(0,1))
        
        # Dynamic range analysis
        min_vals = np.min(img_float, axis=(0,1))
        max_vals = np.max(img_float, axis=(0,1))
        dynamic_range = max_vals - min_vals
        
        # Histogram analysis per channel
        histograms = []
        for channel in range(3):
            hist, _ = np.histogram(img_float[:,:,channel], bins=256, range=(0, 1))
            histograms.append(hist.tolist())
        
        # Shadow/highlight analysis
        shadow_threshold = 0.1
        highlight_threshold = 0.9
        
        shadow_pixels = np.sum(img_float < shadow_threshold) / img_float.size
        highlight_pixels = np.sum(img_float > highlight_threshold) / img_float.size
        
        return {
            'bit_depth': '16-bit',
            'channel_means': [float(x) for x in channel_means],
            'channel_stds': [float(x) for x in channel_stds],
            'dynamic_range': [float(x) for x in dynamic_range],
            'min_values': [float(x) for x in min_vals],
            'max_values': [float(x) for x in max_vals],
            'shadow_pixel_ratio': float(shadow_pixels),
            'highlight_pixel_ratio': float(highlight_pixels),
            'histograms': histograms
        }
    
    def analyze_raw_pair(self, iphone_path, sony_path):
        """Analyze RAW iPhone DNG + Sony ARW pair"""
        print(f"\n🎬 Analyzing RAW pair: {Path(iphone_path).stem} + {Path(sony_path).stem}")
        
        # Extract metadata from both files
        iphone_meta = self.extract_raw_metadata(iphone_path)
        sony_meta = self.extract_raw_metadata(sony_path)
        
        print(f"  📱 iPhone: {iphone_meta['aperture']}, ISO {iphone_meta['iso']}")
        print(f"  📷 Sony: {sony_meta['lens_model']} {sony_meta['aperture']}, ISO {sony_meta['iso']}")
        
        # Process RAW images
        iphone_8bit, iphone_16bit = self.process_raw_image(iphone_path)
        sony_8bit, sony_16bit = self.process_raw_image(sony_path)
        
        if iphone_8bit is None or sony_8bit is None:
            print("  ❌ Failed to process RAW files")
            return None
        
        # Resize Sony to match iPhone for comparison
        iphone_h, iphone_w = iphone_8bit.shape[:2]
        sony_resized = cv2.resize(sony_8bit, (iphone_w, iphone_h))
        
        # Analyze RAW characteristics
        print("  📊 Analyzing 16-bit characteristics...")
        iphone_chars = self.analyze_raw_characteristics(iphone_16bit)
        sony_chars = self.analyze_raw_characteristics(sony_16bit)
        
        # Calculate differences
        color_diff = np.mean(np.abs(iphone_8bit.astype(float) - sony_resized.astype(float)))
        
        # Dynamic range comparison
        iphone_dr = np.mean(iphone_chars['dynamic_range'])
        sony_dr = np.mean(sony_chars['dynamic_range'])
        dr_diff = abs(iphone_dr - sony_dr)
        
        # Create comparison visualization
        comparison = np.hstack([iphone_8bit, sony_resized])
        
        # Add detailed labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        iphone_label = f"iPhone: {iphone_meta['aperture']}, ISO {iphone_meta['iso']}"
        sony_label = f"Sony: {sony_meta['lens_model']} {sony_meta['aperture']}, ISO {sony_meta['iso']}"
        
        cv2.putText(comparison, iphone_label, (10, 30), font, 0.8, (0, 255, 0), 2)
        cv2.putText(comparison, sony_label, (iphone_w + 10, 30), font, 0.8, (0, 255, 0), 2)
        
        # Add analysis info
        analysis_text = f"Color Diff: {color_diff:.1f}, DR Diff: {dr_diff:.3f}"
        cv2.putText(comparison, analysis_text, (10, comparison.shape[0] - 40), font, 0.7, (255, 255, 255), 2)
        
        bit_depth_text = f"16-bit RAW Analysis - iPhone DR: {iphone_dr:.3f}, Sony DR: {sony_dr:.3f}"
        cv2.putText(comparison, bit_depth_text, (10, comparison.shape[0] - 10), font, 0.6, (255, 255, 255), 2)
        
        # Save analysis
        output_name = f"raw_pair_{Path(iphone_path).stem}_{datetime.now().strftime('%H%M%S')}.jpg"
        output_path = self.output_dir / output_name
        cv2.imwrite(str(output_path), comparison)
        
        # Store comprehensive metadata
        pair_metadata = {
            "timestamp": datetime.now().isoformat(),
            "iphone_file": str(iphone_path),
            "sony_file": str(sony_path),
            "iphone_metadata": iphone_meta,
            "sony_metadata": sony_meta,
            "iphone_characteristics": iphone_chars,
            "sony_characteristics": sony_chars,
            "analysis": {
                "color_difference": float(color_diff),
                "dynamic_range_difference": float(dr_diff),
                "iphone_dynamic_range": float(iphone_dr),
                "sony_dynamic_range": float(sony_dr),
            },
            "output_file": output_name
        }
        
        self.training_metadata.append(pair_metadata)
        
        print(f"  📊 Color difference: {color_diff:.1f}")
        print(f"  📊 Dynamic range diff: {dr_diff:.3f}")
        print(f"  📊 iPhone DR: {iphone_dr:.3f}, Sony DR: {sony_dr:.3f}")
        print(f"  ✅ Saved analysis: {output_name}")
        
        return pair_metadata
    
    def process_raw_training_folder(self):
        """Process all RAW pairs in the training folder"""
        print("🎬 RAW TRAINING DATA PIPELINE")
        print("=" * 50)
        print("Processing DNG/ARW files with full bit-depth preservation")
        
        # Look for RAW files
        iphone_files = list(self.input_dir.glob("*.DNG")) + list(self.input_dir.glob("*.dng"))
        sony_files = list(self.input_dir.glob("*.ARW")) + list(self.input_dir.glob("*.arw"))
        
        if not iphone_files or not sony_files:
            print("❌ No RAW files found!")
            print("📁 Expected files in data/training_pairs/:")
            print("   ├── IMG_0526.DNG")
            print("   ├── DSC01229.ARW")
            print("   └── ... (more RAW files)")
            return
        
        print(f"📸 Found {len(iphone_files)} iPhone DNG files, {len(sony_files)} Sony ARW files")
        
        # Sort files for pairing
        iphone_files.sort()
        sony_files.sort()
        
        # Pair files by position (assumes they were shot in sequence)
        min_count = min(len(iphone_files), len(sony_files))
        
        print(f"🔗 Will process {min_count} pairs")
        
        # Process each pair
        for i in range(min_count):
            iphone_path = iphone_files[i]
            sony_path = sony_files[i]
            
            self.analyze_raw_pair(iphone_path, sony_path)
        
        # Save all metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(self.training_metadata, f, indent=2)
        
        print(f"\n✅ RAW processing complete!")
        print(f"📁 Results saved to: {self.output_dir}")
        print(f"📊 Metadata saved to: {self.metadata_file}")
        
        # Summary statistics
        if self.training_metadata:
            avg_color_diff = np.mean([m['analysis']['color_difference'] for m in self.training_metadata])
            avg_dr_diff = np.mean([m['analysis']['dynamic_range_difference'] for m in self.training_metadata])
            
            print(f"\n📈 RAW ANALYSIS SUMMARY:")
            print(f"   Average color difference: {avg_color_diff:.1f}")
            print(f"   Average dynamic range difference: {avg_dr_diff:.3f}")
            print(f"   Total RAW pairs processed: {len(self.training_metadata)}")
            print(f"   Data preserved: 16-bit linear RAW characteristics")


def main():
    print("🎯 RAW TRAINING DATA COLLECTION")
    print("Analyzing iPhone DNG + Sony ARW files with full metadata")
    print("Preserving 16-bit color depth and sensor characteristics")
    
    try:
        pipeline = RAWTrainingPipeline()
        pipeline.process_raw_training_folder()
        
        print("\n🎯 WHAT THIS GIVES YOU:")
        print("✅ Complete aperture, lens, and camera settings")
        print("✅ 16-bit color depth analysis (not 8-bit JPEG)")
        print("✅ True dynamic range measurements")
        print("✅ Sensor-level characteristics before processing")
        print("✅ Perfect training data for your iPhone→Cinema vision")
        
    except ImportError as e:
        if 'rawpy' in str(e):
            print("❌ rawpy not installed. Install with:")
            print("   pip install rawpy")
        elif 'exifread' in str(e):
            print("❌ ExifRead not installed. Install with:")
            print("   pip install ExifRead")
        else:
            print(f"❌ Import error: {e}")

if __name__ == "__main__":
    main()
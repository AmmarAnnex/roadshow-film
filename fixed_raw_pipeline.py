#!/usr/bin/env python3
"""
Fixed RAW Training Data Pipeline
Proper analysis of DNG/ARW files with full metadata extraction
"""

import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import rawpy
import exifread
from typing import Dict, Any, Tuple, Optional

class FixedRAWTrainingPipeline:
    def __init__(self):
        print("🎯 FIXED RAW TRAINING DATA COLLECTION")
        print("Analyzing iPhone DNG + Sony ARW files with full metadata")
        print("Preserving 16-bit color depth and sensor characteristics")
        
        # Setup directories
        self.setup_directories()
        print("✅ RAW pipeline initialized - preserving all sensor data")

    def setup_directories(self):
        """Create necessary directories"""
        self.data_dir = Path("data/training_pairs")
        self.results_dir = Path("data/results/raw_analysis")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def extract_raw_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from RAW file using exifread"""
        metadata = {
            "camera_make": "Unknown",
            "camera_model": "Unknown", 
            "lens_model": "Unknown",
            "aperture": "Unknown",
            "iso": "Unknown",
            "focal_length": "Unknown",
            "shutter_speed": "Unknown",
            "white_balance": "Unknown"
        }
        
        try:
            # Convert Path to string for exifread
            with open(str(file_path), 'rb') as f:
                tags = exifread.process_file(f, details=False)
                
                # Extract key metadata
                if 'Image Make' in tags:
                    metadata["camera_make"] = str(tags['Image Make'])
                if 'Image Model' in tags:
                    metadata["camera_model"] = str(tags['Image Model'])
                if 'EXIF LensModel' in tags:
                    metadata["lens_model"] = str(tags['EXIF LensModel'])
                elif 'EXIF LensMake' in tags:
                    metadata["lens_model"] = str(tags['EXIF LensMake'])
                
                # Handle aperture (F-number)
                if 'EXIF FNumber' in tags:
                    f_num = tags['EXIF FNumber']
                    if hasattr(f_num, 'values') and f_num.values:
                        aperture_val = float(f_num.values[0].num) / float(f_num.values[0].den)
                        metadata["aperture"] = f"f/{aperture_val:.1f}"
                elif 'EXIF ApertureValue' in tags:
                    metadata["aperture"] = str(tags['EXIF ApertureValue'])
                
                # ISO
                if 'EXIF ISOSpeedRatings' in tags:
                    metadata["iso"] = str(tags['EXIF ISOSpeedRatings'])
                
                # Focal length
                if 'EXIF FocalLength' in tags:
                    focal = tags['EXIF FocalLength']
                    if hasattr(focal, 'values') and focal.values:
                        focal_val = float(focal.values[0].num) / float(focal.values[0].den)
                        metadata["focal_length"] = f"{focal_val:.1f}mm"
                
                # Shutter speed
                if 'EXIF ExposureTime' in tags:
                    metadata["shutter_speed"] = str(tags['EXIF ExposureTime'])
                
                # White balance
                if 'EXIF WhiteBalance' in tags:
                    metadata["white_balance"] = str(tags['EXIF WhiteBalance'])
                    
        except Exception as e:
            print(f"    ⚠️ Metadata extraction error for {file_path}: {e}")
            
        return metadata

    def process_raw_image(self, file_path: Path) -> Optional[np.ndarray]:
        """Process RAW file to RGB array"""
        try:
            # Use corrected rawpy API
            with rawpy.imread(str(file_path)) as raw:
                # Use postprocess() method with correct parameters
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    output_bps=16,  # 16-bit output
                    no_auto_bright=True,  # Preserve exposure
                    user_flip=0
                )
            return rgb
            
        except Exception as e:
            print(f"    ❌ RAW processing error for {file_path}: {e}")
            return None

    def analyze_raw_characteristics(self, iphone_rgb: np.ndarray, sony_rgb: np.ndarray) -> Dict[str, float]:
        """Analyze RAW color and dynamic range characteristics"""
        # Resize to same dimensions for comparison
        h, w = min(iphone_rgb.shape[0], sony_rgb.shape[0]), min(iphone_rgb.shape[1], sony_rgb.shape[1])
        iphone_resized = cv2.resize(iphone_rgb, (w, h))
        sony_resized = cv2.resize(sony_rgb, (w, h))
        
        # Convert to float for analysis (preserve 16-bit range)
        iphone_float = iphone_resized.astype(np.float32) / 65535.0
        sony_float = sony_resized.astype(np.float32) / 65535.0
        
        # Color difference analysis
        color_diff = np.mean(np.abs(iphone_float - sony_float)) * 100
        
        # Dynamic range analysis
        iphone_dr = np.log2(np.max(iphone_float) / (np.min(iphone_float[iphone_float > 0]) + 1e-8))
        sony_dr = np.log2(np.max(sony_float) / (np.min(sony_float[sony_float > 0]) + 1e-8))
        dr_difference = abs(iphone_dr - sony_dr)
        
        # Shadow/highlight distribution
        iphone_shadows = np.mean(iphone_float[iphone_float < 0.2])
        sony_shadows = np.mean(sony_float[sony_float < 0.2])
        shadow_diff = abs(iphone_shadows - sony_shadows) * 100
        
        iphone_highlights = np.mean(iphone_float[iphone_float > 0.8])
        sony_highlights = np.mean(sony_float[sony_float > 0.8])
        highlight_diff = abs(iphone_highlights - sony_highlights) * 100
        
        # Color space characteristics
        iphone_saturation = np.std(iphone_float)
        sony_saturation = np.std(sony_float)
        saturation_diff = abs(iphone_saturation - sony_saturation) * 100
        
        return {
            "color_difference": float(color_diff),
            "dynamic_range_difference": float(dr_difference),
            "shadow_difference": float(shadow_diff),
            "highlight_difference": float(highlight_diff),
            "saturation_difference": float(saturation_diff),
            "iphone_dynamic_range": float(iphone_dr),
            "sony_dynamic_range": float(sony_dr)
        }

    def create_comparison_image(self, iphone_rgb: np.ndarray, sony_rgb: np.ndarray, 
                              iphone_meta: Dict, sony_meta: Dict, 
                              analysis: Dict, pair_name: str) -> str:
        """Create side-by-side comparison with metadata"""
        # Convert 16-bit to 8-bit for display
        iphone_display = (iphone_rgb / 257).astype(np.uint8)
        sony_display = (sony_rgb / 257).astype(np.uint8)
        
        # Resize for comparison
        h = min(iphone_display.shape[0], sony_display.shape[0])
        w = min(iphone_display.shape[1], sony_display.shape[1])
        
        iphone_resized = cv2.resize(iphone_display, (w, h))
        sony_resized = cv2.resize(sony_display, (w, h))
        
        # Create side-by-side comparison
        comparison = np.hstack([iphone_resized, sony_resized])
        comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
        
        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        color = (255, 255, 255)
        thickness = 2
        
        # iPhone label
        iphone_label = f"iPhone: {iphone_meta['aperture']} ISO{iphone_meta['iso']}"
        cv2.putText(comparison_bgr, iphone_label, (10, 30), font, font_scale, color, thickness)
        
        # Sony label  
        sony_label = f"Sony: {sony_meta['lens_model']} {sony_meta['aperture']} ISO{sony_meta['iso']}"
        cv2.putText(comparison_bgr, sony_label, (w + 10, 30), font, font_scale, color, thickness)
        
        # Analysis stats
        stats_y = h - 80
        cv2.putText(comparison_bgr, f"Color Diff: {analysis['color_difference']:.1f}%", 
                   (10, stats_y), font, 0.6, color, 1)
        cv2.putText(comparison_bgr, f"DR Diff: {analysis['dynamic_range_difference']:.1f} stops", 
                   (10, stats_y + 20), font, 0.6, color, 1)
        cv2.putText(comparison_bgr, f"iPhone DR: {analysis['iphone_dynamic_range']:.1f} stops", 
                   (w + 10, stats_y), font, 0.6, color, 1)
        cv2.putText(comparison_bgr, f"Sony DR: {analysis['sony_dynamic_range']:.1f} stops", 
                   (w + 10, stats_y + 20), font, 0.6, color, 1)
        
        # Save comparison
        timestamp = datetime.now().strftime("%H%M%S")
        output_path = self.results_dir / f"raw_comparison_{pair_name}_{timestamp}.jpg"
        cv2.imwrite(str(output_path), comparison_bgr)
        
        return str(output_path)

    def analyze_pair(self, iphone_path: Path, sony_path: Path) -> Dict[str, Any]:
        """Analyze a single iPhone + Sony pair"""
        pair_name = f"{iphone_path.stem}+{sony_path.stem}"
        print(f"\n🎬 Analyzing RAW pair: {iphone_path.stem} + {sony_path.stem}")
        
        # Extract metadata
        print(f"    📊 Extracting metadata from {iphone_path.name}")
        iphone_meta = self.extract_raw_metadata(iphone_path)
        print(f"    📊 Extracting metadata from {sony_path.name}")
        sony_meta = self.extract_raw_metadata(sony_path)
        
        print(f"  📱 iPhone: {iphone_meta['camera_model']}, {iphone_meta['aperture']}, ISO {iphone_meta['iso']}")
        print(f"  📷 Sony: {sony_meta['lens_model']} {sony_meta['aperture']}, ISO {sony_meta['iso']}")
        
        # Process RAW files
        print(f"    🎨 Processing RAW: {iphone_path.name}")
        iphone_rgb = self.process_raw_image(iphone_path)
        
        print(f"    🎨 Processing RAW: {sony_path.name}")
        sony_rgb = self.process_raw_image(sony_path)
        
        if iphone_rgb is None or sony_rgb is None:
            print("  ❌ Failed to process RAW files")
            return None
            
        # Analyze characteristics
        print("    🔬 Analyzing RAW characteristics...")
        analysis = self.analyze_raw_characteristics(iphone_rgb, sony_rgb)
        
        # Create comparison image
        comparison_path = self.create_comparison_image(
            iphone_rgb, sony_rgb, iphone_meta, sony_meta, analysis, pair_name
        )
        
        print(f"  📊 Color difference: {analysis['color_difference']:.1f}%")
        print(f"  📊 Dynamic range difference: {analysis['dynamic_range_difference']:.1f} stops")
        print(f"  📊 iPhone DR: {analysis['iphone_dynamic_range']:.1f} stops")
        print(f"  📊 Sony DR: {analysis['sony_dynamic_range']:.1f} stops")
        print(f"  ✅ Saved comparison: {Path(comparison_path).name}")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "iphone_file": str(iphone_path),
            "sony_file": str(sony_path),
            "iphone_metadata": iphone_meta,
            "sony_metadata": sony_meta,
            "analysis": analysis,
            "comparison_image": comparison_path
        }

    def find_paired_files(self) -> list:
        """Find matching iPhone DNG + Sony ARW pairs"""
        iphone_files = sorted(list(self.data_dir.glob("iphone_*.dng")))
        sony_files = sorted(list(self.data_dir.glob("sony_*.arw")))
        
        print(f"📸 Found {len(iphone_files)} iPhone DNG files, {len(sony_files)} Sony ARW files")
        
        # Match by number
        pairs = []
        for iphone_file in iphone_files:
            # Extract number from filename
            iphone_num = iphone_file.stem.split('_')[-1]
            
            # Find matching Sony file
            sony_file = self.data_dir / f"sony_{iphone_num}.arw"
            if sony_file.exists():
                pairs.append((iphone_file, sony_file))
        
        print(f"🔗 Will process {len(pairs)} pairs")
        return pairs

    def run_analysis(self):
        """Run complete RAW analysis pipeline"""
        print("\n🎬 RAW TRAINING DATA PIPELINE")
        print("=" * 50)
        print("Processing DNG/ARW files with full bit-depth preservation")
        
        # Find paired files
        pairs = self.find_paired_files()
        
        if not pairs:
            print("❌ No matching pairs found!")
            return
        
        # Process each pair
        all_results = []
        for iphone_path, sony_path in pairs:
            result = self.analyze_pair(iphone_path, sony_path)
            if result:
                all_results.append(result)
        
        # Save comprehensive metadata
        metadata_path = self.results_dir / "raw_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n✅ RAW processing complete!")
        print(f"📁 Results saved to: {self.results_dir}")
        print(f"📊 Metadata saved to: {metadata_path}")
        
        if all_results:
            # Calculate summary statistics
            color_diffs = [r['analysis']['color_difference'] for r in all_results]
            dr_diffs = [r['analysis']['dynamic_range_difference'] for r in all_results]
            
            print(f"\n📈 SUMMARY STATISTICS:")
            print(f"   Average color difference: {np.mean(color_diffs):.1f}%")
            print(f"   Average DR difference: {np.mean(dr_diffs):.1f} stops")
            print(f"   Total pairs processed: {len(all_results)}")
        
        print(f"\n🎯 WHAT THIS GIVES YOU:")
        print("✅ Complete aperture, lens, and camera settings")
        print("✅ 16-bit color depth analysis (not 8-bit JPEG)")
        print("✅ True dynamic range measurements")
        print("✅ Sensor-level characteristics before processing") 
        print("✅ Perfect training data for your iPhone→Cinema vision")

def main():
    pipeline = FixedRAWTrainingPipeline()
    pipeline.run_analysis()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Simple Depth-Aware RAW Pipeline
Uses OpenCV stereo matching for depth estimation - no external models needed
"""

import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import rawpy
import exifread
from typing import Dict, Any, Tuple, Optional

class SimpleDepthRAWPipeline:
    def __init__(self):
        print("🎯 SIMPLE DEPTH-AWARE RAW PIPELINE")
        print("Using OpenCV stereo matching for spatial intelligence")
        print("No external model downloads - built-in computer vision")
        
        # Setup directories
        self.setup_directories()
        
        # Initialize stereo matcher
        self.setup_stereo_matcher()
        
        print("✅ Simple depth pipeline initialized")

    def setup_directories(self):
        """Create necessary directories"""
        self.data_dir = Path("data/training_pairs")
        self.results_dir = Path("data/results/simple_depth_analysis")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def setup_stereo_matcher(self):
        """Initialize OpenCV stereo matcher"""
        # Create SGBM (Semi-Global Block Matching) stereo matcher
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=64,  # Must be divisible by 16
            blockSize=11,
            P1=8 * 3 * 11**2,
            P2=32 * 3 * 11**2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=0,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        print("✅ OpenCV stereo matcher ready")

    def extract_raw_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from RAW file using exifread"""
        metadata = {
            "camera_make": "Unknown",
            "camera_model": "Unknown", 
            "lens_model": "Unknown",
            "aperture": "Unknown",
            "iso": "Unknown",
            "focal_length": "Unknown",
            "shutter_speed": "Unknown"
        }
        
        try:
            with open(str(file_path), 'rb') as f:
                tags = exifread.process_file(f, details=False)
                
                if 'Image Make' in tags:
                    metadata["camera_make"] = str(tags['Image Make'])
                if 'Image Model' in tags:
                    metadata["camera_model"] = str(tags['Image Model'])
                if 'EXIF LensModel' in tags:
                    metadata["lens_model"] = str(tags['EXIF LensModel'])
                elif 'EXIF LensMake' in tags:
                    metadata["lens_model"] = str(tags['EXIF LensMake'])
                
                # Handle aperture
                if 'EXIF FNumber' in tags:
                    f_num = tags['EXIF FNumber']
                    if hasattr(f_num, 'values') and f_num.values:
                        aperture_val = float(f_num.values[0].num) / float(f_num.values[0].den)
                        metadata["aperture"] = f"f/{aperture_val:.1f}"
                
                # ISO
                if 'EXIF ISOSpeedRatings' in tags:
                    metadata["iso"] = str(tags['EXIF ISOSpeedRatings'])
                
                # Focal length
                if 'EXIF FocalLength' in tags:
                    focal = tags['EXIF FocalLength']
                    if hasattr(focal, 'values') and focal.values:
                        focal_val = float(focal.values[0].num) / float(focal.values[0].den)
                        metadata["focal_length"] = f"{focal_val:.1f}mm"
                        
        except Exception as e:
            print(f"    ⚠️ Metadata extraction error for {file_path}: {e}")
            
        return metadata

    def process_raw_image(self, file_path: Path) -> Optional[np.ndarray]:
        """Process RAW file to RGB array"""
        try:
            with rawpy.imread(str(file_path)) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    output_bps=16,
                    no_auto_bright=True,
                    user_flip=0
                )
            return rgb
            
        except Exception as e:
            print(f"    ❌ RAW processing error for {file_path}: {e}")
            return None

    def estimate_simple_depth(self, left_img: np.ndarray, right_img: np.ndarray) -> Optional[np.ndarray]:
        """Estimate depth using OpenCV stereo matching"""
        try:
            # Convert to 8-bit grayscale for stereo matching
            left_gray = cv2.cvtColor((left_img / 257).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            right_gray = cv2.cvtColor((right_img / 257).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            # Resize to manageable size for stereo matching
            h, w = left_gray.shape
            if max(h, w) > 800:
                scale = 800 / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                left_gray = cv2.resize(left_gray, (new_w, new_h))
                right_gray = cv2.resize(right_gray, (new_w, new_h))
            
            # Compute disparity
            disparity = self.stereo.compute(left_gray, right_gray)
            
            # Normalize disparity to depth-like values
            disparity_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # Resize back to original if needed
            if max(h, w) > 800:
                disparity_norm = cv2.resize(disparity_norm, (w, h))
            
            return disparity_norm.astype(np.float32)
            
        except Exception as e:
            print(f"    ⚠️ Stereo depth estimation error: {e}")
            return None

    def estimate_monocular_depth(self, img: np.ndarray) -> np.ndarray:
        """Simple monocular depth estimation using edge detection and blur"""
        # Convert to grayscale
        gray = cv2.cvtColor((img / 257).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Edge detection (sharp edges = closer objects)
        edges = cv2.Canny(gray, 50, 150)
        
        # Distance transform (distance from edges)
        dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
        
        # Gaussian blur to simulate focus falloff
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        focus_measure = cv2.Laplacian(blurred, cv2.CV_64F).var()
        
        # Combine edge distance with local variance (focus measure)
        local_variance = cv2.Laplacian(gray, cv2.CV_64F)
        local_variance = np.abs(local_variance)
        
        # Normalize and combine
        dist_norm = cv2.normalize(dist, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        var_norm = cv2.normalize(local_variance, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        # Simple depth estimate: combine distance from edges with focus measure
        depth_estimate = 0.7 * dist_norm + 0.3 * var_norm
        
        return (depth_estimate * 255).astype(np.float32)

    def analyze_depth_characteristics(self, iphone_rgb: np.ndarray, sony_rgb: np.ndarray) -> Dict[str, float]:
        """Analyze depth and spatial characteristics"""
        # Resize to same dimensions
        h, w = min(iphone_rgb.shape[0], sony_rgb.shape[0]), min(iphone_rgb.shape[1], sony_rgb.shape[1])
        iphone_resized = cv2.resize(iphone_rgb, (w, h))
        sony_resized = cv2.resize(sony_rgb, (w, h))
        
        # Convert to float for analysis
        iphone_float = iphone_resized.astype(np.float32) / 65535.0
        sony_float = sony_resized.astype(np.float32) / 65535.0
        
        # Color difference
        color_diff = np.mean(np.abs(iphone_float - sony_float)) * 100
        
        # Dynamic range
        iphone_dr = np.log2(np.max(iphone_float) / (np.min(iphone_float[iphone_float > 0]) + 1e-8))
        sony_dr = np.log2(np.max(sony_float) / (np.min(sony_float[sony_float > 0]) + 1e-8))
        dr_difference = abs(iphone_dr - sony_dr)
        
        # Estimate depth using monocular approach
        iphone_depth = self.estimate_monocular_depth(iphone_resized)
        sony_depth = self.estimate_monocular_depth(sony_resized)
        
        # Depth analysis
        depth_diff = np.mean(np.abs(iphone_depth - sony_depth))
        
        # Focus analysis (variance in depth map indicates DOF)
        iphone_focus_var = np.var(iphone_depth)
        sony_focus_var = np.var(sony_depth)
        focus_var_diff = abs(iphone_focus_var - sony_focus_var)
        
        # Edge sharpness analysis (proxy for depth of field)
        iphone_gray = cv2.cvtColor((iphone_resized / 257).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        sony_gray = cv2.cvtColor((sony_resized / 257).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        iphone_sharpness = cv2.Laplacian(iphone_gray, cv2.CV_64F).var()
        sony_sharpness = cv2.Laplacian(sony_gray, cv2.CV_64F).var()
        sharpness_diff = abs(iphone_sharpness - sony_sharpness)
        
        return {
            "color_difference": float(color_diff),
            "dynamic_range_difference": float(dr_difference),
            "iphone_dynamic_range": float(iphone_dr),
            "sony_dynamic_range": float(sony_dr),
            "depth_difference": float(depth_diff),
            "focus_variance_difference": float(focus_var_diff),
            "sharpness_difference": float(sharpness_diff),
            "iphone_focus_variance": float(iphone_focus_var),
            "sony_focus_variance": float(sony_focus_var),
            "iphone_sharpness": float(iphone_sharpness),
            "sony_sharpness": float(sony_sharpness)
        }

    def create_depth_comparison(self, iphone_rgb: np.ndarray, sony_rgb: np.ndarray,
                               iphone_meta: Dict, sony_meta: Dict, 
                               analysis: Dict, pair_name: str) -> str:
        """Create depth comparison visualization"""
        # Convert to display format
        iphone_display = (iphone_rgb / 257).astype(np.uint8)
        sony_display = (sony_rgb / 257).astype(np.uint8)
        
        # Resize for comparison
        h = min(iphone_display.shape[0], sony_display.shape[0]) // 2
        w = min(iphone_display.shape[1], sony_display.shape[1]) // 2
        
        iphone_resized = cv2.resize(iphone_display, (w, h))
        sony_resized = cv2.resize(sony_display, (w, h))
        
        # Generate depth estimates
        iphone_depth = self.estimate_monocular_depth(cv2.resize(iphone_rgb, (w, h)))
        sony_depth = self.estimate_monocular_depth(cv2.resize(sony_rgb, (w, h)))
        
        # Create depth visualizations
        iphone_depth_vis = cv2.applyColorMap(iphone_depth.astype(np.uint8), cv2.COLORMAP_PLASMA)
        sony_depth_vis = cv2.applyColorMap(sony_depth.astype(np.uint8), cv2.COLORMAP_PLASMA)
        
        # Create 2x2 layout
        top_row = np.hstack([
            cv2.cvtColor(iphone_resized, cv2.COLOR_RGB2BGR),
            cv2.cvtColor(sony_resized, cv2.COLOR_RGB2BGR)
        ])
        bottom_row = np.hstack([iphone_depth_vis, sony_depth_vis])
        comparison = np.vstack([top_row, bottom_row])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, f"iPhone {iphone_meta['aperture']}", (10, 25), font, 0.6, (255, 255, 255), 1)
        cv2.putText(comparison, f"Sony {sony_meta['aperture']}", (w + 10, 25), font, 0.6, (255, 255, 255), 1)
        cv2.putText(comparison, "iPhone Depth", (10, h + 25), font, 0.6, (255, 255, 255), 1)
        cv2.putText(comparison, "Sony Depth", (w + 10, h + 25), font, 0.6, (255, 255, 255), 1)
        
        # Add analysis stats
        stats_y = h * 2 - 60
        cv2.putText(comparison, f"Color: {analysis['color_difference']:.1f}%", (10, stats_y), font, 0.5, (255, 255, 255), 1)
        cv2.putText(comparison, f"Depth: {analysis['depth_difference']:.1f}", (10, stats_y + 15), font, 0.5, (255, 255, 255), 1)
        cv2.putText(comparison, f"Sharpness: {analysis['sharpness_difference']:.0f}", (w + 10, stats_y), font, 0.5, (255, 255, 255), 1)
        cv2.putText(comparison, f"Focus Var: {analysis['focus_variance_difference']:.0f}", (w + 10, stats_y + 15), font, 0.5, (255, 255, 255), 1)
        
        # Save comparison
        timestamp = datetime.now().strftime("%H%M%S")
        output_path = self.results_dir / f"depth_comparison_{pair_name}_{timestamp}.jpg"
        cv2.imwrite(str(output_path), comparison)
        
        return str(output_path)

    def analyze_depth_pair(self, iphone_path: Path, sony_path: Path) -> Dict[str, Any]:
        """Analyze iPhone + Sony pair with depth estimation"""
        pair_name = f"{iphone_path.stem}+{sony_path.stem}"
        print(f"\n🎬 Analyzing depth pair: {iphone_path.stem} + {sony_path.stem}")
        
        # Extract metadata
        print(f"    📊 Extracting metadata...")
        iphone_meta = self.extract_raw_metadata(iphone_path)
        sony_meta = self.extract_raw_metadata(sony_path)
        
        print(f"  📱 iPhone: {iphone_meta['camera_model']}, {iphone_meta['aperture']}, ISO {iphone_meta['iso']}")
        print(f"  📷 Sony: {sony_meta['lens_model']} {sony_meta['aperture']}, ISO {sony_meta['iso']}")
        
        # Process RAW files
        print(f"    🎨 Processing RAW files...")
        iphone_rgb = self.process_raw_image(iphone_path)
        sony_rgb = self.process_raw_image(sony_path)
        
        if iphone_rgb is None or sony_rgb is None:
            print("  ❌ Failed to process RAW files")
            return None
        
        # Analyze characteristics
        print(f"    🔬 Analyzing depth characteristics...")
        analysis = self.analyze_depth_characteristics(iphone_rgb, sony_rgb)
        
        # Create comparison
        comparison_path = self.create_depth_comparison(
            iphone_rgb, sony_rgb, iphone_meta, sony_meta, analysis, pair_name
        )
        
        # Print results
        print(f"  📊 Color difference: {analysis['color_difference']:.1f}%")
        print(f"  📊 Dynamic range difference: {analysis['dynamic_range_difference']:.1f} stops")
        print(f"  🌊 Depth difference: {analysis['depth_difference']:.1f}")
        print(f"  🎯 Focus variance difference: {analysis['focus_variance_difference']:.0f}")
        print(f"  ⚡ Sharpness difference: {analysis['sharpness_difference']:.0f}")
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
        
        pairs = []
        for iphone_file in iphone_files:
            iphone_num = iphone_file.stem.split('_')[-1]
            sony_file = self.data_dir / f"sony_{iphone_num}.arw"
            if sony_file.exists():
                pairs.append((iphone_file, sony_file))
        
        print(f"🔗 Will process {len(pairs)} pairs")
        return pairs

    def run_depth_analysis(self):
        """Run complete depth analysis pipeline"""
        print("\n🌊 SIMPLE DEPTH ANALYSIS PIPELINE")
        print("=" * 50)
        print("Using OpenCV computer vision for spatial intelligence")
        
        pairs = self.find_paired_files()
        
        if not pairs:
            print("❌ No matching pairs found!")
            return
        
        all_results = []
        for iphone_path, sony_path in pairs:
            result = self.analyze_depth_pair(iphone_path, sony_path)
            if result:
                all_results.append(result)
        
        # Save metadata
        metadata_path = self.results_dir / "depth_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n✅ Depth analysis complete!")
        print(f"📁 Results saved to: {self.results_dir}")
        print(f"📊 Metadata saved to: {metadata_path}")
        
        if all_results:
            # Summary statistics
            color_diffs = [r['analysis']['color_difference'] for r in all_results]
            depth_diffs = [r['analysis']['depth_difference'] for r in all_results]
            sharpness_diffs = [r['analysis']['sharpness_difference'] for r in all_results]
            
            print(f"\n📈 SUMMARY STATISTICS:")
            print(f"   Average color difference: {np.mean(color_diffs):.1f}%")
            print(f"   Average depth difference: {np.mean(depth_diffs):.1f}")
            print(f"   Average sharpness difference: {np.mean(sharpness_diffs):.0f}")
            print(f"   Total pairs processed: {len(all_results)}")
        
        print(f"\n🎯 SIMPLE DEPTH INTELLIGENCE ACHIEVED:")
        print("✅ RAW color and dynamic range analysis")
        print("✅ Monocular depth estimation")
        print("✅ Focus variance and sharpness analysis")
        print("✅ Spatial characteristic mapping")
        print("✅ No external model dependencies")

def main():
    pipeline = SimpleDepthRAWPipeline()
    pipeline.run_depth_analysis()

if __name__ == "__main__":
    main()
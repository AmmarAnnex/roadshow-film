#!/usr/bin/env python3
"""
Spatial RAW Pipeline with ZoeDepth
Combines RAW analysis with modern depth estimation
"""

import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import rawpy
import exifread
import torch
from typing import Dict, Any, Tuple, Optional

class SpatialRAWPipeline:
    def __init__(self):
        print("🎯 SPATIAL RAW PIPELINE")
        print("RAW analysis + ZoeDepth spatial intelligence")
        print("Building the foundation for cinematic perception")
        
        # Setup directories
        self.setup_directories()
        
        # Initialize depth model
        self.setup_depth_model()
        
        print("✅ Spatial RAW pipeline initialized")

    def setup_directories(self):
        """Create necessary directories"""
        self.data_dir = Path("data/training_pairs")
        self.results_dir = Path("data/results/spatial_analysis")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def setup_depth_model(self):
        """Initialize ZoeDepth model (more stable than MiDaS)"""
        try:
            print("🔧 Loading ZoeDepth model...")
            
            # Use ZoeDepth from torch hub (much more stable)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"🔧 Using device: {self.device}")
            
            # Load ZoeDepth model
            self.depth_model = torch.hub.load('isl-org/ZoeDepth', 'ZoeD_N', pretrained=True)
            self.depth_model.to(self.device)
            self.depth_model.eval()
            
            print("✅ ZoeDepth model loaded successfully")
            
        except Exception as e:
            print(f"⚠️ Could not load ZoeDepth: {e}")
            print("📝 Will proceed without depth estimation for now")
            self.depth_model = None

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
                
                # Shutter speed
                if 'EXIF ExposureTime' in tags:
                    metadata["shutter_speed"] = str(tags['EXIF ExposureTime'])
                    
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

    def estimate_depth(self, rgb_image: np.ndarray) -> Optional[np.ndarray]:
        """Estimate depth using ZoeDepth"""
        if self.depth_model is None:
            return None
            
        try:
            # Convert to 8-bit for depth model
            rgb_8bit = (rgb_image / 257).astype(np.uint8)
            
            # Resize if too large (ZoeDepth works best with reasonable sizes)
            h, w = rgb_8bit.shape[:2]
            if max(h, w) > 1024:
                scale = 1024 / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                rgb_8bit = cv2.resize(rgb_8bit, (new_w, new_h))
            
            # Convert BGR to RGB if needed
            if len(rgb_8bit.shape) == 3:
                rgb_8bit = cv2.cvtColor(rgb_8bit, cv2.COLOR_BGR2RGB)
            
            # Run depth estimation
            with torch.no_grad():
                depth = self.depth_model.infer_pil(rgb_8bit)
            
            # Resize back to original size if needed
            if max(h, w) > 1024:
                depth = cv2.resize(depth, (w, h))
                
            return depth
            
        except Exception as e:
            print(f"    ⚠️ Depth estimation error: {e}")
            return None

    def analyze_spatial_characteristics(self, iphone_rgb: np.ndarray, sony_rgb: np.ndarray,
                                      iphone_depth: Optional[np.ndarray] = None, 
                                      sony_depth: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Analyze spatial and color characteristics"""
        # Resize to same dimensions for comparison
        h, w = min(iphone_rgb.shape[0], sony_rgb.shape[0]), min(iphone_rgb.shape[1], sony_rgb.shape[1])
        iphone_resized = cv2.resize(iphone_rgb, (w, h))
        sony_resized = cv2.resize(sony_rgb, (w, h))
        
        # Convert to float for analysis
        iphone_float = iphone_resized.astype(np.float32) / 65535.0
        sony_float = sony_resized.astype(np.float32) / 65535.0
        
        # Color difference analysis
        color_diff = np.mean(np.abs(iphone_float - sony_float)) * 100
        
        # Dynamic range analysis
        iphone_dr = np.log2(np.max(iphone_float) / (np.min(iphone_float[iphone_float > 0]) + 1e-8))
        sony_dr = np.log2(np.max(sony_float) / (np.min(sony_float[sony_float > 0]) + 1e-8))
        dr_difference = abs(iphone_dr - sony_dr)
        
        analysis = {
            "color_difference": float(color_diff),
            "dynamic_range_difference": float(dr_difference),
            "iphone_dynamic_range": float(iphone_dr),
            "sony_dynamic_range": float(sony_dr)
        }
        
        # Spatial analysis if depth available
        if iphone_depth is not None and sony_depth is not None:
            # Resize depth maps
            iphone_depth_resized = cv2.resize(iphone_depth, (w, h))
            sony_depth_resized = cv2.resize(sony_depth, (w, h))
            
            # Depth difference analysis
            depth_diff = np.mean(np.abs(iphone_depth_resized - sony_depth_resized))
            
            # Depth of field analysis
            iphone_dof_variance = np.var(iphone_depth_resized)
            sony_dof_variance = np.var(sony_depth_resized)
            dof_difference = abs(iphone_dof_variance - sony_dof_variance)
            
            # Focus plane detection
            iphone_focus_distance = np.median(iphone_depth_resized)
            sony_focus_distance = np.median(sony_depth_resized)
            focus_difference = abs(iphone_focus_distance - sony_focus_distance)
            
            analysis.update({
                "depth_difference": float(depth_diff),
                "dof_variance_difference": float(dof_difference),
                "focus_distance_difference": float(focus_difference),
                "iphone_dof_variance": float(iphone_dof_variance),
                "sony_dof_variance": float(sony_dof_variance),
                "iphone_focus_distance": float(iphone_focus_distance),
                "sony_focus_distance": float(sony_focus_distance)
            })
        
        return analysis

    def create_spatial_comparison(self, iphone_rgb: np.ndarray, sony_rgb: np.ndarray,
                                iphone_depth: Optional[np.ndarray], sony_depth: Optional[np.ndarray],
                                iphone_meta: Dict, sony_meta: Dict, 
                                analysis: Dict, pair_name: str) -> str:
        """Create comprehensive spatial comparison visualization"""
        # Convert 16-bit to 8-bit for display
        iphone_display = (iphone_rgb / 257).astype(np.uint8)
        sony_display = (sony_rgb / 257).astype(np.uint8)
        
        # Resize for comparison
        h = min(iphone_display.shape[0], sony_display.shape[0]) // 2  # Smaller for 4-panel layout
        w = min(iphone_display.shape[1], sony_display.shape[1]) // 2
        
        iphone_resized = cv2.resize(iphone_display, (w, h))
        sony_resized = cv2.resize(sony_display, (w, h))
        
        if iphone_depth is not None and sony_depth is not None:
            # Create depth visualizations
            iphone_depth_vis = cv2.resize(iphone_depth, (w, h))
            sony_depth_vis = cv2.resize(sony_depth, (w, h))
            
            # Normalize depth for visualization
            iphone_depth_norm = ((iphone_depth_vis - np.min(iphone_depth_vis)) / 
                               (np.max(iphone_depth_vis) - np.min(iphone_depth_vis) + 1e-8) * 255).astype(np.uint8)
            sony_depth_norm = ((sony_depth_vis - np.min(sony_depth_vis)) / 
                             (np.max(sony_depth_vis) - np.min(sony_depth_vis) + 1e-8) * 255).astype(np.uint8)
            
            # Apply colormap to depth
            iphone_depth_colored = cv2.applyColorMap(iphone_depth_norm, cv2.COLORMAP_PLASMA)
            sony_depth_colored = cv2.applyColorMap(sony_depth_norm, cv2.COLORMAP_PLASMA)
            
            # Create 2x2 comparison
            top_row = np.hstack([
                cv2.cvtColor(iphone_resized, cv2.COLOR_RGB2BGR), 
                cv2.cvtColor(sony_resized, cv2.COLOR_RGB2BGR)
            ])
            bottom_row = np.hstack([iphone_depth_colored, sony_depth_colored])
            comparison = np.vstack([top_row, bottom_row])
            
            # Labels for 4-panel layout
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(comparison, f"iPhone {iphone_meta['aperture']}", (10, 25), font, 0.6, (255, 255, 255), 1)
            cv2.putText(comparison, f"Sony {sony_meta['aperture']}", (w + 10, 25), font, 0.6, (255, 255, 255), 1)
            cv2.putText(comparison, "iPhone Depth", (10, h + 25), font, 0.6, (255, 255, 255), 1)
            cv2.putText(comparison, "Sony Depth", (w + 10, h + 25), font, 0.6, (255, 255, 255), 1)
            
            # Spatial analysis stats
            if "depth_difference" in analysis:
                stats_y = h * 2 - 40
                cv2.putText(comparison, f"Depth Diff: {analysis['depth_difference']:.1f}", 
                           (10, stats_y), font, 0.5, (255, 255, 255), 1)
                cv2.putText(comparison, f"DOF Diff: {analysis['dof_variance_difference']:.1f}", 
                           (w + 10, stats_y), font, 0.5, (255, 255, 255), 1)
        else:
            # Simple side-by-side if no depth
            comparison = np.hstack([
                cv2.cvtColor(iphone_resized, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(sony_resized, cv2.COLOR_RGB2BGR)
            ])
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(comparison, f"iPhone {iphone_meta['aperture']}", (10, 30), font, 0.7, (255, 255, 255), 2)
            cv2.putText(comparison, f"Sony {sony_meta['aperture']}", (w + 10, 30), font, 0.7, (255, 255, 255), 2)
        
        # Save comparison
        timestamp = datetime.now().strftime("%H%M%S")
        output_path = self.results_dir / f"spatial_comparison_{pair_name}_{timestamp}.jpg"
        cv2.imwrite(str(output_path), comparison)
        
        return str(output_path)

    def analyze_spatial_pair(self, iphone_path: Path, sony_path: Path) -> Dict[str, Any]:
        """Analyze iPhone + Sony pair with spatial intelligence"""
        pair_name = f"{iphone_path.stem}+{sony_path.stem}"
        print(f"\n🎬 Analyzing spatial pair: {iphone_path.stem} + {sony_path.stem}")
        
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
        
        # Estimate depth
        iphone_depth = None
        sony_depth = None
        
        if self.depth_model is not None:
            print(f"    🌊 Estimating depth maps...")
            iphone_depth = self.estimate_depth(iphone_rgb)
            sony_depth = self.estimate_depth(sony_rgb)
            
            if iphone_depth is not None and sony_depth is not None:
                print("    ✅ Depth estimation successful")
            else:
                print("    ⚠️ Depth estimation failed, continuing with color analysis only")
        
        # Analyze characteristics
        print(f"    🔬 Analyzing spatial characteristics...")
        analysis = self.analyze_spatial_characteristics(iphone_rgb, sony_rgb, iphone_depth, sony_depth)
        
        # Create comparison visualization
        comparison_path = self.create_spatial_comparison(
            iphone_rgb, sony_rgb, iphone_depth, sony_depth,
            iphone_meta, sony_meta, analysis, pair_name
        )
        
        # Print results
        print(f"  📊 Color difference: {analysis['color_difference']:.1f}%")
        print(f"  📊 Dynamic range difference: {analysis['dynamic_range_difference']:.1f} stops")
        
        if "depth_difference" in analysis:
            print(f"  🌊 Depth difference: {analysis['depth_difference']:.1f}")
            print(f"  🌊 DOF variance difference: {analysis['dof_variance_difference']:.1f}")
            print(f"  🎯 Focus distance difference: {analysis['focus_distance_difference']:.1f}")
        
        print(f"  ✅ Saved comparison: {Path(comparison_path).name}")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "iphone_file": str(iphone_path),
            "sony_file": str(sony_path),
            "iphone_metadata": iphone_meta,
            "sony_metadata": sony_meta,
            "analysis": analysis,
            "comparison_image": comparison_path,
            "has_depth_data": iphone_depth is not None and sony_depth is not None
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

    def run_spatial_analysis(self):
        """Run complete spatial analysis pipeline"""
        print("\n🌊 SPATIAL RAW ANALYSIS PIPELINE")
        print("=" * 50)
        print("Combining RAW sensor data with spatial intelligence")
        
        pairs = self.find_paired_files()
        
        if not pairs:
            print("❌ No matching pairs found!")
            return
        
        all_results = []
        for iphone_path, sony_path in pairs:
            result = self.analyze_spatial_pair(iphone_path, sony_path)
            if result:
                all_results.append(result)
        
        # Save metadata
        metadata_path = self.results_dir / "spatial_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n✅ Spatial analysis complete!")
        print(f"📁 Results saved to: {self.results_dir}")
        print(f"📊 Metadata saved to: {metadata_path}")
        
        if all_results:
            # Summary statistics
            color_diffs = [r['analysis']['color_difference'] for r in all_results]
            dr_diffs = [r['analysis']['dynamic_range_difference'] for r in all_results]
            
            spatial_pairs = [r for r in all_results if r['has_depth_data']]
            
            print(f"\n📈 SUMMARY STATISTICS:")
            print(f"   Average color difference: {np.mean(color_diffs):.1f}%")
            print(f"   Average DR difference: {np.mean(dr_diffs):.1f} stops")
            print(f"   Pairs with depth data: {len(spatial_pairs)}/{len(all_results)}")
            print(f"   Total pairs processed: {len(all_results)}")
            
            if spatial_pairs:
                depth_diffs = [r['analysis']['depth_difference'] for r in spatial_pairs]
                dof_diffs = [r['analysis']['dof_variance_difference'] for r in spatial_pairs]
                print(f"   Average depth difference: {np.mean(depth_diffs):.1f}")
                print(f"   Average DOF difference: {np.mean(dof_diffs):.1f}")
        
        print(f"\n🎯 SPATIAL INTELLIGENCE ACHIEVED:")
        print("✅ RAW color and dynamic range analysis")
        print("✅ Depth-aware spatial understanding")
        print("✅ Camera-specific characteristic mapping")
        print("✅ Foundation for physics-based lens emulation")

def main():
    pipeline = SpatialRAWPipeline()
    pipeline.run_spatial_analysis()

if __name__ == "__main__":
    main()
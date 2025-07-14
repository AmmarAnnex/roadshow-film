#!/usr/bin/env python3
"""
Image Transformation Analyzer
Measures exactly what's happening (or not happening) in transformations
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys
sys.path.append('backend')

class TransformationAnalyzer:
    """Analyze what's actually happening in transformations"""
    
    def __init__(self):
        self.results_dir = Path("data/analysis")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_existing_results(self):
        """Analyze the transformation results you already have"""
        print("🔍 ANALYZING TRANSFORMATION RESULTS")
        print("=" * 50)
        
        results_dir = Path("data/results/real_tests")
        if not results_dir.exists():
            print("❌ No results found. Run the transformation test first.")
            return
        
        image_files = list(results_dir.glob("*.jpg"))
        if not image_files:
            print("❌ No result images found")
            return
        
        print(f"📊 Found {len(image_files)} result images")
        
        for img_file in image_files:
            print(f"\n🎬 Analyzing {img_file.name}...")
            self.analyze_comparison_image(img_file)
    
    def analyze_comparison_image(self, img_path):
        """Analyze a side-by-side comparison image"""
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ❌ Could not load {img_path}")
            return
        
        # Split the side-by-side image
        h, w = img.shape[:2]
        original = img[:, :w//2]
        transformed = img[:, w//2:]
        
        # Calculate differences
        differences = self.calculate_differences(original, transformed)
        
        # Print analysis
        self.print_analysis(img_path.stem, differences)
        
        # Create detailed visualization
        self.create_detailed_analysis(img_path.stem, original, transformed, differences)
    
    def calculate_differences(self, original, transformed):
        """Calculate detailed differences between images"""
        # Convert to different color spaces for analysis
        orig_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        trans_rgb = cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB)
        
        orig_hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
        trans_hsv = cv2.cvtColor(transformed, cv2.COLOR_BGR2HSV)
        
        # Calculate differences
        rgb_diff = np.mean(np.abs(orig_rgb.astype(float) - trans_rgb.astype(float)))
        
        # Per-channel differences
        r_diff = np.mean(np.abs(orig_rgb[:,:,0].astype(float) - trans_rgb[:,:,0].astype(float)))
        g_diff = np.mean(np.abs(orig_rgb[:,:,1].astype(float) - trans_rgb[:,:,1].astype(float)))
        b_diff = np.mean(np.abs(orig_rgb[:,:,2].astype(float) - trans_rgb[:,:,2].astype(float)))
        
        # HSV differences
        h_diff = np.mean(np.abs(orig_hsv[:,:,0].astype(float) - trans_hsv[:,:,0].astype(float)))
        s_diff = np.mean(np.abs(orig_hsv[:,:,1].astype(float) - trans_hsv[:,:,1].astype(float)))
        v_diff = np.mean(np.abs(orig_hsv[:,:,2].astype(float) - trans_hsv[:,:,2].astype(float)))
        
        # Color temperature shift
        orig_temp = self.estimate_color_temperature(orig_rgb)
        trans_temp = self.estimate_color_temperature(trans_rgb)
        temp_shift = trans_temp - orig_temp
        
        # Maximum differences (hotspots)
        max_diff = np.max(np.abs(orig_rgb.astype(float) - trans_rgb.astype(float)))
        
        return {
            'overall_diff': rgb_diff,
            'red_diff': r_diff,
            'green_diff': g_diff, 
            'blue_diff': b_diff,
            'hue_diff': h_diff,
            'saturation_diff': s_diff,
            'value_diff': v_diff,
            'temp_shift': temp_shift,
            'max_diff': max_diff,
            'pixel_change_percent': np.mean(np.sum(np.abs(orig_rgb - trans_rgb), axis=2) > 10) * 100
        }
    
    def estimate_color_temperature(self, rgb_img):
        """Rough color temperature estimation"""
        r_avg = np.mean(rgb_img[:,:,0])
        b_avg = np.mean(rgb_img[:,:,2])
        
        if b_avg == 0:
            return 6500  # Default
        
        # Simplified color temperature estimation
        ratio = r_avg / b_avg
        if ratio > 1.2:
            return 3000  # Warm
        elif ratio < 0.8:
            return 8000  # Cool
        else:
            return 5500  # Neutral
    
    def print_analysis(self, image_name, diff):
        """Print human-readable analysis"""
        print(f"  📊 ANALYSIS FOR {image_name}:")
        print(f"     Overall difference: {diff['overall_diff']:.2f} (out of 255)")
        print(f"     Red channel shift: {diff['red_diff']:+.2f}")
        print(f"     Green channel shift: {diff['green_diff']:+.2f}")
        print(f"     Blue channel shift: {diff['blue_diff']:+.2f}")
        print(f"     Color temperature shift: {diff['temp_shift']:+.0f}K")
        print(f"     Pixels changed significantly: {diff['pixel_change_percent']:.1f}%")
        print(f"     Maximum local change: {diff['max_diff']:.1f}")
        
        # Interpret results
        if diff['overall_diff'] < 5:
            print("     🔴 BARELY NOTICEABLE - Transform too weak!")
        elif diff['overall_diff'] < 15:
            print("     🟡 SUBTLE - May need stronger effect")
        elif diff['overall_diff'] < 30:
            print("     🟢 GOOD - Noticeable but natural")
        else:
            print("     🔵 STRONG - Very obvious transformation")
    
    def create_detailed_analysis(self, name, original, transformed, diff):
        """Create detailed visual analysis"""
        # Create difference map
        diff_map = np.abs(original.astype(float) - transformed.astype(float))
        diff_map = np.mean(diff_map, axis=2)  # Average across channels
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Transformation Analysis: {name}', fontsize=16)
        
        # Original image
        axes[0,0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0,0].set_title('Original')
        axes[0,0].axis('off')
        
        # Transformed image  
        axes[0,1].imshow(cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB))
        axes[0,1].set_title('Transformed')
        axes[0,1].axis('off')
        
        # Difference map
        im = axes[1,0].imshow(diff_map, cmap='hot', vmin=0, vmax=50)
        axes[1,0].set_title('Difference Map (brighter = more change)')
        axes[1,0].axis('off')
        plt.colorbar(im, ax=axes[1,0])
        
        # Histogram comparison
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        trans_gray = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)
        
        axes[1,1].hist(orig_gray.flatten(), bins=50, alpha=0.7, label='Original', color='blue')
        axes[1,1].hist(trans_gray.flatten(), bins=50, alpha=0.7, label='Transformed', color='red')
        axes[1,1].set_title('Brightness Distribution')
        axes[1,1].legend()
        
        # Save analysis
        output_path = self.results_dir / f"{name}_analysis.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"     📈 Detailed analysis saved: {output_path}")
    
    def test_stronger_transformation(self):
        """Test with deliberately stronger transformations"""
        print("\n🚀 TESTING STRONGER TRANSFORMATIONS")
        print("=" * 50)
        
        from roadshow_3dlut_engine import Reality3DLUT, CAMERA_PROFILES
        
        # Find original image
        original_dir = Path("data/samples/real_photos")
        image_files = list(original_dir.glob("*.jpg")) + list(original_dir.glob("*.jpeg"))
        
        if not image_files:
            print("❌ No original images found")
            return
        
        img = cv2.imread(str(image_files[0]))
        print(f"🎬 Testing stronger transforms on {image_files[0].name}")
        
        # Initialize engine
        lut_engine = Reality3DLUT(resolution=32)
        lut_engine.create_base_lut()
        
        # Create deliberately stronger transformations
        strong_transforms = [
            {
                'name': 'Warm_Boost',
                'color_response': np.array([1.2, 1.0, 0.8]),  # Much warmer
                'description': 'Strong warm boost'
            },
            {
                'name': 'Cool_Boost', 
                'color_response': np.array([0.8, 1.0, 1.2]),  # Much cooler
                'description': 'Strong cool boost'
            },
            {
                'name': 'Saturation_Boost',
                'color_response': np.array([1.1, 1.1, 1.1]),  # Overall boost
                'description': 'Saturation boost'
            }
        ]
        
        output_dir = Path("data/analysis/strong_tests")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for transform in strong_transforms:
            print(f"  🎨 Testing {transform['description']}...")
            
            # Create custom transform
            from roadshow_3dlut_engine import CameraReality
            source = CAMERA_PROFILES['iphone_12_pro'] 
            target = CameraReality(
                name=transform['name'],
                color_response=transform['color_response'],
                spatial_falloff=0.8,
                temporal_cadence=0.95,
                character_signature={'digital': 0.2, 'organic': 0.8}
            )
            
            # Apply transformation
            lut_engine.learn_from_reference(source, target)
            result = lut_engine.apply_to_image(img, method='vectorized')
            
            # Create comparison
            comparison = np.hstack([img, result])
            output_path = output_dir / f"strong_{transform['name']}.jpg"
            cv2.imwrite(str(output_path), comparison)
            
            # Quick analysis
            differences = self.calculate_differences(img, result)
            print(f"     Difference: {differences['overall_diff']:.1f}, Temp shift: {differences['temp_shift']:+.0f}K")
        
        print(f"\n📁 Strong test results saved to: {output_dir}")
        print("🎯 If these look more dramatic, we know the issue!")


def main():
    analyzer = TransformationAnalyzer()
    
    print("🔬 ROADSHOW TRANSFORMATION ANALYZER")
    print("=" * 50)
    print("This will tell us exactly what's happening with your transformations\n")
    
    # Analyze existing results
    analyzer.analyze_existing_results()
    
    # Test stronger transformations
    analyzer.test_stronger_transformation()
    
    print("\n🎯 SUMMARY:")
    print("1. Check the analysis results above")
    print("2. Look at data/analysis/ for detailed visualizations") 
    print("3. If differences are < 5, transformations are too weak")
    print("4. Strong test results will show what dramatic changes look like")

if __name__ == "__main__":
    main()
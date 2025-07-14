#!/usr/bin/env python3
"""
Simple Roadshow Sample Tester
Creates test samples and runs transformations - no downloads needed!
"""

import os
import cv2
import numpy as np
from pathlib import Path
import sys

# Add backend to path
sys.path.append('backend')

class SimpleSampleTester:
    """Create samples and test Roadshow transformations"""
    
    def __init__(self):
        self.setup_directories()
        
    def setup_directories(self):
        """Create directory structure"""
        directories = [
            "data/samples/test_images",
            "data/results/quick_tests"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
        print("📁 Created test directories")
    
    def create_test_images(self):
        """Create diverse test images to stress-test the engine"""
        print("🎨 Creating test images...")
        
        test_images = {}
        
        # 1. Portrait test (skin tones)
        portrait = self.create_portrait_test()
        test_images['portrait'] = portrait
        
        # 2. Landscape test (sky/nature)
        landscape = self.create_landscape_test()
        test_images['landscape'] = landscape
        
        # 3. Color chart (technical test)
        color_chart = self.create_color_chart()
        test_images['color_chart'] = color_chart
        
        # 4. Mixed lighting (challenging)
        mixed_lighting = self.create_mixed_lighting_test()
        test_images['mixed_lighting'] = mixed_lighting
        
        # 5. High contrast (extreme test)
        high_contrast = self.create_high_contrast_test()
        test_images['high_contrast'] = high_contrast
        
        # Save all test images
        for name, image in test_images.items():
            cv2.imwrite(f'data/samples/test_images/{name}.jpg', image)
            
        print(f"  ✓ Created {len(test_images)} test images")
        return test_images
    
    def create_portrait_test(self):
        """Create portrait-style test image"""
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        # Background (soft blue-gray)
        img[:, :] = [180, 160, 140]
        
        # "Face" area (skin tone)
        cv2.ellipse(img, (960, 400), (200, 250), 0, 0, 360, (140, 180, 220), -1)
        
        # "Hair" area
        cv2.ellipse(img, (960, 280), (220, 150), 0, 0, 360, (60, 80, 40), -1)
        
        # Eyes
        cv2.circle(img, (920, 380), 15, (255, 255, 255), -1)
        cv2.circle(img, (1000, 380), 15, (255, 255, 255), -1)
        cv2.circle(img, (920, 380), 8, (100, 150, 80), -1)
        cv2.circle(img, (1000, 380), 8, (100, 150, 80), -1)
        
        return img
    
    def create_landscape_test(self):
        """Create landscape test image"""
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        # Sky gradient (blue to white)
        for y in range(500):
            blue_amount = int(255 * (1 - y / 500))
            img[y, :] = [255, 200 + blue_amount//3, blue_amount]
        
        # Ground (green)
        img[500:, :] = [60, 150, 60]
        
        # Mountains
        for i in range(7):
            x_center = i * 300 + 150
            y_peak = 300 + np.random.randint(-50, 50)
            points = np.array([
                [x_center - 100, 500],
                [x_center, y_peak], 
                [x_center + 100, 500]
            ], np.int32)
            cv2.fillPoly(img, [points], (80 + i*10, 100 + i*5, 120 + i*8))
        
        # Sun
        cv2.circle(img, (1600, 200), 80, (100, 200, 255), -1)
        
        return img
    
    def create_color_chart(self):
        """Create color accuracy test chart"""
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        # Standard color patches
        colors = [
            [115, 82, 68],    # Dark skin
            [194, 150, 130],  # Light skin
            [98, 122, 157],   # Blue sky
            [87, 108, 67],    # Foliage
            [133, 128, 177],  # Blue flower
            [103, 189, 170],  # Bluish green
            [214, 126, 44],   # Orange
            [80, 91, 166],    # Purplish blue
            [193, 90, 99],    # Moderate red
            [94, 60, 108],    # Purple
            [157, 188, 64],   # Yellow green
            [224, 163, 46],   # Orange yellow
            [56, 61, 150],    # Blue
            [70, 148, 73],    # Green
            [175, 54, 60],    # Red
            [231, 199, 31],   # Yellow
            [187, 86, 149],   # Magenta
            [8, 133, 161],    # Cyan
            [243, 243, 242],  # White
            [200, 200, 200],  # Neutral 8
            [160, 160, 160],  # Neutral 6.5
            [122, 122, 121],  # Neutral 5
            [85, 85, 85],     # Neutral 3.5
            [52, 52, 52]      # Black
        ]
        
        # Arrange in 6x4 grid
        patch_w, patch_h = 320, 270
        for i, color in enumerate(colors):
            row = i // 6
            col = i % 6
            x1, y1 = col * patch_w, row * patch_h
            x2, y2 = x1 + patch_w, y1 + patch_h
            img[y1:y2, x1:x2] = color
        
        return img
    
    def create_mixed_lighting_test(self):
        """Create mixed lighting scenario"""
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        # Base indoor lighting (warm)
        img[:, :] = [120, 150, 180]
        
        # Window area (cool daylight)
        cv2.rectangle(img, (1200, 100), (1800, 800), (200, 180, 140), -1)
        
        # Lamp area (very warm)
        cv2.circle(img, (300, 300), 200, (80, 120, 200), -1)
        
        # Shadow areas
        cv2.rectangle(img, (0, 600), (600, 1080), (60, 80, 100), -1)
        
        return img
    
    def create_high_contrast_test(self):
        """Create high contrast test"""
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        # Half black, half white
        img[:, :960] = [0, 0, 0]
        img[:, 960:] = [255, 255, 255]
        
        # Gradient in the middle
        for x in range(860, 1060):
            gradient_val = int(255 * (x - 860) / 200)
            img[:, x] = [gradient_val, gradient_val, gradient_val]
        
        # Some color patches in extreme areas
        cv2.circle(img, (400, 400), 100, (0, 0, 255), -1)  # Red in black
        cv2.circle(img, (1520, 400), 100, (255, 255, 0), -1)  # Yellow in white
        
        return img
    
    def test_transformations(self, test_images):
        """Test all transformation methods on all images"""
        print("🚀 Testing transformations...")
        
        try:
            from roadshow_3dlut_engine import Reality3DLUT, CAMERA_PROFILES
            
            # Initialize engine
            lut_engine = Reality3DLUT(resolution=32)
            lut_engine.create_base_lut()
            
            # Set up transformations to test
            transformations = [
                ('iPhone_to_ARRI', CAMERA_PROFILES['iphone_12_pro'], CAMERA_PROFILES['arri_alexa']),
                ('iPhone_to_Blackmagic', CAMERA_PROFILES['iphone_12_pro'], CAMERA_PROFILES['blackmagic_12k']),
                ('iPhone_to_Zeiss', CAMERA_PROFILES['iphone_12_pro'], CAMERA_PROFILES['zeiss_planar_50'])
            ]
            
            methods = ['vectorized', 'trilinear']
            
            for transform_name, source, target in transformations:
                print(f"\n📷 Testing {transform_name}...")
                
                # Learn transformation
                lut_engine.learn_from_reference(source, target)
                
                for method in methods:
                    print(f"  🔧 Method: {method}")
                    
                    for img_name, image in test_images.items():
                        # Apply transformation
                        result = lut_engine.apply_to_image(image, method=method)
                        
                        # Create side-by-side comparison
                        comparison = np.hstack([image, result])
                        
                        # Save result
                        filename = f"{transform_name}_{method}_{img_name}.jpg"
                        output_path = f"data/results/quick_tests/{filename}"
                        cv2.imwrite(output_path, comparison)
                
                print(f"  ✅ {transform_name} complete")
            
            print(f"\n🎉 All transformations complete!")
            print(f"📁 Check data/results/quick_tests/ for results")
            
        except ImportError as e:
            print(f"❌ Could not import engine: {e}")
            return False
            
        return True
    
    def analyze_results(self):
        """Provide analysis of the test results"""
        print("\n🔍 ANALYSIS GUIDE:")
        print("=" * 50)
        print("📊 What to look for in your results:")
        print()
        print("1. COLOR CHART TEST:")
        print("   ✓ Skin tones should look natural, not orange/green")
        print("   ✓ Blues shouldn't shift to purple/cyan")
        print("   ✓ Neutrals should stay neutral")
        print()
        print("2. PORTRAIT TEST:")
        print("   ✓ Skin should look warm and natural")
        print("   ✓ Eyes should maintain color accuracy")
        print("   ✓ Background shouldn't shift dramatically")
        print()
        print("3. LANDSCAPE TEST:")
        print("   ✓ Sky should look natural (not too blue/green)")
        print("   ✓ Foliage should stay green")
        print("   ✓ Sun should maintain warmth")
        print()
        print("4. MIXED LIGHTING:")
        print("   ✓ Different light sources should balance properly")
        print("   ✓ No extreme color casts in any area")
        print()
        print("5. HIGH CONTRAST:")
        print("   ✓ Shadows shouldn't crush to black")
        print("   ✓ Highlights shouldn't blow to white")
        print("   ✓ Gradient should be smooth")
        print()
        print("🎯 NEXT STEPS:")
        print("1. Review all test results")
        print("2. Note any issues (color shifts, artifacts)")
        print("3. Test with your real footage this afternoon")
        print("4. Iterate on transformation parameters")
    
    def run_full_test(self):
        """Run complete test suite"""
        print("🎬 ROADSHOW SIMPLE SAMPLE TESTER")
        print("=" * 50)
        
        # Create test images
        test_images = self.create_test_images()
        
        # Test transformations
        success = self.test_transformations(test_images)
        
        if success:
            # Provide analysis guide
            self.analyze_results()
        else:
            print("❌ Tests failed - check your setup")


if __name__ == "__main__":
    tester = SimpleSampleTester()
    tester.run_full_test()
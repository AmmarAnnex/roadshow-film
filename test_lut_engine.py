#!/usr/bin/env python3
"""
Comprehensive Test Suite for Roadshow 3D LUT Engine
This ensures optimizations don't break existing functionality
"""

import sys
import os
import time
import numpy as np
import cv2
from pathlib import Path

# Add the backend directory to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from roadshow_3dlut_engine import Reality3DLUT, CAMERA_PROFILES
from synthetic_training import SyntheticDataGenerator

class LUTEngineTestSuite:
    """Comprehensive testing for the 3D LUT engine"""
    
    def __init__(self):
        self.test_results = {}
        self.performance_results = {}
        
        # Create test data directory
        os.makedirs('tests/data', exist_ok=True)
        os.makedirs('tests/results', exist_ok=True)
        
        print("🧪 Initializing Roadshow 3D LUT Test Suite")
        print("=" * 50)
    
    def create_test_images(self):
        """Generate standardized test images"""
        print("\n📸 Creating test images...")
        
        test_images = {}
        
        # 1. Simple gradient test
        gradient = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(256):
            gradient[:, i] = [i, i, i]  # Grayscale gradient
        test_images['gradient'] = gradient
        
        # 2. Color cube test
        color_cube = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(256):
            for j in range(256):
                color_cube[i, j] = [i, j, 128]  # Red-Green variation
        test_images['color_cube'] = color_cube
        
        # 3. Pure colors test
        pure_colors = np.zeros((150, 450, 3), dtype=np.uint8)
        pure_colors[:, :150] = [255, 0, 0]    # Red
        pure_colors[:, 150:300] = [0, 255, 0] # Green  
        pure_colors[:, 300:450] = [0, 0, 255] # Blue
        test_images['pure_colors'] = pure_colors
        
        # 4. Checkerboard pattern
        checkerboard = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(256):
            for j in range(256):
                if (i // 32 + j // 32) % 2:
                    checkerboard[i, j] = [255, 255, 255]
        test_images['checkerboard'] = checkerboard
        
        # Save test images
        for name, image in test_images.items():
            cv2.imwrite(f'tests/data/{name}.png', image)
        
        print(f"  ✓ Created {len(test_images)} test images")
        return test_images
    
    def test_basic_functionality(self):
        """Test that basic LUT operations work"""
        print("\n🔧 Testing basic functionality...")
        
        try:
            # Initialize engine
            lut_engine = Reality3DLUT(resolution=16)  # Small for fast testing
            lut_engine.create_base_lut()
            
            # Test identity LUT (should not change image)
            test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            result = lut_engine.apply_to_image(test_image)
            
            # Check if result has same shape
            assert result.shape == test_image.shape, "Output shape mismatch"
            
            # Check if values are reasonable (should be very close for identity)
            diff = np.mean(np.abs(result.astype(float) - test_image.astype(float)))
            assert diff < 10, f"Identity LUT changed image too much: {diff}"
            
            self.test_results['basic_functionality'] = "PASS"
            print("  ✓ Basic functionality test passed")
            
        except Exception as e:
            self.test_results['basic_functionality'] = f"FAIL: {e}"
            print(f"  ❌ Basic functionality test failed: {e}")
    
    def test_camera_profiles(self):
        """Test camera profile transformations"""
        print("\n📷 Testing camera profiles...")
        
        try:
            lut_engine = Reality3DLUT(resolution=16)
            lut_engine.create_base_lut()
            
            # Test iPhone to ARRI transformation
            iphone = CAMERA_PROFILES['iphone_12_pro']
            arri = CAMERA_PROFILES['arri_alexa']
            
            lut_engine.learn_from_reference(iphone, arri)
            
            # Apply to test image
            test_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
            result = lut_engine.apply_to_image(test_image)
            
            # Should produce different result than input
            assert not np.array_equal(result, test_image), "Transformation had no effect"
            
            self.test_results['camera_profiles'] = "PASS"
            print("  ✓ Camera profile test passed")
            
        except Exception as e:
            self.test_results['camera_profiles'] = f"FAIL: {e}"
            print(f"  ❌ Camera profile test failed: {e}")
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        print("\n⚠️  Testing edge cases...")
        
        try:
            lut_engine = Reality3DLUT(resolution=8)
            lut_engine.create_base_lut()
            
            # Test with all black image
            black_image = np.zeros((50, 50, 3), dtype=np.uint8)
            result = lut_engine.apply_to_image(black_image)
            assert result.shape == black_image.shape, "Black image shape changed"
            
            # Test with all white image  
            white_image = np.full((50, 50, 3), 255, dtype=np.uint8)
            result = lut_engine.apply_to_image(white_image)
            assert result.shape == white_image.shape, "White image shape changed"
            
            # Test with single pixel
            pixel = np.array([[[128, 128, 128]]], dtype=np.uint8)
            result = lut_engine.apply_to_image(pixel)
            assert result.shape == pixel.shape, "Single pixel shape changed"
            
            self.test_results['edge_cases'] = "PASS"
            print("  ✓ Edge cases test passed")
            
        except Exception as e:
            self.test_results['edge_cases'] = f"FAIL: {e}"
            print(f"  ❌ Edge cases test failed: {e}")
    
    def benchmark_performance(self, test_images):
        """Benchmark current performance"""
        print("\n⏱️  Benchmarking performance...")
        
        lut_engine = Reality3DLUT(resolution=32)  # Standard resolution
        lut_engine.create_base_lut()
        
        # Set up transformation
        iphone = CAMERA_PROFILES['iphone_12_pro']
        arri = CAMERA_PROFILES['arri_alexa']
        lut_engine.learn_from_reference(iphone, arri)
        
        # Benchmark each test image
        for name, image in test_images.items():
            start_time = time.time()
            result = lut_engine.apply_to_image(image)
            end_time = time.time()
            
            processing_time = end_time - start_time
            pixels_per_second = (image.shape[0] * image.shape[1]) / processing_time
            
            self.performance_results[name] = {
                'time': processing_time,
                'pixels_per_second': pixels_per_second,
                'image_size': f"{image.shape[0]}x{image.shape[1]}"
            }
            
            print(f"  📊 {name}: {processing_time:.3f}s ({pixels_per_second:,.0f} pixels/sec)")
    
    def save_visual_tests(self, test_images):
        """Create visual comparison outputs"""
        print("\n🎨 Creating visual test outputs...")
        
        lut_engine = Reality3DLUT(resolution=32)
        lut_engine.create_base_lut()
        
        # Set up transformation
        iphone = CAMERA_PROFILES['iphone_12_pro']
        arri = CAMERA_PROFILES['arri_alexa']
        lut_engine.learn_from_reference(iphone, arri)
        
        for name, image in test_images.items():
            # Apply transformation
            result = lut_engine.apply_to_image(image)
            
            # Create side-by-side comparison
            comparison = np.hstack([image, result])
            
            # Save comparison
            cv2.imwrite(f'tests/results/{name}_comparison.png', comparison)
        
        print(f"  ✓ Saved visual comparisons to tests/results/")
    
    def run_all_tests(self):
        """Run the complete test suite"""
        print("🚀 Starting comprehensive test suite...")
        
        # Create test data
        test_images = self.create_test_images()
        
        # Run functionality tests
        self.test_basic_functionality()
        self.test_camera_profiles()
        self.test_edge_cases()
        
        # Performance benchmarking
        self.benchmark_performance(test_images)
        
        # Visual outputs
        self.save_visual_tests(test_images)
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test results summary"""
        print("\n" + "=" * 50)
        print("📋 TEST SUITE SUMMARY")
        print("=" * 50)
        
        # Functionality results
        print("\n🔧 FUNCTIONALITY TESTS:")
        for test, result in self.test_results.items():
            status = "✅" if result == "PASS" else "❌"
            print(f"  {status} {test}: {result}")
        
        # Performance results
        print("\n⏱️  PERFORMANCE BENCHMARKS:")
        for test, metrics in self.performance_results.items():
            print(f"  📊 {test} ({metrics['image_size']}): {metrics['time']:.3f}s")
        
        # Overall status
        passed_tests = sum(1 for result in self.test_results.values() if result == "PASS")
        total_tests = len(self.test_results)
        
        print(f"\n🎯 OVERALL: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("✅ All tests passed! Engine is working correctly.")
        else:
            print("❌ Some tests failed. Check issues before optimizing.")
        
        print("\n📁 Check tests/results/ for visual comparisons")


if __name__ == "__main__":
    # Run the test suite
    test_suite = LUTEngineTestSuite()
    test_suite.run_all_tests()
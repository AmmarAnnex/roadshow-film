#!/usr/bin/env python3
"""
Debug Orientation Issue
Find exactly where and why images are getting flipped
"""

import rawpy
import cv2
import numpy as np
from pathlib import Path
import exifread
from PIL import Image

def check_exif_orientation(file_path):
    """Check EXIF orientation data"""
    try:
        with open(file_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
            
        orientation = "Unknown"
        if 'Image Orientation' in tags:
            orientation = str(tags['Image Orientation'])
            
        return orientation
    except:
        return "Error reading EXIF"

def test_rawpy_processing(file_path):
    """Test different rawpy processing options"""
    print(f"\n🔍 Testing: {Path(file_path).name}")
    
    # Check EXIF orientation first
    exif_orient = check_exif_orientation(file_path)
    print(f"📊 EXIF Orientation: {exif_orient}")
    
    # Test different user_flip values
    flip_results = {}
    
    for flip_val in [0, 1, 2, 3]:
        try:
            with rawpy.imread(file_path) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    output_bps=16,
                    no_auto_bright=True,
                    user_flip=flip_val
                )
            
            # Check if image looks normal by analyzing corner brightness
            h, w = rgb.shape[:2]
            top_left = np.mean(rgb[:h//4, :w//4])
            bottom_right = np.mean(rgb[3*h//4:, 3*w//4:])
            
            flip_results[flip_val] = {
                'shape': rgb.shape,
                'top_left_brightness': top_left,
                'bottom_right_brightness': bottom_right,
                'brightness_ratio': bottom_right / (top_left + 1e-6)
            }
            
            print(f"  user_flip={flip_val}: shape={rgb.shape}, TL={top_left:.1f}, BR={bottom_right:.1f}, ratio={bottom_right/(top_left+1e-6):.2f}")
            
        except Exception as e:
            flip_results[flip_val] = f"Error: {e}"
            print(f"  user_flip={flip_val}: ERROR - {e}")
    
    return flip_results

def compare_processing_methods(file_path):
    """Compare different ways of processing the same image"""
    print(f"\n🔬 Comparing processing methods for: {Path(file_path).name}")
    
    try:
        # Method 1: rawpy default
        with rawpy.imread(file_path) as raw:
            rgb1 = raw.postprocess(use_camera_wb=True, output_bps=16, no_auto_bright=True)
        
        # Method 2: rawpy with user_flip=0
        with rawpy.imread(file_path) as raw:
            rgb2 = raw.postprocess(use_camera_wb=True, output_bps=16, no_auto_bright=True, user_flip=0)
        
        # Method 3: rawpy with no orientation correction
        with rawpy.imread(file_path) as raw:
            rgb3 = raw.postprocess(use_camera_wb=True, output_bps=16, no_auto_bright=True, user_flip=-1)
        
        print(f"  Default processing: {rgb1.shape}")
        print(f"  user_flip=0: {rgb2.shape}")
        print(f"  user_flip=-1: {rgb3.shape}")
        
        # Check if they're the same
        if np.array_equal(rgb1, rgb2):
            print("  ✅ Default and user_flip=0 are identical")
        else:
            print("  ⚠️ Default and user_flip=0 are different!")
            
        return rgb1, rgb2, rgb3
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None, None, None

def find_best_orientation(file_path):
    """Determine the best orientation for this image"""
    print(f"\n🎯 Finding best orientation for: {Path(file_path).name}")
    
    results = test_rawpy_processing(file_path)
    
    # Look for the orientation where top is brighter than bottom (typical for overhead lighting)
    best_flip = 0
    best_score = 0
    
    for flip_val, result in results.items():
        if isinstance(result, dict):
            # Prefer orientations where top-left is brighter (assuming overhead lighting)
            score = result['top_left_brightness'] / (result['bottom_right_brightness'] + 1e-6)
            print(f"  user_flip={flip_val}: score={score:.2f} (higher = more likely correct)")
            
            if score > best_score:
                best_score = score
                best_flip = flip_val
    
    print(f"  🎯 Recommended user_flip: {best_flip}")
    return best_flip

def create_orientation_reference(file_path, output_dir):
    """Create reference images showing all orientations"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = Path(file_path).stem
    
    for flip_val in [0, 1, 2, 3]:
        try:
            with rawpy.imread(file_path) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    output_bps=8,  # 8-bit for easier viewing
                    no_auto_bright=True,
                    user_flip=flip_val
                )
            
            # Resize for easier comparison
            rgb_small = cv2.resize(rgb, (256, 256))
            rgb_bgr = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2BGR)
            
            # Add label
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(rgb_bgr, f"user_flip={flip_val}", (10, 30), font, 0.7, (0, 255, 0), 2)
            
            output_path = output_dir / f"{base_name}_flip{flip_val}.jpg"
            cv2.imwrite(str(output_path), rgb_bgr)
            
        except Exception as e:
            print(f"Error with flip {flip_val}: {e}")
    
    print(f"✅ Orientation references saved to: {output_dir}")

def main():
    """Debug orientation issues"""
    print("🔍 ORIENTATION DEBUG TOOL")
    print("=" * 50)
    
    # Test with your training images
    training_dir = Path("data/training_pairs")
    iphone_files = list(training_dir.glob("iphone_*.dng"))[:3]
    
    if not iphone_files:
        print("❌ No iPhone DNG files found!")
        return
    
    for iphone_file in iphone_files:
        print(f"\n" + "="*60)
        
        # Check EXIF and test processing
        test_rawpy_processing(str(iphone_file))
        
        # Compare methods
        compare_processing_methods(str(iphone_file))
        
        # Find best orientation
        best_flip = find_best_orientation(str(iphone_file))
        
        # Create reference images
        create_orientation_reference(str(iphone_file), f"data/orientation_debug/{iphone_file.stem}")
        
        print(f"✅ Best orientation for {iphone_file.name}: user_flip={best_flip}")
    
    print(f"\n🎯 SUMMARY:")
    print("1. Check the orientation_debug folder to visually see all rotations")
    print("2. Use the recommended user_flip values for each image")
    print("3. Consider using a lookup table for consistent orientation")

if __name__ == "__main__":
    main()
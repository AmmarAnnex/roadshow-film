#!/usr/bin/env python3
"""
Fix EXIF Orientation Properly
Force consistent orientation handling based on EXIF data
"""

import cv2
import numpy as np
from pathlib import Path
import rawpy
import exifread

def get_exif_orientation(file_path):
    """Get EXIF orientation value"""
    try:
        with open(file_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
            
        if 'Image Orientation' in tags:
            orientation_str = str(tags['Image Orientation'])
            
            # Map orientation strings to numeric values
            orientation_map = {
                'Horizontal (normal)': 1,
                'Mirrored horizontal': 2, 
                'Rotated 180': 3,
                'Mirrored vertical': 4,
                'Mirrored horizontal then rotated 90 CCW': 5,
                'Rotated 90 CW': 6,
                'Mirrored horizontal then rotated 90 CW': 7,
                'Rotated 90 CCW': 8
            }
            
            return orientation_map.get(orientation_str, 1)
        else:
            return 1  # Default: normal orientation
            
    except Exception as e:
        print(f"Error reading EXIF from {file_path}: {e}")
        return 1

def apply_exif_rotation(image, orientation):
    """Apply rotation based on EXIF orientation value"""
    if orientation == 1:
        # Normal - no rotation needed
        return image
    elif orientation == 2:
        # Mirrored horizontal
        return cv2.flip(image, 1)
    elif orientation == 3:
        # Rotated 180
        return cv2.rotate(image, cv2.ROTATE_180)
    elif orientation == 4:
        # Mirrored vertical  
        return cv2.flip(image, 0)
    elif orientation == 5:
        # Mirrored horizontal then rotated 90 CCW
        return cv2.flip(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)
    elif orientation == 6:
        # Rotated 90 CW
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 7:
        # Mirrored horizontal then rotated 90 CW
        return cv2.flip(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), 1)
    elif orientation == 8:
        # Rotated 90 CCW
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return image

def process_image_with_proper_orientation(file_path, size=256):
    """Process RAW image with proper EXIF orientation handling"""
    try:
        # Step 1: Get EXIF orientation BEFORE processing
        exif_orientation = get_exif_orientation(file_path)
        print(f"    📊 EXIF orientation: {exif_orientation}")
        
        # Step 2: Process RAW with NO auto-rotation (disable rawpy's rotation)
        with rawpy.imread(file_path) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                output_bps=16,
                no_auto_bright=True,
                user_flip=0,  # Force no rotation from rawpy
                use_auto_wb=False,  # Disable auto adjustments
                no_auto_scale=True   # Disable auto scaling
            )
        
        print(f"    📊 RAW shape after processing: {rgb.shape}")
        
        # Step 3: Apply EXIF rotation manually using OpenCV
        if exif_orientation != 1:
            print(f"    🔄 Applying EXIF rotation for orientation {exif_orientation}")
            rgb = apply_exif_rotation(rgb, exif_orientation)
            print(f"    📊 Shape after EXIF rotation: {rgb.shape}")
        
        # Step 4: Normalize and resize
        rgb_norm = rgb.astype(np.float32) / 65535.0
        rgb_resized = cv2.resize(rgb_norm, (size, size))
        
        return np.transpose(rgb_resized, (2, 0, 1))
        
    except Exception as e:
        print(f"    ❌ Error processing {file_path}: {e}")
        return None

def test_fixed_orientation():
    """Test the fixed orientation processing"""
    print("🔧 TESTING FIXED ORIENTATION PROCESSING")
    print("=" * 50)
    
    training_dir = Path("data/training_pairs")
    iphone_files = list(training_dir.glob("iphone_*.dng"))[:3]
    
    results_dir = Path("data/orientation_fixed_test")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for iphone_file in iphone_files:
        print(f"\n🎯 Processing: {iphone_file.name}")
        
        # Process with fixed orientation
        processed = process_image_with_proper_orientation(str(iphone_file))
        
        if processed is not None:
            # Convert back to display format
            display_img = np.transpose(processed, (1, 2, 0))
            display_img = (display_img * 255).astype(np.uint8)
            display_bgr = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
            
            # Add label
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(display_bgr, "EXIF Corrected", (10, 30), font, 0.7, (0, 255, 0), 2)
            
            # Save
            output_path = results_dir / f"fixed_orientation_{iphone_file.stem}.jpg"
            cv2.imwrite(str(output_path), display_bgr)
            
            print(f"    ✅ Saved: {output_path}")
        else:
            print(f"    ❌ Failed to process")
    
    print(f"\n✅ Fixed orientation test complete!")
    print(f"📁 Check results in: {results_dir}")
    print(f"🎯 All images should now be right-side up!")

def create_fixed_dataset_class():
    """Create the fixed dataset class code"""
    
    fixed_code = '''
class ProperOrientationDataset(Dataset):
    """Dataset with proper EXIF orientation handling"""
    
    def process_image_with_exif(self, file_path: str, size: int = 256):
        """Process image with proper EXIF orientation"""
        # Get EXIF orientation
        exif_orientation = get_exif_orientation(file_path)
        
        # Process RAW with no auto-rotation
        with rawpy.imread(file_path) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                output_bps=16,
                no_auto_bright=True,
                user_flip=0,  # No rawpy rotation
                use_auto_wb=False,
                no_auto_scale=True
            )
        
        # Apply EXIF rotation manually
        if exif_orientation != 1:
            rgb = apply_exif_rotation(rgb, exif_orientation)
        
        # Normalize and resize
        rgb_norm = rgb.astype(np.float32) / 65535.0
        rgb_resized = cv2.resize(rgb_norm, (size, size))
        
        return np.transpose(rgb_resized, (2, 0, 1))
'''
    
    print("💾 FIXED DATASET CLASS:")
    print(fixed_code)
    
    with open("proper_orientation_dataset.py", "w") as f:
        f.write(fixed_code)
    
    print("✅ Saved to: proper_orientation_dataset.py")

def main():
    """Test and demonstrate proper orientation handling"""
    print("🔧 PROPER EXIF ORIENTATION HANDLING")
    print("=" * 50)
    
    print("🎯 THE PROBLEM:")
    print("- Your iPhone images have EXIF: 'Rotated 180'")
    print("- rawpy.postprocess() is inconsistently applying this rotation")
    print("- Some images get rotated, others don't")
    
    print("\n🔧 THE SOLUTION:")
    print("- Disable rawpy's auto-rotation (user_flip=0)")
    print("- Read EXIF orientation data manually")
    print("- Apply rotation consistently using OpenCV")
    print("- All images will be right-side up, every time")
    
    print(f"\n" + "="*50)
    
    # Test the fix
    test_fixed_orientation()
    
    # Show the code structure
    create_fixed_dataset_class()
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. Replace your current dataset class with ProperOrientationDataset")
    print("2. All orientation issues should be resolved")
    print("3. No more upside-down images!")

if __name__ == "__main__":
    main()
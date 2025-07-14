#!/usr/bin/env python3
import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.append('backend')

def test_real_images():
    from roadshow_3dlut_engine import Reality3DLUT, CAMERA_PROFILES
    
    # Setup
    input_dir = Path("data/samples/real_photos")
    output_dir = Path("data/results/real_tests")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find images
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg")) + list(input_dir.glob("*.png"))
    
    if not image_files:
        print("❌ No images found in data/samples/real_photos/")
        return
    
    print(f"📸 Found {len(image_files)} images")
    
    # Initialize engine
    lut_engine = Reality3DLUT(resolution=32)
    lut_engine.create_base_lut()
    
    # Test all transformations
    transforms = [
        ('iPhone_to_ARRI', 'iphone_12_pro', 'arri_alexa'),
        ('iPhone_to_Blackmagic', 'iphone_12_pro', 'blackmagic_12k'),
        ('iPhone_to_Zeiss', 'iphone_12_pro', 'zeiss_planar_50')
    ]
    
    for img_file in image_files:
        print(f"\n🎬 Processing {img_file.name}...")
        img = cv2.imread(str(img_file))
        
        if img is None:
            continue
            
        for transform_name, source, target in transforms:
            # Learn transformation
            lut_engine.learn_from_reference(CAMERA_PROFILES[source], CAMERA_PROFILES[target])
            
            # Apply transformation
            result = lut_engine.apply_to_image(img, method='vectorized')
            
            # Create comparison
            comparison = np.hstack([img, result])
            
            # Save
            output_name = f"{img_file.stem}_{transform_name}.jpg"
            output_path = output_dir / output_name
            cv2.imwrite(str(output_path), comparison)
            
            print(f"  ✓ {transform_name}")
    
    print(f"\n✅ All done! Check data/results/real_tests/")

if __name__ == "__main__":
    test_real_images()
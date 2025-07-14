#!/usr/bin/env python3
"""
Test Roadshow 3D LUT System
Run this to see the transformation in action!
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from backend.roadshow_3dlut_engine import Reality3DLUT, CAMERA_PROFILES
from backend.synthetic_training import SyntheticDataGenerator

def main():
    print("🎬 ROADSHOW 3D LUT TEST")
    print("=" * 50)
    
    # Step 1: Generate test data
    print("\n1️⃣ Generating test images...")
    generator = SyntheticDataGenerator()
    generator.generate_samples()
    
    # Step 2: Load test image
    test_image = cv2.imread('data/samples/color_checker_iphone.png')
    if test_image is None:
        print("❌ Could not load test image")
        return
    
    # Step 3: Initialize 3D LUT engine
    print("\n2️⃣ Initializing 3D LUT engine...")
    lut_engine = Reality3DLUT(resolution=32)
    lut_engine.create_base_lut()
    
    # Step 4: Learn transformation
    print("\n3️⃣ Learning iPhone → ARRI transformation...")
    iphone = CAMERA_PROFILES['iphone_12_pro']
    arri = CAMERA_PROFILES['arri_alexa']
    lut_engine.learn_from_reference(iphone, arri)
    
    # Step 5: Apply transformation
    print("\n4️⃣ Applying transformation...")
    result = lut_engine.apply_to_image(test_image)
    
    # Step 6: Save results
    os.makedirs('data/results', exist_ok=True)
    comparison = np.hstack([test_image, result])
    cv2.imwrite('data/results/before_after.png', comparison)
    
    print("\n✅ SUCCESS! Check your results:")
    print("   📁 data/results/before_after.png")
    print("\n📊 What happened:")
    print("   • Created synthetic iPhone footage")
    print("   • Learned ARRI characteristics")
    print("   • Applied 3D LUT transformation")
    print("   • iPhone footage now looks like ARRI!")

if __name__ == "__main__":
    main()
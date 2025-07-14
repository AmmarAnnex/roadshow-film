#!/usr/bin/env python3
"""
Test Cinema Transformation
Quick test script to see iPhone→Sony+Zeiss transformation in action
"""

import cv2
import numpy as np
from pathlib import Path
import torch
import rawpy
from transformation_engine import CinemaTransformNet, CinemaTransformInference

def quick_test_transform():
    """Quick test using your existing training data"""
    print("🎬 TESTING CINEMA TRANSFORMATION")
    print("=" * 50)
    
    # Check for trained model
    model_files = list(Path("data").glob("cinema_transform_model_*.pth"))
    if not model_files:
        print("❌ No trained model found!")
        print("Run transformation_engine.py first to train the model")
        return
    
    # Use the latest model
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    print(f"📊 Using model: {latest_model}")
    
    # Initialize inference
    inference = CinemaTransformInference(latest_model)
    
    # Test on your training images
    training_pairs_dir = Path("data/training_pairs")
    iphone_files = list(training_pairs_dir.glob("iphone_*.dng"))
    
    if not iphone_files:
        print("❌ No iPhone DNG files found for testing!")
        return
    
    # Test on first few images
    test_files = iphone_files[:3]
    results_dir = Path("data/results/transformation_test")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for iphone_file in test_files:
        print(f"\n🎯 Transforming: {iphone_file.name}")
        
        try:
            # Transform image
            transformed = inference.transform_image(str(iphone_file))
            
            # Load original for comparison
            with rawpy.imread(str(iphone_file)) as raw:
                original = raw.postprocess(
                    use_camera_wb=True,
                    output_bps=8,
                    no_auto_bright=True,
                    user_flip=0
                )
            
            # Resize original to match transformed
            original_resized = cv2.resize(original, (512, 512))
            
            # Create side-by-side comparison
            comparison = np.hstack([
                cv2.cvtColor(original_resized, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR)
            ])
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(comparison, "Original iPhone", (10, 30), font, 0.7, (255, 255, 255), 2)
            cv2.putText(comparison, "Transformed to Sony+Zeiss", (520, 30), font, 0.7, (255, 255, 255), 2)
            
            # Save comparison
            output_path = results_dir / f"transform_comparison_{iphone_file.stem}.jpg"
            cv2.imwrite(str(output_path), comparison)
            
            print(f"  ✅ Saved comparison: {output_path}")
            
        except Exception as e:
            print(f"  ❌ Error transforming {iphone_file.name}: {e}")
    
    print(f"\n🎯 Test complete! Check results in: {results_dir}")
    print(f"📊 Compare original iPhone vs transformed Sony+Zeiss characteristics")

def analyze_transformation_quality():
    """Analyze the quality of transformations"""
    results_dir = Path("data/results/transformation_test")
    
    if not results_dir.exists():
        print("❌ No test results found. Run quick_test_transform() first.")
        return
    
    comparison_files = list(results_dir.glob("transform_comparison_*.jpg"))
    
    print(f"\n📊 TRANSFORMATION QUALITY ANALYSIS")
    print("=" * 50)
    print(f"Found {len(comparison_files)} test results")
    
    for comp_file in comparison_files:
        print(f"\n📸 Analyzing: {comp_file.name}")
        
        # Load comparison image
        img = cv2.imread(str(comp_file))
        h, w = img.shape[:2]
        
        # Split into original and transformed
        original = img[:, :w//2]
        transformed = img[:, w//2:]
        
        # Calculate differences
        color_diff = np.mean(np.abs(original.astype(float) - transformed.astype(float)))
        
        # Calculate histogram differences
        orig_hist = cv2.calcHist([original], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
        trans_hist = cv2.calcHist([transformed], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
        hist_diff = cv2.compareHist(orig_hist, trans_hist, cv2.HISTCMP_CHISQR)
        
        print(f"  📊 Color difference: {color_diff:.1f}")
        print(f"  📊 Histogram difference: {hist_diff:.1f}")
        
        if color_diff > 50:
            print("  🎯 Strong transformation applied ✅")
        elif color_diff > 20:
            print("  🎯 Moderate transformation applied ⚠️")
        else:
            print("  🎯 Minimal transformation applied ❌")

def main():
    print("🎬 CINEMA TRANSFORMATION TESTING")
    print("Choose an option:")
    print("1. Quick test transformation")
    print("2. Analyze transformation quality")
    print("3. Both")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        quick_test_transform()
    elif choice == "2":
        analyze_transformation_quality()
    elif choice == "3":
        quick_test_transform()
        analyze_transformation_quality()
    else:
        print("Invalid choice. Running quick test...")
        quick_test_transform()

if __name__ == "__main__":
    main()
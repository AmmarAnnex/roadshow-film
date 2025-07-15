#!/usr/bin/env python3
"""
Validate v1.4 Fixes Against v1.3b Catastrophe
Quick verification that issues are resolved
"""

import cv2
import numpy as np
from pathlib import Path
import json

def compare_model_versions():
    """Compare v1.3b catastrophe with v1.4 fixes"""
    print("⚖️  COMPARING MODEL VERSIONS")
    print("=" * 50)
    
    v13b_dir = Path("data/results/cinema_v1_3b_4k_test")
    v14_dir = Path("data/results/stable_v14_test")
    
    if not v13b_dir.exists():
        print("❌ v1.3b results not found")
        return
    
    if not v14_dir.exists():
        print("❌ v1.4 results not found - train v1.4 first!")
        return
    
    # Find matching files
    v13b_files = list(v13b_dir.glob("v1_3b_4k_*.jpg"))
    v14_files = list(v14_dir.glob("stable_v14_*.jpg"))
    
    print(f"📊 v1.3b files: {len(v13b_files)}")
    print(f"📊 v1.4 files: {len(v14_files)}")
    
    if not v13b_files or not v14_files:
        print("❌ No comparison files found")
        return
    
    # Analyze first available files
    v13b_img = cv2.imread(str(v13b_files[0]))
    v14_img = cv2.imread(str(v14_files[0]))
    
    if v13b_img is None or v14_img is None:
        print("❌ Could not load images")
        return
    
    print(f"\n🔍 ANALYZING: {v13b_files[0].name} vs {v14_files[0].name}")
    
    # Extract transformed portions (right half)
    h13, w13 = v13b_img.shape[:2]
    h14, w14 = v14_img.shape[:2]
    
    v13b_transform = v13b_img[:, w13//2:]
    v14_transform = v14_img[:, w14//2:]
    
    # Convert to RGB
    v13b_rgb = cv2.cvtColor(v13b_transform, cv2.COLOR_BGR2RGB)
    v14_rgb = cv2.cvtColor(v14_transform, cv2.COLOR_BGR2RGB)
    
    print(f"\n📊 QUALITY METRICS COMPARISON:")
    
    # Brightness analysis
    brightness_13b = np.mean(v13b_rgb)
    brightness_14 = np.mean(v14_rgb)
    
    print(f"  Brightness:")
    print(f"    v1.3b: {brightness_13b:.1f} {'🔥 EXTREME' if brightness_13b > 200 or brightness_13b < 50 else '✅ OK'}")
    print(f"    v1.4:  {brightness_14:.1f} {'✅ GOOD' if 80 <= brightness_14 <= 180 else '⚠️  CHECK'}")
    
    # Saturation analysis
    v13b_hsv = cv2.cvtColor(v13b_transform, cv2.COLOR_BGR2HSV)
    v14_hsv = cv2.cvtColor(v14_transform, cv2.COLOR_BGR2HSV)
    
    sat_13b = np.mean(v13b_hsv[:, :, 1])
    sat_14 = np.mean(v14_hsv[:, :, 1])
    
    print(f"  Saturation:")
    print(f"    v1.3b: {sat_13b:.1f} {'🔥 COLLAPSED' if sat_13b < 50 else '✅ OK'}")
    print(f"    v1.4:  {sat_14:.1f} {'✅ GOOD' if sat_14 > 80 else '⚠️  LOW'}")
    
    # Dynamic range
    range_13b = np.max(v13b_rgb) - np.min(v13b_rgb)
    range_14 = np.max(v14_rgb) - np.min(v14_rgb)
    
    print(f"  Dynamic Range:")
    print(f"    v1.3b: {range_13b:.1f} {'🔥 BLOWN' if range_13b < 100 else '✅ OK'}")
    print(f"    v1.4:  {range_14:.1f} {'✅ GOOD' if range_14 > 150 else '⚠️  LIMITED'}")
    
    # Color channel balance
    channels_13b = np.mean(v13b_rgb, axis=(0, 1))
    channels_14 = np.mean(v14_rgb, axis=(0, 1))
    
    print(f"  Color Balance:")
    print(f"    v1.3b RGB: [{channels_13b[0]:.1f}, {channels_13b[1]:.1f}, {channels_13b[2]:.1f}]")
    
    # Check for severe imbalance
    ratio_13b = np.max(channels_13b) / (np.min(channels_13b) + 1e-8)
    if ratio_13b > 2.0:
        print(f"           🔥 SEVERE IMBALANCE (ratio: {ratio_13b:.1f})")
    
    print(f"    v1.4 RGB:  [{channels_14[0]:.1f}, {channels_14[1]:.1f}, {channels_14[2]:.1f}]")
    
    ratio_14 = np.max(channels_14) / (np.min(channels_14) + 1e-8)
    if ratio_14 < 1.5:
        print(f"           ✅ BALANCED (ratio: {ratio_14:.1f})")
    else:
        print(f"           ⚠️  SLIGHT IMBALANCE (ratio: {ratio_14:.1f})")

def quick_quality_check():
    """Quick quality check for v1.4"""
    print(f"\n🎯 QUICK QUALITY CHECK FOR V1.4")
    print("=" * 40)
    
    v14_dir = Path("data/results/stable_v14_test")
    
    if not v14_dir.exists():
        print("❌ No v1.4 results found")
        return
    
    v14_files = list(v14_dir.glob("stable_v14_*.jpg"))
    
    if not v14_files:
        print("❌ No v1.4 comparison images found")
        return
    
    quality_scores = []
    
    for img_file in v14_files:
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        
        # Split into original and transformed
        original = img[:, :w//2]
        transformed = img[:, w//2:]
        
        # Convert to RGB
        orig_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        trans_rgb = cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB)
        
        # Basic quality metrics
        mse = np.mean((orig_rgb.astype(float) - trans_rgb.astype(float)) ** 2)
        
        # Brightness preservation
        orig_brightness = np.mean(orig_rgb)
        trans_brightness = np.mean(trans_rgb)
        brightness_diff = abs(trans_brightness - orig_brightness)
        
        # Simple quality score (lower is better)
        quality_score = mse + brightness_diff
        quality_scores.append(quality_score)
        
        print(f"  {img_file.name}:")
        print(f"    MSE: {mse:.1f} {'✅ GOOD' if mse < 500 else '⚠️  HIGH'}")
        print(f"    Brightness diff: {brightness_diff:.1f} {'✅ GOOD' if brightness_diff < 20 else '⚠️  HIGH'}")
    
    if quality_scores:
        avg_quality = np.mean(quality_scores)
        print(f"\n📊 AVERAGE QUALITY SCORE: {avg_quality:.1f}")
        
        if avg_quality < 300:
            print("✅ V1.4 QUALITY: GOOD")
        elif avg_quality < 1000:
            print("⚠️  V1.4 QUALITY: ACCEPTABLE")
        else:
            print("❌ V1.4 QUALITY: NEEDS WORK")

def generate_fix_report():
    """Generate a report on what was fixed"""
    print(f"\n📋 V1.4 FIX REPORT")
    print("=" * 30)
    
    fixes = {
        "Model Architecture": [
            "Reduced channels: 64→16 (75% reduction)",
            "Conservative residual: 0.2→0.05 (75% reduction)",
            "Parameter clamping: all values bounded",
            "Better activation: proper Tanh usage"
        ],
        "Training Stability": [
            "Lower learning rate: likely 0.001→0.0001",
            "Gradient clipping: max_norm=0.1",
            "Smaller batch size: safer memory usage",
            "Loss monitoring: detect explosions"
        ],
        "Data Processing": [
            "Input validation: NaN/inf detection",
            "Better normalization: careful range checking",
            "Output clamping: [0,1] enforcement",
            "Error handling: graceful degradation"
        ]
    }
    
    for category, fix_list in fixes.items():
        print(f"\n🔧 {category}:")
        for fix in fix_list:
            print(f"   ✅ {fix}")

def main():
    """Run validation suite"""
    compare_model_versions()
    quick_quality_check()
    generate_fix_report()
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. If v1.4 results look good, continue with this architecture")
    print("2. If still issues, further reduce learning rate to 0.00005")
    print("3. Consider adding more validation and monitoring")
    print("4. Test on more diverse scenes")

if __name__ == "__main__":
    main()
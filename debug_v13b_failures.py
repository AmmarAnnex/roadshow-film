#!/usr/bin/env python3
"""
Debug v1.3b Catastrophic Failures
Identify exactly what went wrong
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import json

def analyze_v13b_failures():
    """Analyze the catastrophic v1.3b results"""
    print("🔍 ANALYZING V1.3B CATASTROPHIC FAILURES")
    print("=" * 50)
    
    results_dir = Path("data/results/cinema_v1_3b_4k_test")
    
    if not results_dir.exists():
        print("❌ No v1.3b results found")
        return
    
    # Find comparison images
    comparison_files = list(results_dir.glob("v1_3b_4k_*.jpg"))
    
    if not comparison_files:
        print("❌ No comparison images found")
        return
    
    print(f"📊 Found {len(comparison_files)} comparison images")
    
    issues = {
        'overexposed': 0,
        'underexposed': 0,
        'color_shift': 0,
        'saturation_loss': 0,
        'artifacts': 0
    }
    
    for comp_file in comparison_files[:3]:  # Analyze first 3
        print(f"\n🔍 Analyzing: {comp_file.name}")
        
        # Load comparison image
        img = cv2.imread(str(comp_file))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        
        # Split into original and transformed
        original = img[:, :w//2]
        transformed = img[:, w//2:]
        
        # Convert to RGB for analysis
        orig_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        trans_rgb = cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB)
        
        # Analyze brightness
        orig_brightness = np.mean(orig_rgb)
        trans_brightness = np.mean(trans_rgb)
        brightness_ratio = trans_brightness / (orig_brightness + 1e-8)
        
        print(f"  Brightness: Original={orig_brightness:.1f}, Transformed={trans_brightness:.1f}")
        print(f"  Brightness Ratio: {brightness_ratio:.2f}")
        
        if brightness_ratio > 1.5:
            issues['overexposed'] += 1
            print("  ⚠️  OVEREXPOSED")
        elif brightness_ratio < 0.5:
            issues['underexposed'] += 1
            print("  ⚠️  UNDEREXPOSED")
        
        # Analyze saturation
        orig_hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
        trans_hsv = cv2.cvtColor(transformed, cv2.COLOR_BGR2HSV)
        
        orig_sat = np.mean(orig_hsv[:, :, 1])
        trans_sat = np.mean(trans_hsv[:, :, 1])
        sat_ratio = trans_sat / (orig_sat + 1e-8)
        
        print(f"  Saturation: Original={orig_sat:.1f}, Transformed={trans_sat:.1f}")
        print(f"  Saturation Ratio: {sat_ratio:.2f}")
        
        if sat_ratio < 0.5:
            issues['saturation_loss'] += 1
            print("  ⚠️  SEVERE SATURATION LOSS")
        
        # Check for extreme values
        if np.max(trans_rgb) > 250:
            issues['overexposed'] += 1
            print("  ⚠️  PIXEL VALUES CLIPPED HIGH")
        
        if np.mean(trans_rgb < 10) > 0.3:  # More than 30% very dark
            issues['underexposed'] += 1
            print("  ⚠️  EXCESSIVE DARK REGIONS")
        
        # Color channel analysis
        orig_channels = np.mean(orig_rgb, axis=(0, 1))
        trans_channels = np.mean(trans_rgb, axis=(0, 1))
        
        print(f"  Original RGB: [{orig_channels[0]:.1f}, {orig_channels[1]:.1f}, {orig_channels[2]:.1f}]")
        print(f"  Transformed RGB: [{trans_channels[0]:.1f}, {trans_channels[1]:.1f}, {trans_channels[2]:.1f}]")
        
        # Check for severe color shifts
        channel_ratios = trans_channels / (orig_channels + 1e-8)
        if np.any(channel_ratios > 2.0) or np.any(channel_ratios < 0.5):
            issues['color_shift'] += 1
            print("  ⚠️  SEVERE COLOR CHANNEL IMBALANCE")
    
    print(f"\n📊 ISSUE SUMMARY:")
    print(f"  Overexposed: {issues['overexposed']}")
    print(f"  Underexposed: {issues['underexposed']}")
    print(f"  Color Shift: {issues['color_shift']}")
    print(f"  Saturation Loss: {issues['saturation_loss']}")
    
    # Load model if available to check parameters
    model_path = Path("data/cinema_v1_3b_model.pth")
    if model_path.exists():
        print(f"\n🔍 ANALYZING V1.3B MODEL PARAMETERS:")
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            print(f"  Training Loss: {checkpoint.get('loss', 'unknown')}")
            print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
            
            # Check for suspicious parameter values
            state_dict = checkpoint['model_state_dict']
            
            for name, param in state_dict.items():
                if 'weight' in name or 'bias' in name:
                    param_mean = torch.mean(param).item()
                    param_std = torch.std(param).item()
                    param_max = torch.max(param).item()
                    param_min = torch.min(param).item()
                    
                    print(f"  {name}: mean={param_mean:.4f}, std={param_std:.4f}, range=[{param_min:.4f}, {param_max:.4f}]")
                    
                    # Flag suspicious values
                    if abs(param_mean) > 10:
                        print(f"    ⚠️  EXTREME MEAN VALUE")
                    if param_std > 5:
                        print(f"    ⚠️  HIGH VARIANCE")
                    if param_max > 20 or param_min < -20:
                        print(f"    ⚠️  EXTREME PARAMETER VALUES")
                        
        except Exception as e:
            print(f"  ❌ Error loading model: {e}")

def check_training_data_integrity():
    """Check if training data was corrupted"""
    print(f"\n🔍 CHECKING TRAINING DATA INTEGRITY:")
    
    data_path = Path("data/results/simple_depth_analysis/depth_metadata.json")
    
    if not data_path.exists():
        print("❌ No training metadata found")
        return
    
    with open(data_path, 'r') as f:
        pairs = json.load(f)
    
    print(f"📊 Found {len(pairs)} training pairs")
    
    # Check a few sample files
    for i, pair in enumerate(pairs[:3]):
        print(f"\n  Pair {i+1}:")
        print(f"    iPhone: {pair.get('iphone_file', 'missing')}")
        print(f"    Sony: {pair.get('sony_file', 'missing')}")
        
        # Check if files exist
        iphone_path = Path(pair['iphone_file'])
        sony_path = Path(pair['sony_file'])
        
        if not iphone_path.exists():
            print(f"    ❌ iPhone file missing")
        if not sony_path.exists():
            print(f"    ❌ Sony file missing")

def identify_root_causes():
    """Identify the most likely root causes"""
    print(f"\n🎯 ROOT CAUSE ANALYSIS:")
    print("=" * 30)
    
    print("MOST LIKELY CAUSES OF V1.3B FAILURE:")
    print()
    print("1. LEARNING RATE TOO HIGH:")
    print("   - Model diverged during training")
    print("   - Gradients exploded")
    print("   - Parameters became extreme")
    print()
    print("2. LOSS FUNCTION ISSUES:")
    print("   - Incorrect loss weighting")
    print("   - Unstable histogram loss")
    print("   - No gradient clipping")
    print()
    print("3. MODEL ARCHITECTURE PROBLEMS:")
    print("   - Residual connections wrong")
    print("   - Output not properly clamped")
    print("   - Activation functions inappropriate")
    print()
    print("4. DATA PREPROCESSING ERRORS:")
    print("   - Normalization incorrect")
    print("   - NaN/inf values in training")
    print("   - Target data corrupted")
    print()
    print("FIXES IMPLEMENTED IN V1.4:")
    print("✅ Reduced learning rate: 0.0001 (vs likely 0.001+ in v1.3b)")
    print("✅ Aggressive gradient clipping: max_norm=0.1")
    print("✅ Conservative model: 16 vs 64 channels")
    print("✅ Parameter clamping: all values bounded")
    print("✅ Better validation: NaN/inf checking")
    print("✅ Residual strength: start at 0.05 vs likely 0.2+")

def main():
    """Run complete failure analysis"""
    analyze_v13b_failures()
    check_training_data_integrity()
    identify_root_causes()
    
    print(f"\n🔧 RECOMMENDED ACTIONS:")
    print("1. Train stable v1.4 model immediately")
    print("2. Compare v1.4 results to v1.3b catastrophe")
    print("3. If v1.4 works, analyze what parameters differ")
    print("4. Never use v1.3b model again")

if __name__ == "__main__":
    main()
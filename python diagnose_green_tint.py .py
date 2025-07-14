#!/usr/bin/env python3
"""
Diagnose Green Tint Issue
Analyze what the perceptual model learned and why it's green
"""

import torch
import numpy as np
from pathlib import Path
import json

def analyze_model_parameters():
    """Analyze the learned parameters"""
    print("🔍 DIAGNOSING GREEN TINT ISSUE")
    print("=" * 50)
    
    model_path = Path("data/perceptual_cinema_model.pth")
    if not model_path.exists():
        print("❌ No model found!")
        return
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    print(f"📊 Model trained for {checkpoint['epoch']} epochs")
    print(f"📊 Final losses:")
    print(f"   Total: {checkpoint['loss']:.6f}")
    print(f"   MSE: {checkpoint['mse_loss']:.6f}")
    print(f"   Perceptual: {checkpoint['perceptual_loss']:.6f}")
    print(f"   Histogram: {checkpoint['hist_loss']:.6f}")
    
    # Load model to examine parameters
    from perceptual_loss_model import AdvancedCinemaTransform
    model = AdvancedCinemaTransform()
    model.load_state_dict(checkpoint['model_state_dict'])
    
    with torch.no_grad():
        print(f"\n🎨 LEARNED COLOR SCIENCE:")
        
        # Color matrix analysis
        color_matrix = model.color_matrix.data.numpy()
        print(f"Color Matrix:")
        print(f"  R: {color_matrix[0]}")
        print(f"  G: {color_matrix[1]}")  
        print(f"  B: {color_matrix[2]}")
        
        color_bias = model.color_bias.data.numpy()
        print(f"Color Bias: R={color_bias[0]:.3f}, G={color_bias[1]:.3f}, B={color_bias[2]:.3f}")
        
        # Tone curve
        print(f"\nTone Curve:")
        print(f"  Shadows: {model.shadows.item():.3f}")
        print(f"  Darks: {model.darks.item():.3f}")
        print(f"  Mids: {model.mids.item():.3f}")
        print(f"  Lights: {model.lights.item():.3f}")
        print(f"  Highlights: {model.highlights.item():.3f}")
        
        # Other parameters
        print(f"\nOther Adjustments:")
        print(f"  Contrast: {model.contrast.item():.3f}")
        print(f"  Saturation: {model.saturation.item():.3f}")
        print(f"  Detail Enhancement: {model.detail_enhance.item():.3f}")
        
        # Analyze the green bias
        print(f"\n🔍 GREEN TINT ANALYSIS:")
        
        # Check if green channel is being boosted
        r_strength = color_matrix[0, 0] + color_bias[0]
        g_strength = color_matrix[1, 1] + color_bias[1]
        b_strength = color_matrix[2, 2] + color_bias[2]
        
        print(f"Effective channel strength:")
        print(f"  Red: {r_strength:.3f}")
        print(f"  Green: {g_strength:.3f}")
        print(f"  Blue: {b_strength:.3f}")
        
        if g_strength > r_strength and g_strength > b_strength:
            print("  ❌ GREEN CHANNEL IS STRONGEST - causing green tint")
        elif g_strength > (r_strength + b_strength) / 2 * 1.1:
            print("  ⚠️ GREEN SLIGHTLY DOMINANT - mild green tint")
        else:
            print("  ✅ Balanced color channels")
        
        # Check cross-channel contamination
        print(f"\nCross-channel matrix values:")
        print(f"  R→G: {color_matrix[1, 0]:.3f}, R→B: {color_matrix[2, 0]:.3f}")
        print(f"  G→R: {color_matrix[0, 1]:.3f}, G→B: {color_matrix[2, 1]:.3f}")
        print(f"  B→R: {color_matrix[0, 2]:.3f}, B→G: {color_matrix[1, 2]:.3f}")
        
        # Potential fixes
        print(f"\n💡 POTENTIAL FIXES:")
        if model.saturation.item() < 0.9:
            print("1. Saturation too low - increase saturation loss weight")
        if abs(color_bias[1]) > 0.01:
            print("2. Green bias too high - constrain color bias")
        if g_strength > r_strength * 1.05:
            print("3. Green dominance - add color balance constraint")
        
        return {
            'color_matrix': color_matrix,
            'color_bias': color_bias,
            'saturation': model.saturation.item(),
            'contrast': model.contrast.item()
        }

def test_color_correction():
    """Test a simple color correction"""
    print(f"\n🧪 TESTING COLOR CORRECTION")
    
    # Simulate the learned transformation on a test color
    test_colors = np.array([
        [1.0, 0.0, 0.0],  # Pure red
        [0.0, 1.0, 0.0],  # Pure green  
        [0.0, 0.0, 1.0],  # Pure blue
        [0.5, 0.5, 0.5],  # Gray
        [1.0, 1.0, 1.0],  # White
    ])
    
    params = analyze_model_parameters()
    if params is None:
        return
    
    color_matrix = params['color_matrix']
    color_bias = params['color_bias']
    saturation = params['saturation']
    
    print(f"\nColor transformation test:")
    for i, color in enumerate(test_colors):
        # Apply color matrix
        transformed = np.dot(color_matrix, color) + color_bias
        
        # Apply saturation
        gray = np.mean(transformed)
        transformed = gray + saturation * (transformed - gray)
        
        # Clamp
        transformed = np.clip(transformed, 0, 1)
        
        color_names = ['Red', 'Green', 'Blue', 'Gray', 'White']
        print(f"  {color_names[i]}: {color} → {transformed}")

def suggest_fixes():
    """Suggest concrete fixes"""
    print(f"\n🔧 SUGGESTED FIXES:")
    print("1. **Reduce perceptual loss weight**: 0.5 → 0.3")
    print("2. **Add color balance constraint**: Penalize channel imbalance")
    print("3. **Increase saturation target**: Train to preserve more color")
    print("4. **Color matrix regularization**: Prevent extreme transformations")
    print("5. **White balance constraint**: Force gray to stay gray")
    
    print(f"\n🎯 **Quick Fix Test**: Reduce saturation parameter manually")
    print("Edit model and set saturation = 1.2 to counteract green")

if __name__ == "__main__":
    analyze_model_parameters()
    test_color_correction()
    suggest_fixes()
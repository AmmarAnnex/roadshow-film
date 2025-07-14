#!/usr/bin/env python3
"""
Debug Model v1.3b - Find What's Causing Desaturation
Test each component separately to isolate the issue
"""

import cv2
import numpy as np
from pathlib import Path
import torch
import rawpy
from cinema_model_v1_3b_4k_fixed import CinemaTransformV1_3b, load_all_luts

def test_model_components():
    """Test each component of the model separately"""
    print("🔍 DEBUGGING MODEL COMPONENTS")
    print("=" * 50)
    
    # Load model
    model_path = Path("data/cinema_v1_3b_4k_model.pth")
    if not model_path.exists():
        print("❌ No model found!")
        return
    
    device = torch.device('cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load LUTs
    lut_result = load_all_luts()
    reference_lut = None
    if checkpoint.get('has_lut', False) and lut_result is not None:
        reference_lut, all_luts = lut_result
    
    model = CinemaTransformV1_3b(reference_lut=reference_lut).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded")
    print(f"📊 LUT available: {reference_lut is not None}")
    
    # Load test image
    test_files = list(Path("data/training_pairs").glob("iphone_*.dng"))
    if not test_files:
        print("❌ No test images found!")
        return
    
    test_file = test_files[0]
    print(f"🖼️  Testing with: {test_file.name}")
    
    # Load image
    with rawpy.imread(str(test_file)) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            output_bps=16,
            no_auto_bright=True,
            user_flip=0
        )
    
    rgb_norm = np.clip(rgb.astype(np.float32) / 65535.0, 0, 1)
    
    # Resize for testing
    rgb_small = cv2.resize(rgb_norm, (512, 512))
    rgb_tensor = torch.FloatTensor(np.transpose(rgb_small, (2, 0, 1))).unsqueeze(0)
    
    print(f"📐 Input shape: {rgb_tensor.shape}")
    print(f"📊 Input stats: min={rgb_tensor.min():.3f}, max={rgb_tensor.max():.3f}, mean={rgb_tensor.mean():.3f}")
    
    with torch.no_grad():
        # Test step by step
        print(f"\n🔍 STEP-BY-STEP ANALYSIS:")
        
        # Step 1: Original
        x = rgb_tensor
        print(f"Original: min={x.min():.3f}, max={x.max():.3f}, mean={x.mean():.3f}")
        
        # Step 2: Color matrix
        x_matrix = model.apply_color_matrix(x)
        print(f"After color matrix: min={x_matrix.min():.3f}, max={x_matrix.max():.3f}, mean={x_matrix.mean():.3f}")
        
        # Step 3: Tone curve
        x_tone = model.apply_tone_curve(x_matrix)
        print(f"After tone curve: min={x_tone.min():.3f}, max={x_tone.max():.3f}, mean={x_tone.mean():.3f}")
        
        # Step 4: Color grading
        x_grading = model.apply_color_grading(x_tone)
        print(f"After color grading: min={x_grading.min():.3f}, max={x_grading.max():.3f}, mean={x_grading.mean():.3f}")
        
        # Step 5: LUT (if available)
        if reference_lut is not None:
            from cinema_model_v1_3b_4k_fixed import apply_lut_torch
            x_lut = apply_lut_torch(x_grading, reference_lut)
            print(f"After LUT: min={x_lut.min():.3f}, max={x_lut.max():.3f}, mean={x_lut.mean():.3f}")
            
            lut_blend = torch.clamp(model.lut_blend, 0.7, 0.9)
            x_blended = lut_blend * x_lut + (1 - lut_blend) * x_grading
            print(f"After LUT blend ({lut_blend.item():.3f}): min={x_blended.min():.3f}, max={x_blended.max():.3f}, mean={x_blended.mean():.3f}")
        else:
            x_blended = x_grading
            print(f"No LUT applied")
        
        # Step 6: ML enhancement
        x_final = model.ml_enhancement(x, x_blended)
        print(f"After ML enhancement: min={x_final.min():.3f}, max={x_final.max():.3f}, mean={x_final.mean():.3f}")
        
        # Check learned parameters
        print(f"\n📊 LEARNED PARAMETERS:")
        print(f"Saturation: {model.saturation.item():.3f}")
        print(f"Contrast: {model.contrast.item():.3f}")
        print(f"Warmth: {model.warmth.item():.3f}")
        print(f"ML residual strength: {model.ml_enhancement.residual_strength.item():.3f}")
        if reference_lut is not None:
            print(f"LUT blend: {model.lut_blend.item():.3f}")
        
        # Check color matrix
        print(f"\n🎨 COLOR MATRIX:")
        color_matrix = torch.clamp(model.color_matrix, 0.9, 1.1)
        print(f"R: [{color_matrix[0,0]:.3f}, {color_matrix[0,1]:.3f}, {color_matrix[0,2]:.3f}]")
        print(f"G: [{color_matrix[1,0]:.3f}, {color_matrix[1,1]:.3f}, {color_matrix[1,2]:.3f}]")
        print(f"B: [{color_matrix[2,0]:.3f}, {color_matrix[2,1]:.3f}, {color_matrix[2,2]:.3f}]")
        
        # Save debug images
        debug_dir = Path("data/debug_output")
        debug_dir.mkdir(exist_ok=True)
        
        def save_debug_image(tensor, name):
            img_np = tensor.squeeze(0).numpy()
            img_np = np.transpose(img_np, (1, 2, 0))
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            cv2.imwrite(str(debug_dir / f"{name}.jpg"), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        
        save_debug_image(x, "01_original")
        save_debug_image(x_matrix, "02_color_matrix")
        save_debug_image(x_tone, "03_tone_curve")
        save_debug_image(x_grading, "04_color_grading")
        if reference_lut is not None:
            save_debug_image(x_lut, "05_lut")
            save_debug_image(x_blended, "06_lut_blended")
        save_debug_image(x_final, "07_final")
        
        print(f"\n✅ Debug images saved to: {debug_dir}")
        print(f"📸 Check each step to see where desaturation occurs!")

def test_just_lut():
    """Test LUT application alone"""
    print(f"\n🎨 TESTING LUT ALONE")
    print("=" * 30)
    
    # Load LUT
    lut_result = load_all_luts()
    if lut_result is None:
        print("❌ No LUT found!")
        return
    
    reference_lut, _ = lut_result
    
    # Load test image
    test_files = list(Path("data/training_pairs").glob("iphone_*.dng"))
    test_file = test_files[0]
    
    with rawpy.imread(str(test_file)) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            output_bps=16,
            no_auto_bright=True,
            user_flip=0
        )
    
    rgb_norm = np.clip(rgb.astype(np.float32) / 65535.0, 0, 1)
    rgb_small = cv2.resize(rgb_norm, (512, 512))
    rgb_tensor = torch.FloatTensor(np.transpose(rgb_small, (2, 0, 1))).unsqueeze(0)
    
    # Apply JUST the LUT
    from cinema_model_v1_3b_4k_fixed import apply_lut_torch
    with torch.no_grad():
        lut_only = apply_lut_torch(rgb_tensor, reference_lut)
    
    print(f"Original: min={rgb_tensor.min():.3f}, max={rgb_tensor.max():.3f}, mean={rgb_tensor.mean():.3f}")
    print(f"LUT only: min={lut_only.min():.3f}, max={lut_only.max():.3f}, mean={lut_only.mean():.3f}")
    
    # Save comparison
    debug_dir = Path("data/debug_output")
    debug_dir.mkdir(exist_ok=True)
    
    orig_np = rgb_tensor.squeeze(0).numpy()
    orig_np = np.transpose(orig_np, (1, 2, 0))
    orig_np = np.clip(orig_np * 255, 0, 255).astype(np.uint8)
    
    lut_np = lut_only.squeeze(0).numpy()
    lut_np = np.transpose(lut_np, (1, 2, 0))
    lut_np = np.clip(lut_np * 255, 0, 255).astype(np.uint8)
    
    comparison = np.hstack([orig_np, lut_np])
    cv2.imwrite(str(debug_dir / "lut_only_comparison.jpg"), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    
    print(f"✅ LUT-only comparison saved to: debug_output/lut_only_comparison.jpg")

if __name__ == "__main__":
    test_model_components()
    test_just_lut()
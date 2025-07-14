#!/usr/bin/env python3
"""
Emergency Fix v1.3b - Fix RAW Processing and LUT Issues
"""

import cv2
import numpy as np
from pathlib import Path
import torch
import rawpy

def process_image_correctly(file_path: str):
    """Process image with proper exposure"""
    try:
        with rawpy.imread(file_path) as raw:
            # Try multiple processing approaches
            
            # Approach 1: Auto brightness (like iPhone does)
            rgb_auto = raw.postprocess(
                use_camera_wb=True,
                output_bps=16,
                no_auto_bright=False,  # ← Enable auto brightness!
                user_flip=0
            )
            
            # Approach 2: Manual brightness boost
            rgb_bright = raw.postprocess(
                use_camera_wb=True,
                output_bps=16,
                no_auto_bright=True,
                user_flip=0
            )
            
            # Check which one looks better
            auto_norm = rgb_auto.astype(np.float32) / 65535.0
            bright_norm = rgb_bright.astype(np.float32) / 65535.0
            
            # Apply 2x exposure boost to manual
            bright_boosted = np.clip(bright_norm * 2.0, 0, 1)
            
            print(f"Auto brightness: mean={auto_norm.mean():.3f}, max={auto_norm.max():.3f}")
            print(f"Manual (original): mean={bright_norm.mean():.3f}, max={bright_norm.max():.3f}")
            print(f"Manual (2x boost): mean={bright_boosted.mean():.3f}, max={bright_boosted.max():.3f}")
            
            return auto_norm, bright_norm, bright_boosted
            
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None

def test_lut_without_crushing():
    """Test if LUT works better with proper exposure"""
    print("🔍 TESTING LUT WITH PROPER EXPOSURE")
    print("=" * 50)
    
    # Load test image
    test_files = list(Path("data/training_pairs").glob("iphone_*.dng"))
    test_file = test_files[0]
    
    # Process with different approaches
    auto, manual, boosted = process_image_correctly(str(test_file))
    
    if auto is None:
        return
    
    # Load LUT
    from cinema_model_v1_3b_4k_fixed import load_all_luts, apply_lut_torch
    lut_result = load_all_luts()
    if lut_result is None:
        print("No LUT found!")
        return
    
    reference_lut, _ = lut_result
    
    # Test LUT on each version
    debug_dir = Path("data/debug_exposure")
    debug_dir.mkdir(exist_ok=True)
    
    for name, img in [("auto", auto), ("manual", manual), ("boosted", boosted)]:
        # Resize for testing
        img_small = cv2.resize(img, (512, 512))
        img_tensor = torch.FloatTensor(np.transpose(img_small, (2, 0, 1))).unsqueeze(0)
        
        # Apply LUT
        with torch.no_grad():
            lut_result = apply_lut_torch(img_tensor, reference_lut)
        
        # Convert back
        lut_np = lut_result.squeeze(0).numpy()
        lut_np = np.transpose(lut_np, (1, 2, 0))
        
        print(f"{name} → LUT: {img_tensor.mean():.3f} → {lut_result.mean():.3f}")
        
        # Save comparison
        orig_display = np.clip(img_small * 255, 0, 255).astype(np.uint8)
        lut_display = np.clip(lut_np * 255, 0, 255).astype(np.uint8)
        
        comparison = np.hstack([orig_display, lut_display])
        cv2.imwrite(str(debug_dir / f"lut_test_{name}.jpg"), 
                   cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    
    print(f"✅ Exposure tests saved to: {debug_dir}")

def quick_model_fix():
    """Test model without LUT"""
    print("\n🔧 TESTING MODEL WITHOUT LUT")
    print("=" * 40)
    
    # Load model but force no LUT
    from cinema_model_v1_3b_4k_fixed import CinemaTransformV1_3b
    
    model_path = Path("data/cinema_v1_3b_4k_model.pth")
    if not model_path.exists():
        print("No model found!")
        return
    
    device = torch.device('cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model WITHOUT LUT
    model = CinemaTransformV1_3b(reference_lut=None).to(device)  # ← No LUT!
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test image with proper exposure
    test_files = list(Path("data/training_pairs").glob("iphone_*.dng"))
    test_file = test_files[0]
    
    auto, manual, boosted = process_image_correctly(str(test_file))
    
    # Test model on properly exposed image
    img_small = cv2.resize(auto, (512, 512))  # Use auto-brightness version
    img_tensor = torch.FloatTensor(np.transpose(img_small, (2, 0, 1))).unsqueeze(0)
    
    with torch.no_grad():
        # Boost the learned parameters
        original_sat = model.saturation.data.clone()
        original_contrast = model.contrast.data.clone()
        original_ml = model.ml_enhancement.residual_strength.data.clone()
        
        # Temporary boost
        model.saturation.data = torch.tensor(1.5)  # More saturation
        model.contrast.data = torch.tensor(1.1)    # More contrast  
        model.ml_enhancement.residual_strength.data = torch.tensor(0.3)  # Stronger ML
        
        result = model(img_tensor)
        
        # Restore original values
        model.saturation.data = original_sat
        model.contrast.data = original_contrast
        model.ml_enhancement.residual_strength.data = original_ml
    
    print(f"Input: mean={img_tensor.mean():.3f}, max={img_tensor.max():.3f}")
    print(f"Output: mean={result.mean():.3f}, max={result.max():.3f}")
    
    # Save comparison
    debug_dir = Path("data/debug_exposure")
    debug_dir.mkdir(exist_ok=True)
    
    orig_display = np.clip(img_small * 255, 0, 255).astype(np.uint8)
    
    result_np = result.squeeze(0).numpy()
    result_np = np.transpose(result_np, (1, 2, 0))
    result_display = np.clip(result_np * 255, 0, 255).astype(np.uint8)
    
    comparison = np.hstack([orig_display, result_display])
    cv2.imwrite(str(debug_dir / "model_no_lut_boosted.jpg"), 
               cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    
    print(f"✅ Model test (no LUT, boosted params) saved")

if __name__ == "__main__":
    test_lut_without_crushing()
    quick_model_fix()
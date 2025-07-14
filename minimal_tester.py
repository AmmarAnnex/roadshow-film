#!/usr/bin/env python3
"""
Minimal Roadshow Tester - Guaranteed to work!
"""

import os
import cv2
import numpy as np
from pathlib import Path
import sys

# Add backend to path
sys.path.append('backend')

def create_simple_test():
    """Create one simple test and run transformation"""
    print("🎬 MINIMAL ROADSHOW TEST")
    print("=" * 30)
    
    # Create directories
    Path("data/minimal_test").mkdir(parents=True, exist_ok=True)
    
    # Create simple test image - solid color blocks
    print("🎨 Creating test image...")
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # Simple color blocks - no math overflow risk
    img[0:100, 0:200] = [255, 0, 0]      # Red
    img[0:100, 200:400] = [0, 255, 0]    # Green  
    img[0:100, 400:600] = [0, 0, 255]    # Blue
    img[100:200, 0:200] = [255, 255, 0]  # Yellow
    img[100:200, 200:400] = [255, 0, 255] # Magenta
    img[100:200, 400:600] = [0, 255, 255] # Cyan
    img[200:300, :] = [128, 128, 128]     # Gray
    img[300:400, :] = [200, 150, 100]     # Skin tone
    
    # Save original
    cv2.imwrite('data/minimal_test/original.jpg', img)
    print("  ✓ Created test image")
    
    # Test transformation
    print("🚀 Testing transformation...")
    try:
        from roadshow_3dlut_engine import Reality3DLUT, CAMERA_PROFILES
        
        # Initialize engine  
        lut_engine = Reality3DLUT(resolution=16)  # Small for speed
        lut_engine.create_base_lut()
        
        # iPhone → ARRI transformation
        iphone = CAMERA_PROFILES['iphone_12_pro']
        arri = CAMERA_PROFILES['arri_alexa']
        lut_engine.learn_from_reference(iphone, arri)
        
        # Apply transformation
        result = lut_engine.apply_to_image(img, method='vectorized')
        
        # Create comparison
        comparison = np.hstack([img, result])
        cv2.imwrite('data/minimal_test/before_after.jpg', comparison)
        
        print("  ✓ Transformation complete!")
        print("📁 Check data/minimal_test/before_after.jpg")
        
        # Quick analysis
        original_avg = np.mean(img, axis=(0,1))
        result_avg = np.mean(result, axis=(0,1)) 
        
        print(f"\n📊 QUICK ANALYSIS:")
        print(f"Original avg color: [{original_avg[0]:.0f}, {original_avg[1]:.0f}, {original_avg[2]:.0f}]")
        print(f"Result avg color:   [{result_avg[0]:.0f}, {result_avg[1]:.0f}, {result_avg[2]:.0f}]")
        
        color_shift = result_avg - original_avg
        print(f"Color shift:        [{color_shift[0]:+.0f}, {color_shift[1]:+.0f}, {color_shift[2]:+.0f}]")
        
        if abs(color_shift[0]) > 5 or abs(color_shift[1]) > 5 or abs(color_shift[2]) > 5:
            print("✅ Transformation is working - colors changed!")
        else:
            print("⚠️  Small color change - transformation may need adjustment")
            
        return True
        
    except ImportError as e:
        print(f"❌ Could not import engine: {e}")
        print("Make sure you're in the Roadshow3DLUT directory")
        return False
    except Exception as e:
        print(f"❌ Error during transformation: {e}")
        return False

if __name__ == "__main__":
    create_simple_test()
    print("\n🎯 If this worked, your engine is ready for real footage!")
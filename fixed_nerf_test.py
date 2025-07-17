#!/usr/bin/env python3
"""
NeRF Test Scene - Validate Installation
Fixed Unicode encoding for Windows
"""

import numpy as np
import torch
import cv2
from pathlib import Path

def create_synthetic_test_scene():
    """Create synthetic test data for NeRF validation"""
    print("CREATING TEST SCENE")
    print("=" * 20)
    
    # Create test directory
    test_dir = Path("data/nerf_test_scene")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic camera poses (spiral)
    num_views = 20
    poses = []
    
    for i in range(num_views):
        angle = 2 * np.pi * i / num_views
        
        # Camera position (spiral)
        x = 2 * np.cos(angle)
        y = 2 * np.sin(angle) 
        z = 0.5 * np.sin(2 * angle)
        
        # Look at origin
        camera_pos = np.array([x, y, z])
        target = np.array([0, 0, 0])
        up = np.array([0, 0, 1])
        
        # Create pose matrix
        forward = target - camera_pos
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        
        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = -forward
        pose[:3, 3] = camera_pos
        
        poses.append(pose)
    
    # Save poses
    poses_array = np.array(poses)
    np.save(test_dir / "poses.npy", poses_array)
    
    # Create synthetic images (simple colored spheres)
    for i in range(num_views):
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        
        # Add colored sphere
        center = (200, 200)
        radius = 80
        color = (int(255 * (i / num_views)), 100, 255 - int(255 * (i / num_views)))
        
        cv2.circle(img, center, radius, color, -1)
        
        # Save image
        cv2.imwrite(str(test_dir / f"image_{i:03d}.png"), img)
    
    print(f"Test scene created: {test_dir}")
    print(f"   - {num_views} synthetic views")
    print(f"   - Camera poses saved")
    return test_dir

def test_nerf_pipeline(test_dir):
    """Test NeRF pipeline with synthetic data"""
    print("\nTESTING NeRF PIPELINE")
    print("=" * 25)
    
    try:
        # Test basic NeRF components
        from production_nerf_architecture import NeRFNetwork, PositionalEncoding
        
        # Initialize network
        nerf = NeRFNetwork()
        print("NeRF network initialized")
        
        # Test positional encoding
        pos_enc = PositionalEncoding()
        test_pos = torch.randn(100, 3)
        test_dirs = torch.randn(100, 3)
        
        encoded_pos = pos_enc.encode_position(test_pos)
        encoded_dirs = pos_enc.encode_direction(test_dirs)
        
        print(f"Positional encoding: {test_pos.shape} -> {encoded_pos.shape}")
        
        # Test forward pass
        with torch.no_grad():
            colors, densities = nerf(test_pos, test_dirs)
        
        print(f"Forward pass: colors {colors.shape}, densities {densities.shape}")
        
        return True
        
    except Exception as e:
        print(f"NeRF pipeline test failed: {e}")
        return False

def test_nerfstudio_installation():
    """Test NeRFStudio installation"""
    print("\nTESTING NERFSTUDIO")
    print("=" * 20)
    
    try:
        import nerfstudio
        print(f"NeRFStudio version: {nerfstudio.__version__}")
        
        # Test basic imports
        from nerfstudio.models.nerfacto import NerfactoModel
        from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
        
        print("NeRFStudio core imports successful")
        print("Ready for real scene training!")
        return True
        
    except Exception as e:
        print(f"NeRFStudio test failed: {e}")
        return False

if __name__ == "__main__":
    print("ROADSHOW NeRF INSTALLATION TEST")
    print("=" * 35)
    
    test_dir = create_synthetic_test_scene()
    
    # Test custom NeRF architecture
    nerf_success = test_nerf_pipeline(test_dir)
    
    # Test NeRFStudio installation
    nerfstudio_success = test_nerfstudio_installation()
    
    print("\n" + "=" * 35)
    print("TEST RESULTS:")
    print(f"Custom NeRF Architecture: {'PASS' if nerf_success else 'FAIL'}")
    print(f"NeRFStudio Installation: {'PASS' if nerfstudio_success else 'FAIL'}")
    
    if nerf_success and nerfstudio_success:
        print("\nNeRF TEST PASSED!")
        print("Ready for real scene reconstruction")
        print("\nNext Steps:")
        print("1. Rename your 23 new training pairs (057-079)")
        print("2. Train v1.5 with 79 pairs")
        print("3. Set up NeRF scene for synthetic data generation")
        print("4. Implement super-resolution + bit-depth expansion")
    else:
        print("\nSome tests failed - check installation")

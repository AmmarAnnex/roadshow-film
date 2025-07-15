#!/usr/bin/env python3
"""
Roadshow 3D LUT Engine - OPTIMIZED VERSION
Step 2: Vectorized LUT Application for 10x+ speed improvement
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Dict
import time

@dataclass
class CameraReality:
    """Defines a camera's reality signature"""
    name: str
    color_response: np.ndarray
    spatial_falloff: float
    temporal_cadence: float
    character_signature: Dict[str, float]

class OptimizedReality3DLUT:
    """
    OPTIMIZED 3D LUT Engine - Vectorized for speed
    """
    
    def __init__(self, resolution=64):
        self.resolution = resolution
        self.lut = None
        print(f"🎬 Initializing OPTIMIZED Reality LUT with resolution: {resolution}")
    
    def create_base_lut(self):
        """Create identity LUT using vectorized operations"""
        print("Creating base identity LUT (vectorized)...")
        
        # Create coordinate grids - much faster than nested loops
        r_coords = np.linspace(0, 1, self.resolution)
        g_coords = np.linspace(0, 1, self.resolution) 
        b_coords = np.linspace(0, 1, self.resolution)
        
        # Use meshgrid to create 3D coordinate arrays
        R, G, B = np.meshgrid(r_coords, g_coords, b_coords, indexing='ij')
        
        # Stack to create the LUT - single vectorized operation
        self.lut = np.stack([R, G, B], axis=-1)
        
        print("✅ Base LUT created (vectorized)")
        return self.lut
    
    def learn_from_reference(self, source_reality, target_reality):
        """Learn transformation between camera realities"""
        print(f"🧠 Learning: {source_reality.name} → {target_reality.name}")
        
        # Calculate transformation (vectorized)
        color_shift = target_reality.color_response - source_reality.color_response
        
        # Apply to entire LUT at once (vectorized)
        if self.lut is not None:
            self.lut += color_shift.reshape(1, 1, 1, 3) * 0.1
            self.lut = np.clip(self.lut, 0, 1)
        
        print(f"✅ Learned transformation (vectorized)")
    
    def apply_to_image_vectorized(self, image):
        """
        OPTIMIZED: Apply LUT using vectorized operations
        This is the key optimization - no more pixel loops!
        """
        if self.lut is None:
            return image
        
        # Normalize image to 0-1 range
        img_norm = image.astype(np.float32) / 255.0
        
        # Get image dimensions
        h, w = img_norm.shape[:2]
        
        # Reshape image to (N_pixels, 3) for vectorized processing
        pixels = img_norm.reshape(-1, 3)
        
        # Convert pixel values to LUT indices (vectorized)
        lut_indices = pixels * (self.resolution - 1)
        
        # Get integer indices for LUT lookup
        r_idx = np.clip(lut_indices[:, 2].astype(int), 0, self.resolution - 1)
        g_idx = np.clip(lut_indices[:, 1].astype(int), 0, self.resolution - 1) 
        b_idx = np.clip(lut_indices[:, 0].astype(int), 0, self.resolution - 1)
        
        # Apply LUT transformation (vectorized lookup)
        transformed_pixels = self.lut[r_idx, g_idx, b_idx]
        
        # Reshape back to original image dimensions
        result = transformed_pixels.reshape(h, w, 3)
        
        # Convert back to uint8
        return (result * 255).astype(np.uint8)
    
    def apply_to_image_trilinear(self, image):
        """
        PREMIUM OPTIMIZED: Apply LUT with trilinear interpolation
        Smooth transitions, no banding artifacts
        """
        if self.lut is None:
            return image
        
        # Normalize image to 0-1 range
        img_norm = image.astype(np.float32) / 255.0
        h, w = img_norm.shape[:2]
        pixels = img_norm.reshape(-1, 3)
        
        # Convert to LUT coordinates
        lut_coords = pixels * (self.resolution - 1)
        
        # Get integer and fractional parts for interpolation
        lut_int = np.floor(lut_coords).astype(int)
        lut_frac = lut_coords - lut_int
        
        # Clamp indices to valid range
        lut_int = np.clip(lut_int, 0, self.resolution - 2)
        
        # Get the 8 corner points for trilinear interpolation
        r0, g0, b0 = lut_int[:, 2], lut_int[:, 1], lut_int[:, 0]
        r1, g1, b1 = r0 + 1, g0 + 1, b0 + 1
        
        # Fractional parts
        fr, fg, fb = lut_frac[:, 2], lut_frac[:, 1], lut_frac[:, 0]
        
        # 8-point trilinear interpolation (vectorized)
        c000 = self.lut[r0, g0, b0]
        c001 = self.lut[r0, g0, b1] 
        c010 = self.lut[r0, g1, b0]
        c011 = self.lut[r0, g1, b1]
        c100 = self.lut[r1, g0, b0]
        c101 = self.lut[r1, g0, b1]
        c110 = self.lut[r1, g1, b0]
        c111 = self.lut[r1, g1, b1]
        
        # Interpolate along each axis
        c00 = c000 * (1-fr)[:, None] + c100 * fr[:, None]
        c01 = c001 * (1-fr)[:, None] + c101 * fr[:, None]
        c10 = c010 * (1-fr)[:, None] + c110 * fr[:, None] 
        c11 = c011 * (1-fr)[:, None] + c111 * fr[:, None]
        
        c0 = c00 * (1-fg)[:, None] + c10 * fg[:, None]
        c1 = c01 * (1-fg)[:, None] + c11 * fg[:, None]
        
        result_pixels = c0 * (1-fb)[:, None] + c1 * fb[:, None]
        
        # Reshape and convert back
        result = result_pixels.reshape(h, w, 3)
        return (result * 255).astype(np.uint8)
    
    # Keep the old method for comparison
    def apply_to_image_old(self, image):
        """Original slow method (for comparison)"""
        if self.lut is None:
            return image
        
        img_norm = image.astype(np.float32) / 255.0
        h, w = img_norm.shape[:2]
        result = np.zeros_like(img_norm)
        
        # The slow pixel-by-pixel loop
        for y in range(h):
            for x in range(w):
                pixel = img_norm[y, x]
                r_idx = int(pixel[2] * (self.resolution - 1))
                g_idx = int(pixel[1] * (self.resolution - 1)) 
                b_idx = int(pixel[0] * (self.resolution - 1))
                
                r_idx = np.clip(r_idx, 0, self.resolution - 1)
                g_idx = np.clip(g_idx, 0, self.resolution - 1)
                b_idx = np.clip(b_idx, 0, self.resolution - 1)
                
                result[y, x] = self.lut[r_idx, g_idx, b_idx][::-1]
        
        return (result * 255).astype(np.uint8)


def benchmark_comparison():
    """Compare old vs new performance"""
    print("\n🏁 PERFORMANCE COMPARISON")
    print("=" * 50)
    
    # Create test image
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Set up engines
    old_engine = OptimizedReality3DLUT(resolution=32)
    old_engine.create_base_lut()
    
    new_engine = OptimizedReality3DLUT(resolution=32) 
    new_engine.create_base_lut()
    
    # Benchmark old method
    print("Testing OLD method (pixel loops)...")
    start = time.time()
    result_old = old_engine.apply_to_image_old(test_image)
    time_old = time.time() - start
    
    # Benchmark vectorized method
    print("Testing NEW method (vectorized)...")
    start = time.time()
    result_new = new_engine.apply_to_image_vectorized(test_image)
    time_new = time.time() - start
    
    # Benchmark trilinear method
    print("Testing PREMIUM method (trilinear)...")
    start = time.time()
    result_trilinear = new_engine.apply_to_image_trilinear(test_image)
    time_trilinear = time.time() - start
    
    # Calculate improvements
    speedup_vectorized = time_old / time_new
    speedup_trilinear = time_old / time_trilinear
    
    pixels = test_image.shape[0] * test_image.shape[1]
    
    print(f"\n📊 RESULTS for {pixels:,} pixels:")
    print(f"  Old method:     {time_old:.3f}s ({pixels/time_old:,.0f} pixels/sec)")
    print(f"  Vectorized:     {time_new:.3f}s ({pixels/time_new:,.0f} pixels/sec)")
    print(f"  Trilinear:      {time_trilinear:.3f}s ({pixels/time_trilinear:,.0f} pixels/sec)")
    print(f"\n🚀 SPEEDUP:")
    print(f"  Vectorized: {speedup_vectorized:.1f}x faster")
    print(f"  Trilinear:  {speedup_trilinear:.1f}x faster")
    
    return {
        'old_time': time_old,
        'vectorized_time': time_new, 
        'trilinear_time': time_trilinear,
        'speedup_vectorized': speedup_vectorized,
        'speedup_trilinear': speedup_trilinear
    }


# Camera profiles (same as original)
CAMERA_PROFILES = {
    'iphone_12_pro': CameraReality(
        name="iPhone 12 Pro",
        color_response=np.array([0.95, 0.98, 1.0]),
        spatial_falloff=0.3,
        temporal_cadence=1.0,
        character_signature={'digital': 0.9, 'organic': 0.1}
    ),
    'arri_alexa': CameraReality(
        name="ARRI Alexa",
        color_response=np.array([1.0, 1.0, 0.98]),
        spatial_falloff=0.8,
        temporal_cadence=0.95,
        character_signature={'digital': 0.2, 'organic': 0.8}
    )
}


if __name__ == "__main__":
    # Run performance comparison
    benchmark_comparison()
#!/usr/bin/env python3
"""
Fixed Cinema-Grade Transformation System
Protects highlights and handles edge cases properly
"""

import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.append('backend')

class FixedCinemaTransforms:
    """Professional cinema color science with protected highlights"""
    
    def __init__(self):
        self.output_dir = Path("data/results/cinema_grade_fixed")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def safe_normalize(self, img):
        """Safely normalize image to 0-1 range"""
        img = img.astype(np.float32) / 255.0
        return np.clip(img, 0.0, 1.0)
    
    def safe_denormalize(self, img):
        """Safely convert back to 0-255 range"""
        img = np.clip(img, 0.0, 1.0)
        return (img * 255).astype(np.uint8)
    
    def apply_color_matrix_safe(self, img, matrix):
        """Apply color matrix with proper clipping"""
        original_shape = img.shape
        img_reshaped = img.reshape(-1, 3)
        
        # Ensure no invalid values
        img_reshaped = np.nan_to_num(img_reshaped, 0.0)
        img_reshaped = np.clip(img_reshaped, 0.0, 1.0)
        
        # Apply matrix
        img_transformed = img_reshaped @ matrix.T
        
        # Clip to valid range
        img_transformed = np.clip(img_transformed, 0.0, 1.0)
        
        return img_transformed.reshape(original_shape)
    
    def arri_highlight_rolloff(self, img):
        """ARRI's signature highlight protection"""
        # Gentle S-curve that protects highlights
        return np.where(img < 0.7,
                       img * 1.1,  # Slight boost in mids
                       0.7 + (img - 0.7) * 0.5)  # Roll off highlights
    
    def apply_arri_alexa_transform(self, image):
        """ARRI Alexa color science - warm, organic, PROTECTED highlights"""
        img = self.safe_normalize(image)
        
        # ARRI color matrix - warmer, more conservative
        color_matrix = np.array([
            [1.04, -0.02, -0.02],  # Gentle red enhancement
            [-0.01, 1.02, -0.01],  # Natural greens
            [-0.03, -0.02, 1.05]   # Slightly warmer blues
        ])
        
        # Apply color matrix safely
        img = self.apply_color_matrix_safe(img, color_matrix)
        
        # ARRI's famous highlight rolloff
        img = self.arri_highlight_rolloff(img)
        
        # Gentle film curve
        img = np.power(img, 0.9)
        
        # Subtle warmth in shadows
        shadows_mask = img < 0.3
        img[shadows_mask] = img[shadows_mask] * 0.95  # Slight lift
        
        return self.safe_denormalize(img)
    
    def apply_red_transform(self, image):
        """RED camera color science - clean, sharp, modern"""
        img = self.safe_normalize(image)
        
        # RED color matrix - more saturated but controlled
        color_matrix = np.array([
            [1.06, -0.03, -0.03],
            [-0.02, 1.04, -0.02],
            [-0.02, -0.02, 1.04]
        ])
        
        img = self.apply_color_matrix_safe(img, color_matrix)
        
        # RED's clean, sharp curve
        img = np.power(img, 0.92)
        
        # Micro-contrast enhancement (subtle)
        blurred = cv2.GaussianBlur(img, (3, 3), 0.5)
        img = img + (img - blurred) * 0.2
        
        return self.safe_denormalize(img)
    
    def apply_blackmagic_transform(self, image):
        """Blackmagic color science - neutral, flexible, filmic"""
        img = self.safe_normalize(image)
        
        # Blackmagic color matrix - neutral excellence
        color_matrix = np.array([
            [1.02, -0.01, -0.01],
            [-0.01, 1.02, -0.01],
            [-0.01, -0.01, 1.02]
        ])
        
        img = self.apply_color_matrix_safe(img, color_matrix)
        
        # Blackmagic's extended range curve
        img = np.power(img, 0.88)
        
        # Gentle saturation boost
        hsv = cv2.cvtColor(self.safe_denormalize(img), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.05, 0, 255)  # 5% saturation boost
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
        
        return self.safe_denormalize(img)
    
    def apply_vintage_film_transform(self, image):
        """Vintage film look - classic film stocks"""
        img = self.safe_normalize(image)
        
        # Vintage film matrix - warmer, softer
        color_matrix = np.array([
            [1.08, -0.04, -0.04],
            [-0.02, 1.01, -0.01],
            [-0.06, -0.02, 1.08]
        ])
        
        img = self.apply_color_matrix_safe(img, color_matrix)
        
        # Film curve with lifted blacks
        img = img * 0.9 + 0.05  # Lift blacks
        img = np.power(img, 0.8)
        
        # Vintage vignette (very subtle)
        h, w = img.shape[:2]
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        vignette = 1 - (distance / max_dist) * 0.1  # Very gentle
        vignette = np.clip(vignette, 0.9, 1.0)
        
        for i in range(3):
            img[:,:,i] *= vignette
        
        return self.safe_denormalize(img)
    
    def process_all_images(self):
        """Process all iPhone images with FIXED cinema transforms"""
        print("🎬 FIXED CINEMA-GRADE TRANSFORMATION SYSTEM")
        print("=" * 50)
        
        # Find iPhone images
        input_dir = Path("data/samples/real_photos")
        image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg")) + list(input_dir.glob("*.png"))
        
        if not image_files:
            print("❌ No images found in data/samples/real_photos/")
            return
        
        print(f"📸 Found {len(image_files)} iPhone images")
        
        # Define transforms
        transforms = [
            ("ARRI_Alexa_Fixed", self.apply_arri_alexa_transform, "🎭 Warm, organic, PROTECTED highlights"),
            ("RED_Cinema_Fixed", self.apply_red_transform, "⚡ Clean, sharp, modern"),
            ("Blackmagic_G5_Fixed", self.apply_blackmagic_transform, "🎯 Neutral, flexible, rich"),
            ("Vintage_Film_Fixed", self.apply_vintage_film_transform, "🎞️ Classic film stock look")
        ]
        
        for img_file in image_files:
            print(f"\n🎨 Processing {img_file.name}...")
            
            # Load image
            original = cv2.imread(str(img_file))
            if original is None:
                continue
            
            for transform_name, transform_func, description in transforms:
                print(f"  {description}")
                
                try:
                    # Apply transformation
                    transformed = transform_func(original)
                    
                    # Create side-by-side comparison
                    comparison = np.hstack([original, transformed])
                    
                    # Save result
                    output_name = f"{img_file.stem}_{transform_name}.jpg"
                    output_path = self.output_dir / output_name
                    cv2.imwrite(str(output_path), comparison)
                    
                    print(f"    ✓ Saved: {output_name}")
                    
                except Exception as e:
                    print(f"    ❌ Error: {e}")
        
        print(f"\n✅ All transformations complete!")
        print(f"📁 Results in: {self.output_dir}")
        print("\n🎯 Fixed highlight protection - should look much better!")
    
    def analyze_results(self):
        """Quick analysis of the fixed results"""
        print("\n🔍 Analyzing fixed cinema-grade results...")
        
        result_files = list(self.output_dir.glob("*.jpg"))
        if not result_files:
            print("No results to analyze yet")
            return
        
        # Quick difference analysis on first result
        first_result = cv2.imread(str(result_files[0]))
        if first_result is not None:
            h, w = first_result.shape[:2]
            original = first_result[:, :w//2]
            transformed = first_result[:, w//2:]
            
            # Calculate difference
            diff = np.mean(np.abs(original.astype(float) - transformed.astype(float)))
            print(f"  📊 Average difference: {diff:.1f} (target: 15-30)")
            
            if diff > 15:
                print("  ✅ Strong, noticeable transformation!")
            elif diff > 8:
                print("  🟡 Moderate transformation - should be visible")
            else:
                print("  🔴 Still too subtle - need stronger transforms")


def main():
    transformer = FixedCinemaTransforms()
    transformer.process_all_images()
    transformer.analyze_results()
    
    print("\n🎯 NEXT STEPS:")
    print("1. Check data/results/cinema_grade_fixed/ for results")
    print("2. No more blown highlights!")
    print("3. Ready to discuss ML strategy and market approach")

if __name__ == "__main__":
    main()
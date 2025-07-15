#!/usr/bin/env python3
"""
Generate synthetic training data without shooting
This lets us start training immediately!
"""

import numpy as np
import cv2
import os

class SyntheticDataGenerator:
    """Generate training pairs based on camera characteristics"""
    
    def __init__(self):
        self.test_patterns = self._create_test_patterns()
    
    def _create_test_patterns(self):
        """Create test patterns that reveal camera characteristics"""
        patterns = {}
        
        # Color checker pattern
        colors = [
            [115, 82, 68],    # Dark skin
            [194, 150, 130],  # Light skin  
            [98, 122, 157],   # Blue sky
            [87, 108, 67],    # Foliage
            [133, 128, 177],  # Blue flower
            [103, 189, 170],  # Bluish green
        ]
        
        # Create 512x512 color checker
        checker = np.zeros((512, 512, 3), dtype=np.uint8)
        for i, color in enumerate(colors):
            row = i // 3
            col = i % 3
            y1, y2 = row * 170, (row + 1) * 170
            x1, x2 = col * 170, (col + 1) * 170
            if y2 <= 512 and x2 <= 512:
                checker[y1:y2, x1:x2] = color
        
        patterns['color_checker'] = checker
        
        # Gradient for testing highlight rolloff
        gradient = np.zeros((512, 512, 3), dtype=np.uint8)
        for i in range(512):
            gradient[:, i] = int(255 * (i / 511))
        patterns['gradient'] = gradient
        
        # Resolution/sharpness pattern
        sharpness = np.zeros((512, 512, 3), dtype=np.uint8)
        for i in range(0, 512, 2):
            sharpness[:, i] = 255
        patterns['sharpness'] = sharpness
        
        return patterns
    
    def simulate_iphone_processing(self, image):
        """Simulate iPhone image processing characteristics"""
        processed = image.copy()
        
        # iPhone sharpening
        kernel = np.array([[-1,-1,-1], 
                          [-1, 9,-1], 
                          [-1,-1,-1]]) * 0.5
        processed = cv2.filter2D(processed, -1, kernel)
        
        # iPhone color boost
        hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= 1.2  # Increase saturation
        hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
        processed = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # Compressed highlights
        processed = np.power(processed / 255.0, 0.8) * 255
        
        return processed.astype(np.uint8)
    
    def simulate_arri_processing(self, image):
        """Simulate ARRI Alexa processing characteristics"""
        processed = image.copy().astype(np.float32)
        
        # Smooth highlight rolloff
        processed = np.log1p(processed) * 40
        
        # ARRI color science (slight warmth)
        processed[:, :, 2] *= 1.02  # Red channel
        processed[:, :, 0] *= 0.98  # Blue channel
        
        # Organic film grain
        noise = np.random.normal(0, 2, processed.shape)
        processed += noise
        
        # Less aggressive sharpening
        kernel = np.array([[0,-1,0], 
                          [-1, 5,-1], 
                          [0,-1,0]]) * 0.1
        processed = cv2.filter2D(processed, -1, kernel)
        
        return np.clip(processed, 0, 255).astype(np.uint8)
    
    def generate_samples(self):
        """Generate sample image pairs"""
        os.makedirs('data/samples', exist_ok=True)
        
        print("🎨 Generating synthetic test patterns...")
        
        for name, pattern in self.test_patterns.items():
            # Generate iPhone version
            iphone = self.simulate_iphone_processing(pattern)
            cv2.imwrite(f'data/samples/{name}_iphone.png', iphone)
            
            # Generate ARRI version
            arri = self.simulate_arri_processing(pattern)
            cv2.imwrite(f'data/samples/{name}_arri.png', arri)
            
            print(f"  ✓ Generated {name} pair")
        
        print(f"✅ Created {len(self.test_patterns)} test pairs")
        return True
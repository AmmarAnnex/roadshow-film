#!/usr/bin/env python3
"""
Roadshow Sample Footage Downloader & Organizer
Download test samples and prepare them for 3D LUT testing
"""

import os
import requests
import cv2
import numpy as np
from pathlib import Path

class SampleManager:
    """Download and organize test footage for Roadshow testing"""
    
    def __init__(self):
        self.base_dir = Path("data/samples")
        self.setup_directories()
        
    def setup_directories(self):
        """Create organized directory structure"""
        directories = [
            "data/samples/iphone",
            "data/samples/cinema", 
            "data/samples/red_raw",
            "data/samples/extracted_frames",
            "data/results/transformations"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
        print("📁 Created sample directory structure")
    
    def download_red_samples(self):
        """Download RED camera sample files"""
        print("🎥 Downloading RED camera samples...")
        
        # RED sample URLs (these are the actual download links)
        red_samples = {
            "epic_5k_sample.r3d": "https://www.red.com/assets/r3d/epic_5k_sample.r3d",
            "weapon_8k_sample.r3d": "https://www.red.com/assets/r3d/weapon_8k_sample.r3d"
        }
        
        for filename, url in red_samples.items():
            try:
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    filepath = self.base_dir / "red_raw" / filename
                    with open(filepath, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"  ✓ Downloaded {filename}")
                else:
                    print(f"  ❌ Could not download {filename}")
            except Exception as e:
                print(f"  ⚠️ {filename}: {e}")
    
    def create_test_samples(self):
        """Create synthetic test samples if downloads fail"""
        print("🎨 Creating synthetic test samples...")
        
        # iPhone-style sample (more saturated, sharp)
        iphone_sample = self.create_iphone_style_image()
        cv2.imwrite(str(self.base_dir / "iphone" / "iphone_test.jpg"), iphone_sample)
        
        # Cinema-style sample (more organic, film-like)
        cinema_sample = self.create_cinema_style_image()
        cv2.imwrite(str(self.base_dir / "cinema" / "cinema_test.jpg"), cinema_sample)
        
        print("  ✓ Created synthetic samples")
    
    def create_iphone_style_image(self):
        """Create iPhone-style test image"""
        # Base landscape image
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        # Sky gradient (blue to white)
        for y in range(400):
            color = int(255 * (y / 400))
            img[y, :] = [255, color + 100, 180]  # iPhone blue tint
        
        # Ground (green)
        img[400:, :] = [60, 180, 60]  # Saturated green
        
        # Add some "buildings" 
        for i in range(5):
            x_start = i * 300 + 100
            x_end = x_start + 200
            y_start = 300 - i * 40
            img[y_start:400, x_start:x_end] = [40, 40, 40]  # Dark buildings
        
        # iPhone processing characteristics
        img = self.apply_iphone_processing(img)
        
        return img
    
    def create_cinema_style_image(self):
        """Create cinema-style test image"""
        # Same base as iPhone but with cinema characteristics
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        # Warmer, more organic sky
        for y in range(400):
            color = int(255 * (y / 400))
            img[y, :] = [color + 80, color + 120, 255]  # Warmer tones
        
        # More natural ground color
        img[400:, :] = [45, 120, 45]  # Less saturated
        
        # Buildings with more organic shadows
        for i in range(5):
            x_start = i * 300 + 100
            x_end = x_start + 200
            y_start = 300 - i * 40
            img[y_start:400, x_start:x_end] = [30, 35, 40]  # Slightly blue shadows
        
        # Cinema processing
        img = self.apply_cinema_processing(img)
        
        return img
    
    def apply_iphone_processing(self, img):
        """Apply iPhone-style processing"""
        img = img.astype(np.float32)
        
        # iPhone characteristics
        # 1. Higher saturation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] *= 1.3  # Boost saturation
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # 2. Sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img = cv2.filter2D(img, -1, kernel * 0.3)
        
        # 3. Contrast boost
        img = np.clip(img * 1.1, 0, 255)
        
        return img.astype(np.uint8)
    
    def apply_cinema_processing(self, img):
        """Apply cinema-style processing"""
        img = img.astype(np.float32)
        
        # Cinema characteristics
        # 1. Film grain
        noise = np.random.normal(0, 2, img.shape)
        img += noise
        
        # 2. Softer contrast curve
        img = np.power(img / 255.0, 1.1) * 255
        
        # 3. Slight warmth
        img[:, :, 2] *= 1.05  # Slight red push
        img[:, :, 0] *= 0.98  # Slight blue reduction
        
        return np.clip(img, 0, 255).astype(np.uint8)
    
    def extract_video_frames(self, video_path, output_dir, max_frames=10):
        """Extract frames from video files for testing"""
        print(f"🎞️ Extracting frames from {video_path}...")
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        extracted = 0
        
        while extracted < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Extract every 30th frame
            if frame_count % 30 == 0:
                output_path = f"{output_dir}/frame_{extracted:03d}.jpg"
                cv2.imwrite(output_path, frame)
                extracted += 1
                
            frame_count += 1
        
        cap.release()
        print(f"  ✓ Extracted {extracted} frames")
        return extracted
    
    def test_on_samples(self):
        """Run Roadshow 3D LUT on all available samples"""
        print("🚀 Testing Roadshow 3D LUT on samples...")
        
        # Import your engine
        try:
            import sys
            sys.path.append('../backend')
            from roadshow_3dlut_engine import Reality3DLUT, CAMERA_PROFILES
            
            # Initialize engine
            lut_engine = Reality3DLUT(resolution=32)
            lut_engine.create_base_lut()
            
            # Set up iPhone → ARRI transformation
            iphone = CAMERA_PROFILES['iphone_12_pro']
            arri = CAMERA_PROFILES['arri_alexa']
            lut_engine.learn_from_reference(iphone, arri)
            
            # Test on all sample images
            sample_dirs = ["iphone", "cinema", "extracted_frames"]
            
            for sample_dir in sample_dirs:
                input_dir = self.base_dir / sample_dir
                if not input_dir.exists():
                    continue
                    
                print(f"\n📸 Processing {sample_dir} samples...")
                
                for img_file in input_dir.glob("*.jpg"):
                    # Load image
                    image = cv2.imread(str(img_file))
                    if image is None:
                        continue
                    
                    # Apply transformation
                    result_vectorized = lut_engine.apply_to_image(image, method='vectorized')
                    result_trilinear = lut_engine.apply_to_image(image, method='trilinear')
                    
                    # Create comparison
                    comparison = np.hstack([image, result_vectorized, result_trilinear])
                    
                    # Save result
                    output_path = self.base_dir.parent / "results" / "transformations" / f"{img_file.stem}_transform.jpg"
                    cv2.imwrite(str(output_path), comparison)
                    
                    print(f"  ✓ Processed {img_file.name}")
            
            print(f"\n✅ All samples processed!")
            print(f"📁 Check data/results/transformations/ for before/after comparisons")
            
        except ImportError as e:
            print(f"❌ Could not import Roadshow engine: {e}")
            print("Make sure you're running from the main Roadshow3DLUT directory")
    
    def run_full_setup(self):
        """Run complete sample setup and testing"""
        print("🎬 ROADSHOW SAMPLE MANAGER")
        print("=" * 50)
        
        # Try to download real samples
        self.download_red_samples()
        
        # Create synthetic samples for immediate testing
        self.create_test_samples()
        
        # Test the engine on samples
        self.test_on_samples()
        
        print("\n✅ Sample setup complete!")
        print("🎯 Next steps:")
        print("  1. Check data/results/transformations/ for test results")
        print("  2. Bring real footage from your studio this afternoon")
        print("  3. Iterate on the transformations based on results")


if __name__ == "__main__":
    manager = SampleManager()
    manager.run_full_setup()
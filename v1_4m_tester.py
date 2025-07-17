#!/usr/bin/env python3
"""
Cinema v1.4m - 4K Visual Tester
NO BULLSHIT - Just transformations and comparisons
"""

import cv2
import numpy as np
from pathlib import Path
import torch
import rawpy
from cinema_v1_4m import ExposureFixedColorTransform

class CinemaV14mTester:
    """4K visual tester for v1.4m model"""
    
    def __init__(self, model_path="models/cinema_v1_4m_model.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.results_dir = Path("data/results/v1_4m_4k_test")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🎬 Cinema v1.4m 4K Tester Ready")
        print(f"🔧 Device: {self.device}")
        print(f"📁 Output: {self.results_dir}")
    
    def load_model(self, model_path):
        """Load v1.4m model"""
        model = ExposureFixedColorTransform().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✅ Model loaded: {model_path}")
        return model
    
    def load_proper_iphone_image(self, dng_path, size=None):
        """Load iPhone DNG with PROPER exposure (like your JPG)"""
        try:
            with rawpy.imread(str(dng_path)) as raw:
                # PROPER iPhone processing - not dark bullshit
                rgb = raw.postprocess(
                    demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                    half_size=False,
                    use_camera_wb=True,
                    output_color=rawpy.ColorSpace.sRGB,
                    output_bps=16,
                    bright=1.0,
                    no_auto_bright=False,  # FIXED: Allow proper exposure
                    gamma=(2.222, 4.5),    # FIXED: Proper sRGB gamma, not linear
                    user_flip=0
                )
            
            # Convert to float
            rgb_float = rgb.astype(np.float32) / 65535.0
            
            # Resize if requested
            if size:
                h, w = rgb_float.shape[:2]
                if max(h, w) != size:
                    scale = size / max(h, w)
                    new_h, new_w = int(h * scale), int(w * scale)
                    rgb_float = cv2.resize(rgb_float, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            return rgb_float
            
        except Exception as e:
            print(f"❌ Error loading {dng_path}: {e}")
            return None
    
    def load_proper_sony_image(self, arw_path, size=None):
        """Load Sony ARW with proper processing"""
        try:
            with rawpy.imread(str(arw_path)) as raw:
                rgb = raw.postprocess(
                    demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                    half_size=False,
                    use_camera_wb=True,
                    output_color=rawpy.ColorSpace.sRGB,
                    output_bps=16,
                    bright=1.0,
                    no_auto_bright=False,
                    gamma=(2.222, 4.5),
                    user_flip=0
                )
            
            rgb_float = rgb.astype(np.float32) / 65535.0
            
            if size:
                h, w = rgb_float.shape[:2]
                if max(h, w) != size:
                    scale = size / max(h, w)
                    new_h, new_w = int(h * scale), int(w * scale)
                    rgb_float = cv2.resize(rgb_float, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            return rgb_float
            
        except Exception as e:
            print(f"❌ Error loading {arw_path}: {e}")
            return None
    
    def transform_image(self, iphone_img):
        """Transform iPhone image using v1.4m model"""
        # Prepare for model (768x768)
        h, w = iphone_img.shape[:2]
        img_768 = cv2.resize(iphone_img, (768, 768), interpolation=cv2.INTER_LANCZOS4)
        
        # Convert to tensor
        tensor = torch.from_numpy(img_768).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Transform
        with torch.no_grad():
            transformed_tensor = self.model(tensor)
        
        # Convert back
        transformed_768 = transformed_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        transformed_768 = np.clip(transformed_768, 0, 1)
        
        # Resize back to original size
        transformed = cv2.resize(transformed_768, (w, h), interpolation=cv2.INTER_LANCZOS4)
        
        return transformed
    
    def create_4k_comparison(self, pair_id, save_size=4096):
        """Create 4K comparison image"""
        print(f"\n🎯 Processing Pair {pair_id:03d} at {save_size}px")
        
        # File paths
        training_pairs_dir = Path("data/training_pairs")
        iphone_file = training_pairs_dir / f"iphone_{pair_id:03d}.dng"
        sony_file = training_pairs_dir / f"sony_{pair_id:03d}.arw"
        
        if not (iphone_file.exists() and sony_file.exists()):
            print(f"❌ Pair {pair_id:03d} not found")
            return False
        
        # Load images at high resolution
        print("   📷 Loading iPhone image...")
        iphone_img = self.load_proper_iphone_image(iphone_file, size=save_size)
        
        print("   📷 Loading Sony target...")
        sony_img = self.load_proper_sony_image(sony_file, size=save_size)
        
        if iphone_img is None or sony_img is None:
            print(f"❌ Failed to load pair {pair_id:03d}")
            return False
        
        # Transform iPhone image
        print("   🎬 Applying cinema transformation...")
        transformed = self.transform_image(iphone_img)
        
        # Convert to display format (0-255)
        iphone_display = (np.clip(iphone_img, 0, 1) * 255).astype(np.uint8)
        transformed_display = (np.clip(transformed, 0, 1) * 255).astype(np.uint8)
        sony_display = (np.clip(sony_img, 0, 1) * 255).astype(np.uint8)
        
        # Create side-by-side comparison
        h, w = iphone_display.shape[:2]
        
        # Three-way comparison
        comparison = np.hstack([
            cv2.cvtColor(iphone_display, cv2.COLOR_RGB2BGR),
            cv2.cvtColor(transformed_display, cv2.COLOR_RGB2BGR),
            cv2.cvtColor(sony_display, cv2.COLOR_RGB2BGR)
        ])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(2.0, save_size / 2000)
        thickness = max(4, int(save_size / 1000))
        
        # iPhone Original
        cv2.putText(comparison, "iPhone Original", 
                   (50, 100), font, font_scale, (0, 255, 0), thickness)
        
        # Cinema v1.4m
        cv2.putText(comparison, "Cinema v1.4m", 
                   (w + 50, 100), font, font_scale, (0, 100, 255), thickness)
        
        # Sony A7S3 Target
        cv2.putText(comparison, "Sony A7S3 Target", 
                   (2*w + 50, 100), font, font_scale, (0, 200, 0), thickness)
        
        # Save comparison
        comparison_path = self.results_dir / f"4k_comparison_{pair_id:03d}_{save_size}px.jpg"
        cv2.imwrite(str(comparison_path), comparison, [cv2.IMWRITE_JPEG_QUALITY, 98])
        
        # Save individual images
        iphone_path = self.results_dir / f"iphone_original_{pair_id:03d}_{save_size}px.jpg"
        transformed_path = self.results_dir / f"cinema_v1_4m_{pair_id:03d}_{save_size}px.jpg"
        sony_path = self.results_dir / f"sony_target_{pair_id:03d}_{save_size}px.jpg"
        
        cv2.imwrite(str(iphone_path), cv2.cvtColor(iphone_display, cv2.COLOR_RGB2BGR), 
                   [cv2.IMWRITE_JPEG_QUALITY, 98])
        cv2.imwrite(str(transformed_path), cv2.cvtColor(transformed_display, cv2.COLOR_RGB2BGR), 
                   [cv2.IMWRITE_JPEG_QUALITY, 98])
        cv2.imwrite(str(sony_path), cv2.cvtColor(sony_display, cv2.COLOR_RGB2BGR), 
                   [cv2.IMWRITE_JPEG_QUALITY, 98])
        
        print(f"   ✅ Saved: {comparison_path}")
        print(f"   ✅ Individual images saved")
        
        return True
    
    def batch_test(self, start_pair=1, num_pairs=5, size=4096):
        """Test multiple pairs"""
        print(f"\n🚀 BATCH TESTING - {num_pairs} pairs at {size}px")
        print("=" * 60)
        
        success_count = 0
        
        for i in range(start_pair, start_pair + num_pairs):
            if self.create_4k_comparison(i, save_size=size):
                success_count += 1
        
        print(f"\n✅ BATCH COMPLETE: {success_count}/{num_pairs} pairs processed")
        print(f"📁 Results in: {self.results_dir}")


def main():
    """Interactive testing interface"""
    print("🎬 CINEMA v1.4m - 4K VISUAL TESTER")
    print("=" * 50)
    print("NO BULLSHIT - Just pure transformations")
    
    tester = CinemaV14mTester()
    
    while True:
        print("\n🎯 OPTIONS:")
        print("1. Single pair test")
        print("2. Batch test (multiple pairs)")
        print("3. Custom resolution test")
        print("4. Exit")
        
        try:
            choice = input("\nChoice (1-4): ").strip()
            
            if choice == "1":
                pair_input = input("Pair ID (1-79): ").strip()
                pair_id = int(pair_input)
                
                size_input = input("Resolution (default 4096): ").strip()
                size = int(size_input) if size_input else 4096
                
                success = tester.create_4k_comparison(pair_id, save_size=size)
                if success:
                    print(f"✅ SUCCESS: Pair {pair_id:03d} processed at {size}px")
                else:
                    print(f"❌ FAILED: Could not process pair {pair_id:03d}")
                
            elif choice == "2":
                start_input = input("Start pair (default 1): ").strip()
                start = int(start_input) if start_input else 1
                
                count_input = input("Number of pairs (default 5): ").strip()
                count = int(count_input) if count_input else 5
                
                size_input = input("Resolution (default 4096): ").strip()
                size = int(size_input) if size_input else 4096
                
                tester.batch_test(start_pair=start, num_pairs=count, size=size)
                
            elif choice == "3":
                pair_input = input("Pair ID: ").strip()
                pair_id = int(pair_input)
                
                size_input = input("Custom resolution: ").strip()
                size = int(size_input)
                
                success = tester.create_4k_comparison(pair_id, save_size=size)
                if success:
                    print(f"✅ SUCCESS: Pair {pair_id:03d} processed at {size}px")
                else:
                    print(f"❌ FAILED: Could not process pair {pair_id:03d}")
                
            elif choice == "4":
                print("👋 Done!")
                break
                
            else:
                print("❌ Invalid choice - enter 1, 2, 3, or 4")
                
        except KeyboardInterrupt:
            print("\n👋 Interrupted!")
            break
        except ValueError as e:
            print(f"❌ Invalid input - please enter a valid number")
            print(f"Error details: {e}")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
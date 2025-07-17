#!/usr/bin/env python3
"""
Cinema v1.4m - 4K Visual Tester - WORKING VERSION
Uses the v1.4m model with proper aspect ratio handling
"""

import cv2
import numpy as np
from pathlib import Path
import torch
import rawpy
import traceback

# Import the correct v1.4m model
from cinema_v1_4m import ExposureFixedColorTransform

class CinemaV14mTester:
    """4K visual tester for v1.4m model"""
    
    def __init__(self, model_path="models/cinema_v1_4m_model.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.results_dir = Path("data/results/v1_4m_4k_test")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🎬 Cinema v1.4m 4K Tester Ready (Exposure Fixed)")
        print(f"🔧 Device: {self.device}")
        print(f"📁 Output: {self.results_dir}")
    
    def load_model(self, model_path):
        """Load the v1.4m model"""
        model_path = Path(model_path)
        
        # Check multiple possible model locations for v1.4m
        possible_paths = [
            model_path,
            Path("models/cinema_v1_4m_model.pth"),
            Path("data/cinema_v1_4m_model.pth")
        ]
        
        # Find first existing model
        actual_model_path = None
        for path in possible_paths:
            if path.exists():
                actual_model_path = path
                break
        
        if actual_model_path is None:
            print("❌ No v1.4m model found in any of these locations:")
            for path in possible_paths:
                print(f"   - {path}")
            print("\n💡 Train v1.4m model first:")
            print("   python cinema_v1_4m.py")
            raise FileNotFoundError("No v1.4m model found")
        
        # Load v1.4m model
        try:
            model = ExposureFixedColorTransform().to(self.device)
            checkpoint = torch.load(actual_model_path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            print(f"✅ Loaded v1.4m model: {actual_model_path}")
            print(f"   Training pairs: {checkpoint.get('training_pairs', 'Unknown')}")
            print(f"   Best loss: {checkpoint.get('loss', 'Unknown'):.4f}")
            
            # Print exposure parameters
            with torch.no_grad():
                print(f"   Exposure factor: {model.global_exposure.item():.3f}")
                print(f"   Shadow lift: {model.shadows.item():.3f}")
                print(f"   Residual strength: {model.residual_strength.item():.3f}")
                
        except Exception as e:
            print(f"❌ Error loading v1.4m model: {e}")
            traceback.print_exc()
            raise
        
        return model
    
    def load_proper_iphone_image(self, dng_path, size=None):
        """Load iPhone DNG with proper exposure and aspect ratio handling"""
        try:
            with rawpy.imread(str(dng_path)) as raw:
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
            print(f"     iPhone original size: {rgb_float.shape[:2]}")
            
            # Smart crop to square (center crop)
            h, w = rgb_float.shape[:2]
            crop_size = min(h, w)
            
            start_y = (h - crop_size) // 2
            start_x = (w - crop_size) // 2
            rgb_square = rgb_float[start_y:start_y + crop_size, start_x:start_x + crop_size]
            
            print(f"     iPhone cropped to square: {rgb_square.shape[:2]}")
            
            # Resize to requested size
            if size and size != crop_size:
                rgb_resized = cv2.resize(rgb_square, (size, size), interpolation=cv2.INTER_LANCZOS4)
                print(f"     iPhone resized to: {rgb_resized.shape[:2]}")
                return rgb_resized
            
            return rgb_square
            
        except Exception as e:
            print(f"❌ Error loading iPhone image {dng_path}: {e}")
            traceback.print_exc()
            return None
    
    def load_proper_sony_image(self, arw_path, size=None):
        """Load Sony ARW with proper processing and aspect ratio handling"""
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
            print(f"     Sony original size: {rgb_float.shape[:2]}")
            
            # Smart crop to square (center crop)
            h, w = rgb_float.shape[:2]
            crop_size = min(h, w)
            
            start_y = (h - crop_size) // 2
            start_x = (w - crop_size) // 2
            rgb_square = rgb_float[start_y:start_y + crop_size, start_x:start_x + crop_size]
            
            print(f"     Sony cropped to square: {rgb_square.shape[:2]}")
            
            # Resize to requested size
            if size and size != crop_size:
                rgb_resized = cv2.resize(rgb_square, (size, size), interpolation=cv2.INTER_LANCZOS4)
                print(f"     Sony resized to: {rgb_resized.shape[:2]}")
                return rgb_resized
            
            return rgb_square
            
        except Exception as e:
            print(f"❌ Error loading Sony image {arw_path}: {e}")
            traceback.print_exc()
            return None
    
    def transform_image(self, iphone_img):
        """Transform iPhone image using v1.4m model"""
        try:
            # Prepare for model (768x768 - same as training)
            h, w = iphone_img.shape[:2]
            img_768 = cv2.resize(iphone_img, (768, 768), interpolation=cv2.INTER_LANCZOS4)
            
            # Convert to tensor
            tensor = torch.from_numpy(img_768).permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            # Transform with v1.4m
            with torch.no_grad():
                transformed_tensor = self.model(tensor)
            
            # Convert back
            transformed_768 = transformed_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            transformed_768 = np.clip(transformed_768, 0, 1)
            
            # Resize back to original size
            transformed = cv2.resize(transformed_768, (w, h), interpolation=cv2.INTER_LANCZOS4)
            
            return transformed
            
        except Exception as e:
            print(f"❌ Error in transform_image: {e}")
            traceback.print_exc()
            return None
    
    def create_4k_comparison(self, pair_id, save_size=4096):
        """Create 4K comparison image with proper error handling"""
        print(f"\n🎯 Processing Pair {pair_id:03d} at {save_size}px")
        
        try:
            # File paths
            training_pairs_dir = Path("data/training_pairs")
            iphone_file = training_pairs_dir / f"iphone_{pair_id:03d}.dng"
            sony_file = training_pairs_dir / f"sony_{pair_id:03d}.arw"
            
            if not (iphone_file.exists() and sony_file.exists()):
                print(f"❌ Pair {pair_id:03d} files not found:")
                print(f"   iPhone: {iphone_file} - {'✓' if iphone_file.exists() else '✗'}")
                print(f"   Sony: {sony_file} - {'✓' if sony_file.exists() else '✗'}")
                return False
            
            # Load images at save_size (both will be square after cropping)
            print("   📷 Loading iPhone image...")
            iphone_img = self.load_proper_iphone_image(iphone_file, size=save_size)
            
            print("   📷 Loading Sony target...")
            sony_img = self.load_proper_sony_image(sony_file, size=save_size)
            
            if iphone_img is None:
                print(f"❌ Failed to load iPhone image: {iphone_file}")
                return False
                
            if sony_img is None:
                print(f"❌ Failed to load Sony image: {sony_file}")
                return False
            
            # Verify they're the same size and square
            if iphone_img.shape != sony_img.shape:
                print(f"❌ Size mismatch after processing:")
                print(f"   iPhone: {iphone_img.shape}")
                print(f"   Sony: {sony_img.shape}")
                return False
            
            print(f"   ✅ Both images processed to: {iphone_img.shape}")
            
            # Transform iPhone image
            print("   🎬 Applying v1.4m cinema transformation...")
            transformed = self.transform_image(iphone_img)
            
            if transformed is None:
                print(f"❌ Failed to transform image")
                return False
            
            print(f"   ✅ Transform complete: {transformed.shape}")
            
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
            
            print(f"   ✅ Comparison created: {comparison.shape}")
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(2.0, save_size / 2000)
            thickness = max(4, int(save_size / 1000))
            
            # iPhone Original
            cv2.putText(comparison, "iPhone Original", 
                       (50, 100), font, font_scale, (0, 255, 0), thickness)
            
            # Cinema v1.4m (Exposure Fixed)
            cv2.putText(comparison, "Cinema v1.4m (Exposure Fixed)", 
                       (w + 50, 100), font, font_scale, (0, 100, 255), thickness)
            
            # Sony A7S3 Target
            cv2.putText(comparison, "Sony A7S3 Target", 
                       (2*w + 50, 100), font, font_scale, (0, 200, 0), thickness)
            
            # Save comparison
            comparison_path = self.results_dir / f"4k_comparison_{pair_id:03d}_{save_size}px.jpg"
            success = cv2.imwrite(str(comparison_path), comparison, [cv2.IMWRITE_JPEG_QUALITY, 98])
            
            if not success:
                print(f"❌ Failed to save comparison image")
                return False
            
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
            
        except Exception as e:
            print(f"❌ Error processing pair {pair_id:03d}: {e}")
            traceback.print_exc()
            return False
    
    def batch_test(self, start_pair=1, num_pairs=5, size=4096):
        """Test multiple pairs with proper error handling"""
        print(f"\n🚀 BATCH TESTING v1.4m - {num_pairs} pairs at {size}px")
        print("=" * 60)
        
        success_count = 0
        
        for i in range(start_pair, start_pair + num_pairs):
            if self.create_4k_comparison(i, save_size=size):
                success_count += 1
        
        print(f"\n✅ BATCH COMPLETE: {success_count}/{num_pairs} pairs processed")
        print(f"📁 Results in: {self.results_dir}")


def main():
    """Interactive testing interface for v1.4m"""
    print("🎬 CINEMA v1.4m - 4K VISUAL TESTER (EXPOSURE FIXED)")
    print("=" * 50)
    print("Testing the exposure-corrected v1.4m model")
    
    try:
        tester = CinemaV14mTester()
    except Exception as e:
        print(f"❌ Failed to initialize tester: {e}")
        return
    
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
                if not pair_input:
                    print("❌ Please enter a pair ID")
                    continue
                    
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
                if not pair_input:
                    print("❌ Please enter a pair ID")
                    continue
                    
                pair_id = int(pair_input)
                
                size_input = input("Custom resolution: ").strip()
                if not size_input:
                    print("❌ Please enter a resolution")
                    continue
                    
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
        except Exception as e:
            print(f"❌ Error: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
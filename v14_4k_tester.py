#!/usr/bin/env python3
"""
Cinema v1.4 4K - High Resolution Visual Tester
Tests the stable StableColorTransform4K model at full resolution
"""

import cv2
import numpy as np
from pathlib import Path
import torch
import rawpy
import traceback

# Import the v1.4 4K model
from cinema_v14_4k import StableColorTransform4K

class CinemaV14_4KTester:
    """High resolution tester for v1.4 4K model"""
    
    def __init__(self, model_path="data/cinema_v14_4k_model.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.results_dir = Path("data/results/v14_4k_high_res_test")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🎬 Cinema v1.4 4K High Resolution Tester Ready")
        print(f"🔧 Device: {self.device}")
        print(f"📁 Output: {self.results_dir}")
    
    def load_model(self, model_path):
        """Load the v1.4 4K model"""
        model_path = Path(model_path)
        
        # Check multiple possible model locations
        possible_paths = [
            model_path,
            Path("data/cinema_v14_4k_model.pth"),
            Path("models/cinema_v14_4k_model.pth")
        ]
        
        # Find first existing model
        actual_model_path = None
        for path in possible_paths:
            if path.exists():
                actual_model_path = path
                break
        
        if actual_model_path is None:
            print("❌ No v1.4 4K model found in any of these locations:")
            for path in possible_paths:
                print(f"   - {path}")
            print("\n💡 Train v1.4 4K model first:")
            print("   python cinema_v14_4k.py")
            raise FileNotFoundError("No v1.4 4K model found")
        
        # Load v1.4 4K model
        try:
            model = StableColorTransform4K().to(self.device)
            checkpoint = torch.load(actual_model_path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            print(f"✅ Loaded v1.4 4K model: {actual_model_path}")
            print(f"   Training epoch: {checkpoint.get('epoch', 'Unknown')}")
            print(f"   Training loss: {checkpoint.get('loss', 'Unknown'):.6f}")
            print(f"   Training size: {checkpoint.get('target_size', 'Unknown')}x{checkpoint.get('target_size', 'Unknown')}")
            
            # Print model parameters
            with torch.no_grad():
                print(f"   Channel adjust: R={model.channel_adjust[0]:.4f}, G={model.channel_adjust[1]:.4f}, B={model.channel_adjust[2]:.4f}")
                print(f"   Shadows: {model.shadows.item():.4f}")
                print(f"   Residual strength: {model.residual_strength.item():.4f}")
                
        except Exception as e:
            print(f"❌ Error loading v1.4 4K model: {e}")
            traceback.print_exc()
            raise
        
        return model
    
    def load_high_res_image(self, file_path, max_size=None):
        """Load image at maximum resolution with smart cropping"""
        try:
            with rawpy.imread(str(file_path)) as raw:
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
            print(f"     Original size: {rgb_float.shape[:2]}")
            
            # Smart crop to square (center crop for best composition)
            h, w = rgb_float.shape[:2]
            crop_size = min(h, w)
            
            start_y = (h - crop_size) // 2
            start_x = (w - crop_size) // 2
            rgb_square = rgb_float[start_y:start_y + crop_size, start_x:start_x + crop_size]
            
            print(f"     Cropped to square: {rgb_square.shape[:2]}")
            
            # Optionally resize to max_size while maintaining quality
            if max_size and crop_size > max_size:
                rgb_resized = cv2.resize(rgb_square, (max_size, max_size), interpolation=cv2.INTER_LANCZOS4)
                print(f"     Resized to: {rgb_resized.shape[:2]}")
                return rgb_resized
            
            return rgb_square
            
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            traceback.print_exc()
            return None
    
    def transform_high_res_image(self, image):
        """Transform image using v1.4 4K model - handles any resolution"""
        try:
            # The v1.4 4K model was trained at 1024x1024 but can handle any resolution
            # For consistency, we'll process at training resolution
            h, w = image.shape[:2]
            training_size = 1024
            
            # Resize to training resolution for model processing
            img_1024 = cv2.resize(image, (training_size, training_size), interpolation=cv2.INTER_LANCZOS4)
            
            # Convert to tensor
            tensor = torch.from_numpy(img_1024).permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            # Transform with v1.4 4K model
            with torch.no_grad():
                transformed_tensor = self.model(tensor)
            
            # Convert back
            transformed_1024 = transformed_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            transformed_1024 = np.clip(transformed_1024, 0, 1)
            
            # Resize back to original resolution
            transformed = cv2.resize(transformed_1024, (w, h), interpolation=cv2.INTER_LANCZOS4)
            
            return transformed
            
        except Exception as e:
            print(f"❌ Error in transform_high_res_image: {e}")
            traceback.print_exc()
            return None
    
    def create_high_res_comparison(self, pair_id, max_size=4096):
        """Create high resolution comparison at maximum quality"""
        print(f"\n🎯 Processing Pair {pair_id:03d} at max {max_size}px")
        
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
            
            # Load images at maximum resolution
            print("   📷 Loading iPhone image at max resolution...")
            iphone_img = self.load_high_res_image(iphone_file, max_size=max_size)
            
            print("   📷 Loading Sony target at max resolution...")
            sony_img = self.load_high_res_image(sony_file, max_size=max_size)
            
            if iphone_img is None:
                print(f"❌ Failed to load iPhone image: {iphone_file}")
                return False
                
            if sony_img is None:
                print(f"❌ Failed to load Sony image: {sony_file}")
                return False
            
            # Ensure both images are the same size
            if iphone_img.shape != sony_img.shape:
                print(f"❌ Size mismatch:")
                print(f"   iPhone: {iphone_img.shape}")
                print(f"   Sony: {sony_img.shape}")
                return False
            
            final_size = iphone_img.shape[:2]
            print(f"   ✅ Both images: {final_size}")
            
            # Transform iPhone image
            print("   🎬 Applying v1.4 4K cinema transformation...")
            transformed = self.transform_high_res_image(iphone_img)
            
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
            
            # Add labels with size-appropriate font
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(1.5, h / 1500)
            thickness = max(3, int(h / 800))
            
            # iPhone Original
            cv2.putText(comparison, "iPhone Original", 
                       (30, 80), font, font_scale, (0, 255, 0), thickness)
            
            # Cinema v1.4 4K
            cv2.putText(comparison, "Cinema v1.4 4K (Stable)", 
                       (w + 30, 80), font, font_scale, (0, 100, 255), thickness)
            
            # Sony A7S3 Target
            cv2.putText(comparison, "Sony A7S3 Target", 
                       (2*w + 30, 80), font, font_scale, (0, 200, 0), thickness)
            
            # Save comparison at maximum quality
            comparison_path = self.results_dir / f"4k_comparison_{pair_id:03d}_{h}x{w}.jpg"
            success = cv2.imwrite(str(comparison_path), comparison, [cv2.IMWRITE_JPEG_QUALITY, 100])
            
            if not success:
                print(f"❌ Failed to save comparison image")
                return False
            
            # Save individual images at maximum quality
            iphone_path = self.results_dir / f"iphone_original_{pair_id:03d}_{h}x{w}.jpg"
            transformed_path = self.results_dir / f"cinema_v14_4k_{pair_id:03d}_{h}x{w}.jpg"
            sony_path = self.results_dir / f"sony_target_{pair_id:03d}_{h}x{w}.jpg"
            
            cv2.imwrite(str(iphone_path), cv2.cvtColor(iphone_display, cv2.COLOR_RGB2BGR), 
                       [cv2.IMWRITE_JPEG_QUALITY, 100])
            cv2.imwrite(str(transformed_path), cv2.cvtColor(transformed_display, cv2.COLOR_RGB2BGR), 
                       [cv2.IMWRITE_JPEG_QUALITY, 100])
            cv2.imwrite(str(sony_path), cv2.cvtColor(sony_display, cv2.COLOR_RGB2BGR), 
                       [cv2.IMWRITE_JPEG_QUALITY, 100])
            
            print(f"   ✅ Saved: {comparison_path}")
            print(f"   ✅ Individual images saved at maximum quality")
            
            # Calculate simple quality metrics
            mse = np.mean((iphone_img - transformed) ** 2)
            brightness_change = np.mean(transformed) - np.mean(iphone_img)
            
            print(f"   📊 MSE: {mse:.6f}")
            print(f"   📊 Brightness change: {brightness_change:+.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error processing pair {pair_id:03d}: {e}")
            traceback.print_exc()
            return False
    
    def batch_test_high_res(self, start_pair=1, num_pairs=5, max_size=4096):
        """Test multiple pairs at high resolution"""
        print(f"\n🚀 BATCH HIGH-RES TESTING v1.4 4K - {num_pairs} pairs at max {max_size}px")
        print("=" * 70)
        
        success_count = 0
        
        for i in range(start_pair, start_pair + num_pairs):
            if self.create_high_res_comparison(i, max_size=max_size):
                success_count += 1
        
        print(f"\n✅ BATCH COMPLETE: {success_count}/{num_pairs} pairs processed")
        print(f"📁 Results in: {self.results_dir}")
        print("🔍 Check the high-resolution comparisons!")


def main():
    """Interactive testing interface for v1.4 4K"""
    print("🎬 CINEMA v1.4 4K - HIGH RESOLUTION VISUAL TESTER")
    print("=" * 50)
    print("Testing the stable v1.4 4K model at maximum resolution")
    
    try:
        tester = CinemaV14_4KTester()
    except Exception as e:
        print(f"❌ Failed to initialize tester: {e}")
        return
    
    while True:
        print("\n🎯 HIGH-RES OPTIONS:")
        print("1. Single pair test (max quality)")
        print("2. Batch test (multiple pairs)")
        print("3. Ultra high-res test (native resolution)")
        print("4. Exit")
        
        try:
            choice = input("\nChoice (1-4): ").strip()
            
            if choice == "1":
                pair_input = input("Pair ID (1-79): ").strip()
                if not pair_input:
                    print("❌ Please enter a pair ID")
                    continue
                    
                pair_id = int(pair_input)
                
                size_input = input("Max resolution (default 4096): ").strip()
                max_size = int(size_input) if size_input else 4096
                
                success = tester.create_high_res_comparison(pair_id, max_size=max_size)
                if success:
                    print(f"✅ SUCCESS: Pair {pair_id:03d} processed at max {max_size}px")
                else:
                    print(f"❌ FAILED: Could not process pair {pair_id:03d}")
                
            elif choice == "2":
                start_input = input("Start pair (default 1): ").strip()
                start = int(start_input) if start_input else 1
                
                count_input = input("Number of pairs (default 5): ").strip()
                count = int(count_input) if count_input else 5
                
                size_input = input("Max resolution (default 4096): ").strip()
                max_size = int(size_input) if size_input else 4096
                
                tester.batch_test_high_res(start_pair=start, num_pairs=count, max_size=max_size)
                
            elif choice == "3":
                pair_input = input("Pair ID: ").strip()
                if not pair_input:
                    print("❌ Please enter a pair ID")
                    continue
                    
                pair_id = int(pair_input)
                
                print("🔥 Ultra high-res mode - using native resolution (no downscaling)")
                success = tester.create_high_res_comparison(pair_id, max_size=None)
                if success:
                    print(f"✅ SUCCESS: Pair {pair_id:03d} processed at native resolution")
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
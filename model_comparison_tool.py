#!/usr/bin/env python3
"""
Cinema Model Comparison Tool - v1.4m vs v1.4 4K
Compare the exposure-fixed v1.4m against the stable v1.4 4K model
"""

import cv2
import numpy as np
from pathlib import Path
import torch
import rawpy
import traceback
import json
from datetime import datetime

# Import both models
from cinema_v1_4m import ExposureFixedColorTransform
from cinema_v14_4k import StableColorTransform4K

class ModelComparator:
    """Compare v1.4m and v1.4 4K models side by side"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_v14m = None
        self.model_v14_4k = None
        self.results_dir = Path("data/results/model_comparison_v14m_vs_v14_4k")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🎬 Cinema Model Comparison Tool")
        print(f"🔧 Device: {self.device}")
        print(f"📁 Output: {self.results_dir}")
        
        self.load_models()
    
    def load_models(self):
        """Load both v1.4m and v1.4 4K models"""
        print("\n📦 Loading Models...")
        
        # Load v1.4m model
        v14m_paths = [
            Path("models/cinema_v1_4m_model.pth"),
            Path("data/cinema_v1_4m_model.pth")
        ]
        
        for path in v14m_paths:
            if path.exists():
                try:
                    self.model_v14m = ExposureFixedColorTransform().to(self.device)
                    checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                    self.model_v14m.load_state_dict(checkpoint['model_state_dict'])
                    self.model_v14m.eval()
                    
                    print(f"✅ Loaded v1.4m model: {path}")
                    print(f"   Loss: {checkpoint.get('loss', 'Unknown'):.4f}")
                    print(f"   Exposure: {self.model_v14m.global_exposure.item():.3f}")
                    break
                except Exception as e:
                    print(f"❌ Error loading v1.4m from {path}: {e}")
        
        # Load v1.4 4K model
        v14_4k_paths = [
            Path("data/cinema_v14_4k_model.pth"),
            Path("models/cinema_v14_4k_model.pth")
        ]
        
        for path in v14_4k_paths:
            if path.exists():
                try:
                    self.model_v14_4k = StableColorTransform4K().to(self.device)
                    checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                    self.model_v14_4k.load_state_dict(checkpoint['model_state_dict'])
                    self.model_v14_4k.eval()
                    
                    print(f"✅ Loaded v1.4 4K model: {path}")
                    print(f"   Loss: {checkpoint.get('loss', 'Unknown'):.6f}")
                    print(f"   Residual: {self.model_v14_4k.residual_strength.item():.4f}")
                    break
                except Exception as e:
                    print(f"❌ Error loading v1.4 4K from {path}: {e}")
        
        if self.model_v14m is None:
            print("⚠️ v1.4m model not loaded - comparisons will be limited")
        if self.model_v14_4k is None:
            print("⚠️ v1.4 4K model not loaded - comparisons will be limited")
    
    def load_and_process_image(self, file_path, target_size=2048):
        """Load and process image for comparison"""
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
            
            # Smart crop to square
            h, w = rgb_float.shape[:2]
            crop_size = min(h, w)
            
            start_y = (h - crop_size) // 2
            start_x = (w - crop_size) // 2
            rgb_square = rgb_float[start_y:start_y + crop_size, start_x:start_x + crop_size]
            
            # Resize to target size
            if crop_size != target_size:
                rgb_resized = cv2.resize(rgb_square, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
                return rgb_resized
            
            return rgb_square
            
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            return None
    
    def transform_v14m(self, image):
        """Transform with v1.4m model"""
        if self.model_v14m is None:
            return None
        
        try:
            # v1.4m expects 768x768
            h, w = image.shape[:2]
            img_768 = cv2.resize(image, (768, 768), interpolation=cv2.INTER_LANCZOS4)
            
            tensor = torch.from_numpy(img_768).permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                transformed_tensor = self.model_v14m(tensor)
            
            transformed_768 = transformed_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            transformed_768 = np.clip(transformed_768, 0, 1)
            
            # Resize back to original size
            transformed = cv2.resize(transformed_768, (w, h), interpolation=cv2.INTER_LANCZOS4)
            
            return transformed
            
        except Exception as e:
            print(f"❌ Error in v1.4m transform: {e}")
            return None
    
    def transform_v14_4k(self, image):
        """Transform with v1.4 4K model"""
        if self.model_v14_4k is None:
            return None
        
        try:
            # v1.4 4K expects 1024x1024
            h, w = image.shape[:2]
            img_1024 = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LANCZOS4)
            
            tensor = torch.from_numpy(img_1024).permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                transformed_tensor = self.model_v14_4k(tensor)
            
            transformed_1024 = transformed_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            transformed_1024 = np.clip(transformed_1024, 0, 1)
            
            # Resize back to original size
            transformed = cv2.resize(transformed_1024, (w, h), interpolation=cv2.INTER_LANCZOS4)
            
            return transformed
            
        except Exception as e:
            print(f"❌ Error in v1.4 4K transform: {e}")
            return None
    
    def create_5_way_comparison(self, pair_id, target_size=2048):
        """Create 5-way comparison: Original | v1.4m | v1.4 4K | Sony | Metrics"""
        print(f"\n🎯 Comparing Models on Pair {pair_id:03d} at {target_size}x{target_size}")
        
        try:
            # File paths
            training_pairs_dir = Path("data/training_pairs")
            iphone_file = training_pairs_dir / f"iphone_{pair_id:03d}.dng"
            sony_file = training_pairs_dir / f"sony_{pair_id:03d}.arw"
            
            if not (iphone_file.exists() and sony_file.exists()):
                print(f"❌ Pair {pair_id:03d} files not found")
                return False
            
            # Load images
            print("   📷 Loading iPhone and Sony images...")
            iphone_img = self.load_and_process_image(iphone_file, target_size)
            sony_img = self.load_and_process_image(sony_file, target_size)
            
            if iphone_img is None or sony_img is None:
                print(f"❌ Failed to load images")
                return False
            
            print(f"   ✅ Images loaded: {iphone_img.shape}")
            
            # Transform with both models
            print("   🎬 Applying v1.4m transformation...")
            transformed_v14m = self.transform_v14m(iphone_img)
            
            print("   🎬 Applying v1.4 4K transformation...")
            transformed_v14_4k = self.transform_v14_4k(iphone_img)
            
            # Calculate metrics
            metrics = self.calculate_comparison_metrics(iphone_img, transformed_v14m, transformed_v14_4k, sony_img)
            
            # Create visual comparison
            comparison_img = self.create_visual_comparison(
                iphone_img, transformed_v14m, transformed_v14_4k, sony_img, metrics, pair_id
            )
            
            if comparison_img is None:
                print(f"❌ Failed to create comparison")
                return False
            
            # Save comparison
            comparison_path = self.results_dir / f"model_comparison_{pair_id:03d}_{target_size}px.jpg"
            success = cv2.imwrite(str(comparison_path), comparison_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            if not success:
                print(f"❌ Failed to save comparison")
                return False
            
            # Save individual transformed images
            if transformed_v14m is not None:
                v14m_path = self.results_dir / f"v14m_result_{pair_id:03d}.jpg"
                cv2.imwrite(str(v14m_path), cv2.cvtColor((np.clip(transformed_v14m, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), 
                           [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            if transformed_v14_4k is not None:
                v14_4k_path = self.results_dir / f"v14_4k_result_{pair_id:03d}.jpg"
                cv2.imwrite(str(v14_4k_path), cv2.cvtColor((np.clip(transformed_v14_4k, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), 
                           [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Save metrics
            metrics_path = self.results_dir / f"metrics_{pair_id:03d}.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            print(f"   ✅ Saved: {comparison_path}")
            print(f"   ✅ Metrics saved: {metrics_path}")
            
            # Print key metrics
            if metrics:
                print(f"   📊 v1.4m brightness change: {metrics.get('v14m_brightness_change', 'N/A'):+.4f}")
                print(f"   📊 v1.4 4K brightness change: {metrics.get('v14_4k_brightness_change', 'N/A'):+.4f}")
                print(f"   📊 v1.4m MSE to target: {metrics.get('v14m_mse_to_target', 'N/A'):.6f}")
                print(f"   📊 v1.4 4K MSE to target: {metrics.get('v14_4k_mse_to_target', 'N/A'):.6f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in comparison: {e}")
            traceback.print_exc()
            return False
    
    def calculate_comparison_metrics(self, original, v14m_result, v14_4k_result, target):
        """Calculate detailed comparison metrics"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'original_brightness': float(np.mean(original)),
            'target_brightness': float(np.mean(target))
        }
        
        if v14m_result is not None:
            metrics.update({
                'v14m_brightness': float(np.mean(v14m_result)),
                'v14m_brightness_change': float(np.mean(v14m_result) - np.mean(original)),
                'v14m_mse_to_target': float(np.mean((v14m_result - target) ** 2)),
                'v14m_mse_to_original': float(np.mean((v14m_result - original) ** 2)),
                'v14m_preserves_brightness': bool(np.mean(v14m_result) >= np.mean(original) * 0.9)
            })
        
        if v14_4k_result is not None:
            metrics.update({
                'v14_4k_brightness': float(np.mean(v14_4k_result)),
                'v14_4k_brightness_change': float(np.mean(v14_4k_result) - np.mean(original)),
                'v14_4k_mse_to_target': float(np.mean((v14_4k_result - target) ** 2)),
                'v14_4k_mse_to_original': float(np.mean((v14_4k_result - original) ** 2)),
                'v14_4k_preserves_brightness': bool(np.mean(v14_4k_result) >= np.mean(original) * 0.9)
            })
        
        # Comparison metrics
        if v14m_result is not None and v14_4k_result is not None:
            metrics.update({
                'v14m_vs_v14_4k_mse': float(np.mean((v14m_result - v14_4k_result) ** 2)),
                'v14m_closer_to_target': bool(metrics['v14m_mse_to_target'] < metrics['v14_4k_mse_to_target']),
                'v14_4k_more_stable': bool(abs(metrics['v14_4k_brightness_change']) < abs(metrics['v14m_brightness_change']))
            })
        
        return metrics
    
    def create_visual_comparison(self, original, v14m_result, v14_4k_result, target, metrics, pair_id):
        """Create visual comparison with metrics overlay"""
        try:
            # Convert to display format
            original_display = (np.clip(original, 0, 1) * 255).astype(np.uint8)
            target_display = (np.clip(target, 0, 1) * 255).astype(np.uint8)
            
            # Handle missing results
            if v14m_result is not None:
                v14m_display = (np.clip(v14m_result, 0, 1) * 255).astype(np.uint8)
            else:
                v14m_display = np.zeros_like(original_display)
            
            if v14_4k_result is not None:
                v14_4k_display = (np.clip(v14_4k_result, 0, 1) * 255).astype(np.uint8)
            else:
                v14_4k_display = np.zeros_like(original_display)
            
            # Create 4-way horizontal comparison
            h, w = original_display.shape[:2]
            
            comparison = np.hstack([
                cv2.cvtColor(original_display, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(v14m_display, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(v14_4k_display, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(target_display, cv2.COLOR_RGB2BGR)
            ])
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(1.0, h / 1000)
            thickness = max(2, int(h / 500))
            
            # Labels
            cv2.putText(comparison, "iPhone Original", (20, 50), font, font_scale, (255, 255, 255), thickness)
            cv2.putText(comparison, "v1.4m (Exposure Fixed)", (w + 20, 50), font, font_scale, (0, 100, 255), thickness)
            cv2.putText(comparison, "v1.4 4K (Stable)", (2*w + 20, 50), font, font_scale, (255, 165, 0), thickness)
            cv2.putText(comparison, "Sony A7S3 Target", (3*w + 20, 50), font, font_scale, (0, 255, 0), thickness)
            
            # Add metrics text overlay
            if metrics:
                y_start = h - 200
                line_height = 25
                text_size = 0.6
                
                # v1.4m metrics
                if 'v14m_brightness_change' in metrics:
                    brightness_text = f"v1.4m Brightness: {metrics['v14m_brightness_change']:+.3f}"
                    cv2.putText(comparison, brightness_text, (w + 20, y_start), font, text_size, (0, 100, 255), 2)
                    
                    mse_text = f"v1.4m MSE: {metrics['v14m_mse_to_target']:.4f}"
                    cv2.putText(comparison, mse_text, (w + 20, y_start + line_height), font, text_size, (0, 100, 255), 2)
                
                # v1.4 4K metrics
                if 'v14_4k_brightness_change' in metrics:
                    brightness_text = f"v1.4 4K Brightness: {metrics['v14_4k_brightness_change']:+.3f}"
                    cv2.putText(comparison, brightness_text, (2*w + 20, y_start), font, text_size, (255, 165, 0), 2)
                    
                    mse_text = f"v1.4 4K MSE: {metrics['v14_4k_mse_to_target']:.4f}"
                    cv2.putText(comparison, mse_text, (2*w + 20, y_start + line_height), font, text_size, (255, 165, 0), 2)
                
                # Winner annotation
                if 'v14m_closer_to_target' in metrics:
                    if metrics['v14m_closer_to_target']:
                        cv2.putText(comparison, "CLOSER TO TARGET", (w + 20, y_start + 2*line_height), font, text_size, (0, 255, 0), 2)
                    else:
                        cv2.putText(comparison, "CLOSER TO TARGET", (2*w + 20, y_start + 2*line_height), font, text_size, (0, 255, 0), 2)
            
            return comparison
            
        except Exception as e:
            print(f"❌ Error creating visual comparison: {e}")
            return None
    
    def batch_comparison(self, start_pair=1, num_pairs=5, target_size=2048):
        """Compare multiple pairs"""
        print(f"\n🚀 BATCH MODEL COMPARISON - {num_pairs} pairs at {target_size}x{target_size}")
        print("=" * 70)
        
        success_count = 0
        all_metrics = []
        
        for i in range(start_pair, start_pair + num_pairs):
            if self.create_5_way_comparison(i, target_size=target_size):
                success_count += 1
                
                # Load and accumulate metrics
                metrics_path = self.results_dir / f"metrics_{i:03d}.json"
                if metrics_path.exists():
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                        metrics['pair_id'] = i
                        all_metrics.append(metrics)
        
        # Generate summary report
        if all_metrics:
            self.generate_summary_report(all_metrics)
        
        print(f"\n✅ BATCH COMPARISON COMPLETE: {success_count}/{num_pairs} pairs processed")
        print(f"📁 Results in: {self.results_dir}")
    
    def generate_summary_report(self, all_metrics):
        """Generate comprehensive summary report"""
        print(f"\n📊 GENERATING SUMMARY REPORT")
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'pairs_analyzed': len(all_metrics),
            'v14m_stats': {},
            'v14_4k_stats': {},
            'comparison_stats': {}
        }
        
        # Extract metrics for analysis
        v14m_brightness_changes = [m['v14m_brightness_change'] for m in all_metrics if 'v14m_brightness_change' in m]
        v14_4k_brightness_changes = [m['v14_4k_brightness_change'] for m in all_metrics if 'v14_4k_brightness_change' in m]
        
        v14m_mse_values = [m['v14m_mse_to_target'] for m in all_metrics if 'v14m_mse_to_target' in m]
        v14_4k_mse_values = [m['v14_4k_mse_to_target'] for m in all_metrics if 'v14_4k_mse_to_target' in m]
        
        # v1.4m statistics
        if v14m_brightness_changes:
            summary['v14m_stats'] = {
                'avg_brightness_change': float(np.mean(v14m_brightness_changes)),
                'brightness_change_std': float(np.std(v14m_brightness_changes)),
                'avg_mse_to_target': float(np.mean(v14m_mse_values)) if v14m_mse_values else None,
                'exposure_fix_rate': float(np.mean([1 if x >= -0.05 else 0 for x in v14m_brightness_changes]))
            }
        
        # v1.4 4K statistics
        if v14_4k_brightness_changes:
            summary['v14_4k_stats'] = {
                'avg_brightness_change': float(np.mean(v14_4k_brightness_changes)),
                'brightness_change_std': float(np.std(v14_4k_brightness_changes)),
                'avg_mse_to_target': float(np.mean(v14_4k_mse_values)) if v14_4k_mse_values else None,
                'stability_score': float(1.0 - np.std(v14_4k_brightness_changes))  # Lower std = more stable
            }
        
        # Comparison statistics
        if v14m_brightness_changes and v14_4k_brightness_changes:
            v14m_wins = sum(1 for m in all_metrics if m.get('v14m_closer_to_target', False))
            v14_4k_wins = len(all_metrics) - v14m_wins
            
            summary['comparison_stats'] = {
                'v14m_wins': v14m_wins,
                'v14_4k_wins': v14_4k_wins,
                'v14m_win_rate': float(v14m_wins / len(all_metrics)),
                'v14_4k_more_stable_rate': float(np.mean([1 if m.get('v14_4k_more_stable', False) else 0 for m in all_metrics])),
                'recommendation': 'v1.4m' if v14m_wins > v14_4k_wins else 'v1.4 4K'
            }
        
        # Save summary
        summary_path = self.results_dir / "comparison_summary_report.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print(f"✅ Analyzed {summary['pairs_analyzed']} pairs")
        
        if 'v14m_stats' in summary and summary['v14m_stats']:
            stats = summary['v14m_stats']
            print(f"🔥 v1.4m: Avg brightness change {stats['avg_brightness_change']:+.3f}, MSE {stats.get('avg_mse_to_target', 0):.4f}")
        
        if 'v14_4k_stats' in summary and summary['v14_4k_stats']:
            stats = summary['v14_4k_stats']
            print(f"⚡ v1.4 4K: Avg brightness change {stats['avg_brightness_change']:+.3f}, MSE {stats.get('avg_mse_to_target', 0):.4f}")
        
        if 'comparison_stats' in summary and summary['comparison_stats']:
            comp = summary['comparison_stats']
            print(f"🏆 Winner: {comp['recommendation']} ({comp['v14m_wins']} vs {comp['v14_4k_wins']})")
            print(f"📊 v1.4 4K more stable: {comp['v14_4k_more_stable_rate']*100:.1f}% of pairs")
        
        print(f"💾 Summary: {summary_path}")


def main():
    """Interactive comparison interface"""
    print("🎬 CINEMA MODEL COMPARISON - v1.4m vs v1.4 4K")
    print("=" * 50)
    print("Compare exposure-fixed v1.4m against stable v1.4 4K")
    
    try:
        comparator = ModelComparator()
    except Exception as e:
        print(f"❌ Failed to initialize comparator: {e}")
        return
    
    while True:
        print("\n🎯 COMPARISON OPTIONS:")
        print("1. Single pair comparison")
        print("2. Batch comparison (multiple pairs)")
        print("3. High-resolution comparison")
        print("4. Exit")
        
        try:
            choice = input("\nChoice (1-4): ").strip()
            
            if choice == "1":
                pair_input = input("Pair ID (1-79): ").strip()
                if not pair_input:
                    print("❌ Please enter a pair ID")
                    continue
                    
                pair_id = int(pair_input)
                
                success = comparator.create_5_way_comparison(pair_id, target_size=2048)
                if success:
                    print(f"✅ SUCCESS: Pair {pair_id:03d} comparison complete")
                else:
                    print(f"❌ FAILED: Could not compare pair {pair_id:03d}")
                
            elif choice == "2":
                start_input = input("Start pair (default 1): ").strip()
                start = int(start_input) if start_input else 1
                
                count_input = input("Number of pairs (default 5): ").strip()
                count = int(count_input) if count_input else 5
                
                comparator.batch_comparison(start_pair=start, num_pairs=count, target_size=2048)
                
            elif choice == "3":
                pair_input = input("Pair ID: ").strip()
                if not pair_input:
                    print("❌ Please enter a pair ID")
                    continue
                    
                pair_id = int(pair_input)
                
                size_input = input("Target resolution (default 4096): ").strip()
                size = int(size_input) if size_input else 4096
                
                success = comparator.create_5_way_comparison(pair_id, target_size=size)
                if success:
                    print(f"✅ SUCCESS: High-res comparison of pair {pair_id:03d}")
                else:
                    print(f"❌ FAILED: Could not compare pair {pair_id:03d}")
                
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
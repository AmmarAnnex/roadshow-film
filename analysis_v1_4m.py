#!/usr/bin/env python3
"""
Professional Analysis Tool for Cinema Model v1.4m - Exposure Fixed
Based on industry standards for color science validation
Compatible with the new ExposureFixedColorTransform architecture
"""

import cv2
import numpy as np
from pathlib import Path
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
import rawpy

class ProfessionalColorAnalyzer:
    """Professional color analysis for v1.4m model"""
    
    def __init__(self, model_path="models/cinema_v1_4m_model.pth"):
        self.model_path = Path(model_path)
        self.results_dir = Path("data/results/v1_4m_analysis")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model if available
        self.model = None
        if self.model_path.exists():
            self.load_model()
    
    def load_model(self):
        """Load the trained v1.4m model"""
        try:
            from cinema_v1_4m import ExposureFixedColorTransform
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = ExposureFixedColorTransform().to(device)
            
            # Load checkpoint with weights_only=False for compatibility
            checkpoint = torch.load(self.model_path, map_location=device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            print(f"✅ Model v1.4m loaded: {self.model_path}")
            print(f"   Training pairs: {checkpoint.get('training_pairs', 'Unknown')}")
            print(f"   Best loss: {checkpoint.get('loss', 'Unknown'):.4f}")
            
            # Print exposure parameters for verification
            with torch.no_grad():
                print(f"   Exposure factor: {self.model.global_exposure.item():.3f}")
                print(f"   Shadow lift: {self.model.shadows.item():.3f}")
                print(f"   Residual strength: {self.model.residual_strength.item():.3f}")
            
        except Exception as e:
            print(f"❌ Error loading v1.4m model: {e}")
            self.model = None
    
    def load_raw_image(self, file_path, target_size=768):
        """Load RAW image with professional processing"""
        try:
            with rawpy.imread(str(file_path)) as raw:
                rgb = raw.postprocess(
                    demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                    half_size=False,
                    use_camera_wb=True,
                    output_color=rawpy.ColorSpace.sRGB,
                    output_bps=16,
                    bright=1.0,
                    no_auto_bright=True,
                    gamma=(1, 1)  # Linear
                )
            
            # Process same as training
            rgb_float = rgb.astype(np.float32) / 65535.0
            h, w = rgb_float.shape[:2]
            size = min(h, w)
            start_y = (h - size) // 2
            start_x = (w - size) // 2
            rgb_crop = rgb_float[start_y:start_y + size, start_x:start_x + size]
            rgb_resized = cv2.resize(rgb_crop, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
            
            return rgb_resized
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def calculate_delta_e(self, rgb1, rgb2):
        """Calculate Delta E color difference (professional metric)"""
        def rgb_to_lab(rgb):
            # Convert to XYZ first
            rgb_linear = np.where(rgb > 0.04045, 
                                 np.power((rgb + 0.055) / 1.055, 2.4), 
                                 rgb / 12.92)
            
            # sRGB to XYZ matrix
            M = np.array([
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041]
            ])
            
            # Reshape for matrix multiplication
            shape = rgb_linear.shape
            rgb_flat = rgb_linear.reshape(-1, 3)
            xyz_flat = np.dot(rgb_flat, M.T)
            xyz = xyz_flat.reshape(shape)
            
            # Normalize by D65 illuminant
            xyz[:,:,0] /= 0.95047
            xyz[:,:,1] /= 1.00000
            xyz[:,:,2] /= 1.08883
            
            # XYZ to LAB with safety checks
            def f(t):
                return np.where(t > 0.008856, np.power(np.maximum(t, 1e-8), 1/3), (7.787 * t + 16/116))
            
            fx = f(xyz[:,:,0])
            fy = f(xyz[:,:,1])
            fz = f(xyz[:,:,2])
            
            L = 116 * fy - 16
            a = 500 * (fx - fy)
            b = 200 * (fy - fz)
            
            return np.stack([L, a, b], axis=-1)
        
        lab1 = rgb_to_lab(rgb1)
        lab2 = rgb_to_lab(rgb2)
        
        # Delta E calculation
        delta_L = lab1[:,:,0] - lab2[:,:,0]
        delta_a = lab1[:,:,1] - lab2[:,:,1]
        delta_b = lab1[:,:,2] - lab2[:,:,2]
        
        delta_e = np.sqrt(delta_L**2 + delta_a**2 + delta_b**2)
        return np.mean(delta_e)
    
    def analyze_exposure_improvement(self, original, transformed, target):
        """Analyze exposure improvements from v1.4m"""
        metrics = {}
        
        # Brightness analysis
        orig_brightness = np.mean(original)
        trans_brightness = np.mean(transformed)
        target_brightness = np.mean(target)
        
        metrics['original_brightness'] = orig_brightness
        metrics['transformed_brightness'] = trans_brightness
        metrics['target_brightness'] = target_brightness
        
        # Exposure correction metrics
        metrics['brightness_change'] = trans_brightness - orig_brightness
        metrics['brightness_accuracy'] = 1 - abs(trans_brightness - target_brightness)
        
        # Check for underexposure (key issue in v1.4l)
        metrics['is_underexposed'] = trans_brightness < orig_brightness * 0.8
        metrics['exposure_improvement'] = not metrics['is_underexposed']
        
        # Shadow detail analysis
        shadow_mask = original < 0.2
        if np.any(shadow_mask):
            orig_shadow_mean = np.mean(original[shadow_mask])
            trans_shadow_mean = np.mean(transformed[shadow_mask])
            metrics['shadow_lift'] = trans_shadow_mean - orig_shadow_mean
        else:
            metrics['shadow_lift'] = 0
        
        # Highlight preservation
        highlight_mask = original > 0.8
        if np.any(highlight_mask):
            orig_highlight_mean = np.mean(original[highlight_mask])
            trans_highlight_mean = np.mean(transformed[highlight_mask])
            metrics['highlight_preservation'] = 1 - abs(trans_highlight_mean - orig_highlight_mean)
        else:
            metrics['highlight_preservation'] = 1
        
        return metrics
    
    def analyze_color_reproduction(self, original, transformed, target):
        """Professional color reproduction analysis"""
        metrics = {}
        
        # Delta E (professional color difference)
        metrics['delta_e_original_target'] = self.calculate_delta_e(original, target)
        metrics['delta_e_transformed_target'] = self.calculate_delta_e(transformed, target)
        metrics['delta_e_improvement'] = metrics['delta_e_original_target'] - metrics['delta_e_transformed_target']
        
        # SSIM (structural similarity) - Fixed for proper data range
        orig_norm = np.clip(original, 0, 1)
        trans_norm = np.clip(transformed, 0, 1)
        target_norm = np.clip(target, 0, 1)
        
        ssim_r = ssim(orig_norm[:,:,0], trans_norm[:,:,0], data_range=1.0)
        ssim_g = ssim(orig_norm[:,:,1], trans_norm[:,:,1], data_range=1.0)
        ssim_b = ssim(orig_norm[:,:,2], trans_norm[:,:,2], data_range=1.0)
        metrics['ssim_avg'] = (ssim_r + ssim_g + ssim_b) / 3
        
        # Color accuracy per channel
        for i, channel in enumerate(['R', 'G', 'B']):
            orig_target_mse = mean_squared_error(original[:,:,i].flatten(), target[:,:,i].flatten())
            trans_target_mse = mean_squared_error(transformed[:,:,i].flatten(), target[:,:,i].flatten())
            metrics[f'mse_{channel.lower()}_improvement'] = orig_target_mse - trans_target_mse
            metrics[f'accuracy_{channel.lower()}'] = 1 - trans_target_mse
        
        # Histogram analysis
        def hist_correlation(img1, img2):
            img1_255 = (np.clip(img1, 0, 1) * 255).astype(np.uint8)
            img2_255 = (np.clip(img2, 0, 1) * 255).astype(np.uint8)
            hist1 = cv2.calcHist([img1_255], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
            hist2 = cv2.calcHist([img2_255], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
            return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        metrics['hist_corr_orig_target'] = hist_correlation(original, target)
        metrics['hist_corr_trans_target'] = hist_correlation(transformed, target)
        
        return metrics
    
    def create_v1_4m_comparison(self, original, transformed, target, pair_id, save_path):
        """Create professional comparison with v1.4m specific metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Cinema v1.4m Analysis - Pair {pair_id:03d} (Exposure Fixed)', fontsize=16, fontweight='bold')
        
        # Images - Ensure proper display range
        orig_display = np.clip(original, 0, 1)
        trans_display = np.clip(transformed, 0, 1)
        target_display = np.clip(target, 0, 1)
        
        axes[0,0].imshow(orig_display)
        axes[0,0].set_title('iPhone Original', fontweight='bold')
        axes[0,0].axis('off')
        
        axes[0,1].imshow(trans_display)
        axes[0,1].set_title('Cinema v1.4m Transform', fontweight='bold', color='blue')
        axes[0,1].axis('off')
        
        axes[0,2].imshow(target_display)
        axes[0,2].set_title('Sony A7S3 Target', fontweight='bold', color='green')
        axes[0,2].axis('off')
        
        # Analyze both color and exposure
        color_metrics = self.analyze_color_reproduction(original, transformed, target)
        exposure_metrics = self.analyze_exposure_improvement(original, transformed, target)
        
        # Delta E comparison
        delta_e_data = [
            color_metrics['delta_e_original_target'],
            color_metrics['delta_e_transformed_target']
        ]
        axes[1,0].bar(['Original→Target', 'Transform→Target'], delta_e_data, 
                     color=['red', 'blue'], alpha=0.7)
        axes[1,0].set_title('Delta E Color Difference\n(Lower = Better)')
        axes[1,0].set_ylabel('Delta E')
        
        # Add improvement text
        improvement = color_metrics['delta_e_improvement']
        color = 'green' if improvement > 0 else 'red'
        axes[1,0].text(0.5, max(delta_e_data) * 0.8, f'Improvement: {improvement:.2f}', 
                      ha='center', color=color, fontweight='bold')
        
        # Exposure analysis
        brightness_data = [
            exposure_metrics['original_brightness'],
            exposure_metrics['transformed_brightness'],
            exposure_metrics['target_brightness']
        ]
        axes[1,1].bar(['Original', 'Transform', 'Target'], brightness_data, 
                     color=['gray', 'blue', 'green'], alpha=0.7)
        axes[1,1].set_title('Brightness Analysis\n(v1.4m Exposure Fix)')
        axes[1,1].set_ylabel('Average Brightness')
        
        # Add exposure status
        if exposure_metrics['exposure_improvement']:
            status_text = "✅ No Underexposure"
            status_color = 'green'
        else:
            status_text = "⚠️ Still Underexposed"
            status_color = 'red'
        
        axes[1,1].text(0.5, max(brightness_data) * 0.9, status_text, 
                      ha='center', color=status_color, fontweight='bold')
        
        # Professional metrics summary
        metrics_text = f"""v1.4m Professional Metrics:

EXPOSURE ANALYSIS:
Brightness Change: {exposure_metrics['brightness_change']:+.3f}
Shadow Lift: {exposure_metrics['shadow_lift']:+.3f}
Highlight Preservation: {exposure_metrics['highlight_preservation']:.3f}
Exposure Fixed: {exposure_metrics['exposure_improvement']}

COLOR ANALYSIS:
SSIM: {color_metrics['ssim_avg']:.3f}
Delta E Improvement: {color_metrics['delta_e_improvement']:.2f}
Histogram Correlation: {color_metrics['hist_corr_trans_target']:.3f}

Channel Accuracy:
Red: {color_metrics['accuracy_r']:.3f}
Green: {color_metrics['accuracy_g']:.3f}
Blue: {color_metrics['accuracy_b']:.3f}"""
        
        axes[1,2].text(0.05, 0.95, metrics_text, transform=axes[1,2].transAxes,
                      verticalalignment='top', fontfamily='monospace', fontsize=8)
        axes[1,2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Combine metrics
        combined_metrics = {**color_metrics, **exposure_metrics}
        return combined_metrics
    
    def analyze_training_pairs(self, max_pairs=10):
        """Analyze multiple training pairs for v1.4m evaluation"""
        print("🔍 PROFESSIONAL COLOR ANALYSIS v1.4m (EXPOSURE FIXED)")
        print("=" * 60)
        
        if self.model is None:
            print("❌ Model not loaded. Train v1.4m model first.")
            return
        
        training_pairs_dir = Path("data/training_pairs")
        device = next(self.model.parameters()).device
        
        all_metrics = []
        exposure_fixed_count = 0
        
        for i in range(1, max_pairs + 1):
            iphone_file = training_pairs_dir / f"iphone_{i:03d}.dng"
            sony_file = training_pairs_dir / f"sony_{i:03d}.arw"
            
            if not (iphone_file.exists() and sony_file.exists()):
                continue
            
            print(f"\n📷 Analyzing Pair {i:03d}")
            
            # Load images
            original = self.load_raw_image(iphone_file)
            target = self.load_raw_image(sony_file)
            
            if original is None or target is None:
                print(f"   ❌ Failed to load pair {i}")
                continue
            
            # Transform with v1.4m model
            original_tensor = torch.from_numpy(original).permute(2, 0, 1).unsqueeze(0).to(device)
            
            with torch.no_grad():
                transformed_tensor = self.model(original_tensor)
                transformed = transformed_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            # Create comparison
            comparison_path = self.results_dir / f"v1_4m_analysis_{i:03d}.png"
            metrics = self.create_v1_4m_comparison(original, transformed, target, i, comparison_path)
            
            # Add pair info
            metrics['pair_id'] = i
            all_metrics.append(metrics)
            
            # Track exposure improvements
            if metrics['exposure_improvement']:
                exposure_fixed_count += 1
            
            print(f"   ✅ Delta E: {metrics['delta_e_transformed_target']:.2f}")
            print(f"   ✅ SSIM: {metrics['ssim_avg']:.3f}")
            print(f"   ✅ Exposure Fixed: {metrics['exposure_improvement']}")
            print(f"   ✅ Brightness Change: {metrics['brightness_change']:+.3f}")
            print(f"   ✅ Saved: {comparison_path}")
        
        # Generate v1.4m summary report
        self.generate_v1_4m_summary_report(all_metrics, exposure_fixed_count)
        
        return all_metrics
    
    def generate_v1_4m_summary_report(self, all_metrics, exposure_fixed_count):
        """Generate comprehensive v1.4m summary report"""
        if not all_metrics:
            return
        
        print(f"\n📊 GENERATING V1.4M SUMMARY REPORT")
        print("=" * 40)
        
        # Calculate statistics
        delta_e_values = [m['delta_e_transformed_target'] for m in all_metrics]
        ssim_values = [m['ssim_avg'] for m in all_metrics]
        improvements = [m['delta_e_improvement'] for m in all_metrics]
        brightness_changes = [m['brightness_change'] for m in all_metrics]
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'model_version': 'v1.4m (Exposure Fixed)',
            'pairs_analyzed': len(all_metrics),
            'exposure_statistics': {
                'exposure_fixed_count': exposure_fixed_count,
                'exposure_fix_rate': exposure_fixed_count / len(all_metrics),
                'avg_brightness_change': np.mean(brightness_changes),
                'underexposure_eliminated': exposure_fixed_count == len(all_metrics)
            },
            'color_statistics': {
                'delta_e_mean': np.mean(delta_e_values),
                'delta_e_std': np.std(delta_e_values),
                'ssim_mean': np.mean(ssim_values),
                'improvement_rate': np.mean([1 if x > 0 else 0 for x in improvements])
            },
            'detailed_metrics': all_metrics
        }
        
        # Save detailed report
        report_path = self.results_dir / "v1_4m_summary_report.json"
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        exp_stats = summary['exposure_statistics']
        color_stats = summary['color_statistics']
        
        print(f"✅ Analyzed {summary['pairs_analyzed']} pairs")
        print(f"🔆 Exposure Fix Rate: {exp_stats['exposure_fix_rate']*100:.1f}%")
        print(f"🔆 Underexposure Eliminated: {exp_stats['underexposure_eliminated']}")
        print(f"🔆 Avg Brightness Change: {exp_stats['avg_brightness_change']:+.3f}")
        print(f"📊 Delta E: {color_stats['delta_e_mean']:.2f}")
        print(f"📊 SSIM: {color_stats['ssim_mean']:.3f}")
        print(f"💾 Report: {report_path}")

def main():
    """Main analysis routine for v1.4m"""
    analyzer = ProfessionalColorAnalyzer()
    
    print("🎬 PROFESSIONAL COLOR ANALYSIS FOR CINEMA v1.4m")
    print("Exposure-Fixed Model Analysis")
    print("\nOptions:")
    print("1. Analyze first 10 pairs")
    print("2. Analyze first 5 pairs (quick test)")
    print("3. Full analysis (all available pairs)")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            analyzer.analyze_training_pairs(max_pairs=10)
        elif choice == "2":
            analyzer.analyze_training_pairs(max_pairs=5)
        elif choice == "3":
            analyzer.analyze_training_pairs(max_pairs=79)
        else:
            print("Invalid choice")
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
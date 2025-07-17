#!/usr/bin/env python3
"""
Professional Analysis Tool for Cinema Model v1.4l
Based on industry standards for color science validation
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
    """Professional color analysis based on industry standards"""
    
    def __init__(self, model_path="models/cinema_v1_4l_model.pth"):
        self.model_path = Path(model_path)
        self.results_dir = Path("data/results/v1_4l_analysis")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model if available
        self.model = None
        if self.model_path.exists():
            self.load_model()
    
    def load_model(self):
        """Load the trained v1.4l model"""
        try:
            from cinema_v1_4l import ProfessionalColorTransform
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = ProfessionalColorTransform().to(device)
            
            checkpoint = torch.load(self.model_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            print(f"✅ Model loaded: {self.model_path}")
            print(f"   Training pairs: {checkpoint.get('training_pairs', 'Unknown')}")
            print(f"   Best loss: {checkpoint.get('loss', 'Unknown'):.4f}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
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
        # Convert RGB to LAB for perceptually uniform color difference
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
            
            # XYZ to LAB
            def f(t):
                return np.where(t > 0.008856, np.power(t, 1/3), (7.787 * t + 16/116))
            
            fx = f(xyz[:,:,0])
            fy = f(xyz[:,:,1])
            fz = f(xyz[:,:,2])
            
            L = 116 * fy - 16
            a = 500 * (fx - fy)
            b = 200 * (fy - fz)
            
            return np.stack([L, a, b], axis=-1)
        
        lab1 = rgb_to_lab(rgb1)
        lab2 = rgb_to_lab(rgb2)
        
        # Delta E 2000 approximation
        delta_L = lab1[:,:,0] - lab2[:,:,0]
        delta_a = lab1[:,:,1] - lab2[:,:,1]
        delta_b = lab1[:,:,2] - lab2[:,:,2]
        
        delta_e = np.sqrt(delta_L**2 + delta_a**2 + delta_b**2)
        return np.mean(delta_e)
    
    def analyze_color_reproduction(self, original, transformed, target):
        """Professional color reproduction analysis"""
        metrics = {}
        
        # Delta E (professional color difference)
        metrics['delta_e_original_target'] = self.calculate_delta_e(original, target)
        metrics['delta_e_transformed_target'] = self.calculate_delta_e(transformed, target)
        metrics['delta_e_improvement'] = metrics['delta_e_original_target'] - metrics['delta_e_transformed_target']
        
        # SSIM (structural similarity)
        ssim_r = ssim(original[:,:,0], transformed[:,:,0])
        ssim_g = ssim(original[:,:,1], transformed[:,:,1])
        ssim_b = ssim(original[:,:,2], transformed[:,:,2])
        metrics['ssim_avg'] = (ssim_r + ssim_g + ssim_b) / 3
        
        # Color accuracy per channel
        for i, channel in enumerate(['R', 'G', 'B']):
            orig_target_mse = mean_squared_error(original[:,:,i].flatten(), target[:,:,i].flatten())
            trans_target_mse = mean_squared_error(transformed[:,:,i].flatten(), target[:,:,i].flatten())
            metrics[f'mse_{channel.lower()}_improvement'] = orig_target_mse - trans_target_mse
            metrics[f'accuracy_{channel.lower()}'] = 1 - trans_target_mse
        
        # Histogram analysis
        def hist_correlation(img1, img2):
            hist1 = cv2.calcHist([img1], [0, 1, 2], None, [32, 32, 32], [0, 1, 0, 1, 0, 1])
            hist2 = cv2.calcHist([img2], [0, 1, 2], None, [32, 32, 32], [0, 1, 0, 1, 0, 1])
            return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        metrics['hist_corr_orig_target'] = hist_correlation(original, target)
        metrics['hist_corr_trans_target'] = hist_correlation(transformed, target)
        
        # Professional gamut analysis
        def analyze_gamut(img):
            """Analyze color gamut coverage"""
            # Convert to HSV for gamut analysis
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            saturation_mean = np.mean(hsv[:,:,1])
            saturation_std = np.std(hsv[:,:,1])
            return saturation_mean, saturation_std
        
        orig_sat_mean, orig_sat_std = analyze_gamut(original)
        trans_sat_mean, trans_sat_std = analyze_gamut(transformed)
        target_sat_mean, target_sat_std = analyze_gamut(target)
        
        metrics['saturation_accuracy'] = 1 - abs(trans_sat_mean - target_sat_mean)
        metrics['saturation_consistency'] = 1 - abs(trans_sat_std - target_sat_std)
        
        return metrics
    
    def create_professional_comparison(self, original, transformed, target, pair_id, save_path):
        """Create professional side-by-side comparison"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Professional Color Analysis - Pair {pair_id:03d}', fontsize=16, fontweight='bold')
        
        # Images
        axes[0,0].imshow(original)
        axes[0,0].set_title('iPhone Original', fontweight='bold')
        axes[0,0].axis('off')
        
        axes[0,1].imshow(transformed)
        axes[0,1].set_title('Cinema v1.4l Transform', fontweight='bold', color='blue')
        axes[0,1].axis('off')
        
        axes[0,2].imshow(target)
        axes[0,2].set_title('Sony A7S3 Target', fontweight='bold', color='green')
        axes[0,2].axis('off')
        
        # Color analysis
        metrics = self.analyze_color_reproduction(original, transformed, target)
        
        # Delta E comparison
        delta_e_data = [
            metrics['delta_e_original_target'],
            metrics['delta_e_transformed_target']
        ]
        axes[1,0].bar(['Original→Target', 'Transform→Target'], delta_e_data, 
                     color=['red', 'blue'], alpha=0.7)
        axes[1,0].set_title('Delta E Color Difference\n(Lower = Better)')
        axes[1,0].set_ylabel('Delta E')
        
        # Add improvement text
        improvement = metrics['delta_e_improvement']
        color = 'green' if improvement > 0 else 'red'
        axes[1,0].text(0.5, max(delta_e_data) * 0.8, f'Improvement: {improvement:.2f}', 
                      ha='center', color=color, fontweight='bold')
        
        # Channel accuracy
        channel_acc = [
            metrics['accuracy_r'],
            metrics['accuracy_g'], 
            metrics['accuracy_b']
        ]
        axes[1,1].bar(['Red', 'Green', 'Blue'], channel_acc, 
                     color=['red', 'green', 'blue'], alpha=0.7)
        axes[1,1].set_title('Channel Accuracy\n(Higher = Better)')
        axes[1,1].set_ylabel('Accuracy')
        axes[1,1].set_ylim(0, 1)
        
        # Professional metrics summary
        metrics_text = f"""Professional Metrics:
        
SSIM: {metrics['ssim_avg']:.3f}
Delta E Improvement: {metrics['delta_e_improvement']:.2f}
Histogram Correlation: {metrics['hist_corr_trans_target']:.3f}
Saturation Accuracy: {metrics['saturation_accuracy']:.3f}

Color Accuracy:
Red: {metrics['accuracy_r']:.3f}
Green: {metrics['accuracy_g']:.3f}  
Blue: {metrics['accuracy_b']:.3f}"""
        
        axes[1,2].text(0.05, 0.95, metrics_text, transform=axes[1,2].transAxes,
                      verticalalignment='top', fontfamily='monospace', fontsize=9)
        axes[1,2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return metrics
    
    def analyze_training_pairs(self, max_pairs=10):
        """Analyze multiple training pairs for comprehensive evaluation"""
        print("🔍 PROFESSIONAL COLOR ANALYSIS v1.4l")
        print("=" * 50)
        
        if self.model is None:
            print("❌ Model not loaded. Train model first.")
            return
        
        training_pairs_dir = Path("data/training_pairs")
        device = next(self.model.parameters()).device
        
        all_metrics = []
        
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
            
            # Transform with model
            original_tensor = torch.from_numpy(original).permute(2, 0, 1).unsqueeze(0).to(device)
            
            with torch.no_grad():
                transformed_tensor = self.model(original_tensor)
                transformed = transformed_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            # Create comparison
            comparison_path = self.results_dir / f"professional_analysis_{i:03d}.png"
            metrics = self.create_professional_comparison(original, transformed, target, i, comparison_path)
            
            # Add pair info
            metrics['pair_id'] = i
            all_metrics.append(metrics)
            
            print(f"   ✅ Delta E: {metrics['delta_e_transformed_target']:.2f}")
            print(f"   ✅ SSIM: {metrics['ssim_avg']:.3f}")
            print(f"   ✅ Saved: {comparison_path}")
        
        # Generate summary report
        self.generate_summary_report(all_metrics)
        
        return all_metrics
    
    def generate_summary_report(self, all_metrics):
        """Generate comprehensive summary report"""
        if not all_metrics:
            return
        
        print(f"\n📊 GENERATING SUMMARY REPORT")
        print("=" * 35)
        
        # Calculate statistics
        delta_e_values = [m['delta_e_transformed_target'] for m in all_metrics]
        ssim_values = [m['ssim_avg'] for m in all_metrics]
        improvements = [m['delta_e_improvement'] for m in all_metrics]
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'model_version': 'v1.4l',
            'pairs_analyzed': len(all_metrics),
            'statistics': {
                'delta_e_mean': np.mean(delta_e_values),
                'delta_e_std': np.std(delta_e_values),
                'delta_e_min': np.min(delta_e_values),
                'delta_e_max': np.max(delta_e_values),
                'ssim_mean': np.mean(ssim_values),
                'ssim_std': np.std(ssim_values),
                'improvement_mean': np.mean(improvements),
                'improvement_rate': np.mean([1 if x > 0 else 0 for x in improvements])
            },
            'detailed_metrics': all_metrics
        }
        
        # Save detailed report
        report_path = self.results_dir / "professional_summary_report.json"
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create summary visualization
        self.create_summary_visualization(summary)
        
        # Print summary
        stats = summary['statistics']
        print(f"✅ Analyzed {summary['pairs_analyzed']} pairs")
        print(f"📊 Delta E: {stats['delta_e_mean']:.2f} ± {stats['delta_e_std']:.2f}")
        print(f"📊 SSIM: {stats['ssim_mean']:.3f} ± {stats['ssim_std']:.3f}")
        print(f"📈 Improvement Rate: {stats['improvement_rate']*100:.1f}%")
        print(f"💾 Report: {report_path}")
    
    def create_summary_visualization(self, summary):
        """Create summary visualization dashboard"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Cinema v1.4l - Professional Analysis Summary', fontsize=16, fontweight='bold')
        
        metrics = summary['detailed_metrics']
        stats = summary['statistics']
        
        # Delta E distribution
        delta_e_values = [m['delta_e_transformed_target'] for m in metrics]
        axes[0,0].hist(delta_e_values, bins=10, alpha=0.7, color='blue', edgecolor='black')
        axes[0,0].axvline(stats['delta_e_mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {stats["delta_e_mean"]:.2f}')
        axes[0,0].set_title('Delta E Distribution')
        axes[0,0].set_xlabel('Delta E')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].legend()
        
        # SSIM distribution
        ssim_values = [m['ssim_avg'] for m in metrics]
        axes[0,1].hist(ssim_values, bins=10, alpha=0.7, color='green', edgecolor='black')
        axes[0,1].axvline(stats['ssim_mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {stats["ssim_mean"]:.3f}')
        axes[0,1].set_title('SSIM Distribution')
        axes[0,1].set_xlabel('SSIM')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].legend()
        
        # Performance by pair
        pair_ids = [m['pair_id'] for m in metrics]
        axes[1,0].scatter(pair_ids, delta_e_values, alpha=0.7, color='blue')
        axes[1,0].set_title('Delta E by Training Pair')
        axes[1,0].set_xlabel('Pair ID')
        axes[1,0].set_ylabel('Delta E')
        
        # Channel accuracy comparison
        avg_r = np.mean([m['accuracy_r'] for m in metrics])
        avg_g = np.mean([m['accuracy_g'] for m in metrics])
        avg_b = np.mean([m['accuracy_b'] for m in metrics])
        
        axes[1,1].bar(['Red', 'Green', 'Blue'], [avg_r, avg_g, avg_b], 
                     color=['red', 'green', 'blue'], alpha=0.7)
        axes[1,1].set_title('Average Channel Accuracy')
        axes[1,1].set_ylabel('Accuracy')
        axes[1,1].set_ylim(0, 1)
        
        plt.tight_layout()
        summary_viz_path = self.results_dir / "summary_visualization.png"
        plt.savefig(summary_viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Summary visualization: {summary_viz_path}")

def main():
    """Main analysis routine"""
    analyzer = ProfessionalColorAnalyzer()
    
    print("🎬 PROFESSIONAL COLOR ANALYSIS FOR CINEMA v1.4l")
    print("Based on industry standards and professional workflows")
    print("\nOptions:")
    print("1. Analyze first 10 pairs")
    print("2. Analyze specific pair")
    print("3. Full analysis (all available pairs)")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            analyzer.analyze_training_pairs(max_pairs=10)
        elif choice == "2":
            pair_id = int(input("Enter pair ID (1-79): "))
            analyzer.analyze_training_pairs(max_pairs=1)  # Could be modified for specific pair
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
            
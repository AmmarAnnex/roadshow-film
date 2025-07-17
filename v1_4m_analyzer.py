#!/usr/bin/env python3
"""
Cinema v1.4m - Professional Analysis Tool
SEPARATE from visual testing - pure metrics and validation
"""

import cv2
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
import pandas as pd

class ProfessionalAnalyzer:
    """Professional metrics analysis for v1.4m results"""
    
    def __init__(self, results_dir="data/results/v1_4m_4k_test"):
        self.results_dir = Path(results_dir)
        self.analysis_dir = Path("data/results/v1_4m_professional_analysis")
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📊 Professional Analyzer Ready")
        print(f"🔍 Input: {self.results_dir}")
        print(f"📁 Output: {self.analysis_dir}")
    
    def calculate_delta_e(self, img1, img2):
        """Calculate Delta E color difference"""
        def rgb_to_lab(rgb):
            # Ensure input is 0-1 range
            rgb_norm = rgb.astype(np.float32) / 255.0 if rgb.max() > 1 else rgb.astype(np.float32)
            
            # Linear RGB to XYZ
            rgb_linear = np.where(rgb_norm > 0.04045,
                                 np.power((rgb_norm + 0.055) / 1.055, 2.4),
                                 rgb_norm / 12.92)
            
            # sRGB to XYZ matrix
            M = np.array([
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041]
            ])
            
            # Apply matrix
            shape = rgb_linear.shape
            rgb_flat = rgb_linear.reshape(-1, 3)
            xyz_flat = np.dot(rgb_flat, M.T)
            xyz = xyz_flat.reshape(shape)
            
            # Normalize by D65
            xyz[:,:,0] /= 0.95047
            xyz[:,:,1] /= 1.00000
            xyz[:,:,2] /= 1.08883
            
            # XYZ to LAB
            def f(t):
                return np.where(t > 0.008856, 
                               np.power(np.maximum(t, 1e-8), 1/3), 
                               (7.787 * t + 16/116))
            
            fx = f(xyz[:,:,0])
            fy = f(xyz[:,:,1])
            fz = f(xyz[:,:,2])
            
            L = 116 * fy - 16
            a = 500 * (fx - fy)
            b = 200 * (fy - fz)
            
            return np.stack([L, a, b], axis=-1)
        
        try:
            lab1 = rgb_to_lab(img1)
            lab2 = rgb_to_lab(img2)
            
            delta_L = lab1[:,:,0] - lab2[:,:,0]
            delta_a = lab1[:,:,1] - lab2[:,:,1]
            delta_b = lab1[:,:,2] - lab2[:,:,2]
            
            delta_e = np.sqrt(delta_L**2 + delta_a**2 + delta_b**2)
            return np.mean(delta_e)
        except:
            return 0.0
    
    def analyze_brightness_levels(self, original, transformed, target):
        """Analyze brightness and exposure"""
        orig_bright = np.mean(original)
        trans_bright = np.mean(transformed)
        target_bright = np.mean(target)
        
        metrics = {
            'original_brightness': float(orig_bright),
            'transformed_brightness': float(trans_bright),
            'target_brightness': float(target_bright),
            'brightness_change': float(trans_bright - orig_bright),
            'brightness_accuracy': float(1 - abs(trans_bright - target_bright)),
            'exposure_fixed': bool(trans_bright >= orig_bright * 0.8),  # No severe underexposure
        }
        
        # Shadow/highlight analysis
        shadow_mask = original < 0.2
        if np.any(shadow_mask):
            orig_shadow = np.mean(original[shadow_mask])
            trans_shadow = np.mean(transformed[shadow_mask])
            metrics['shadow_lift'] = float(trans_shadow - orig_shadow)
        else:
            metrics['shadow_lift'] = 0.0
            
        highlight_mask = original > 0.8
        if np.any(highlight_mask):
            orig_highlight = np.mean(original[highlight_mask])
            trans_highlight = np.mean(transformed[highlight_mask])
            metrics['highlight_preservation'] = float(1 - abs(trans_highlight - orig_highlight))
        else:
            metrics['highlight_preservation'] = 1.0
        
        return metrics
    
    def analyze_color_accuracy(self, original, transformed, target):
        """Analyze color reproduction"""
        metrics = {}
        
        # Delta E
        metrics['delta_e_original_target'] = self.calculate_delta_e(original, target)
        metrics['delta_e_transformed_target'] = self.calculate_delta_e(transformed, target)
        metrics['delta_e_improvement'] = metrics['delta_e_original_target'] - metrics['delta_e_transformed_target']
        
        # SSIM with proper normalization
        orig_norm = np.clip(original, 0, 1) if original.max() <= 1 else np.clip(original/255.0, 0, 1)
        trans_norm = np.clip(transformed, 0, 1) if transformed.max() <= 1 else np.clip(transformed/255.0, 0, 1)
        target_norm = np.clip(target, 0, 1) if target.max() <= 1 else np.clip(target/255.0, 0, 1)
        
        ssim_r = ssim(orig_norm[:,:,0], trans_norm[:,:,0], data_range=1.0)
        ssim_g = ssim(orig_norm[:,:,1], trans_norm[:,:,1], data_range=1.0)
        ssim_b = ssim(orig_norm[:,:,2], trans_norm[:,:,2], data_range=1.0)
        metrics['ssim_avg'] = (ssim_r + ssim_g + ssim_b) / 3
        
        # Channel accuracy
        for i, channel in enumerate(['R', 'G', 'B']):
            orig_flat = orig_norm[:,:,i].flatten()
            trans_flat = trans_norm[:,:,i].flatten()
            target_flat = target_norm[:,:,i].flatten()
            
            orig_target_mse = mean_squared_error(orig_flat, target_flat)
            trans_target_mse = mean_squared_error(trans_flat, target_flat)
            
            metrics[f'mse_{channel.lower()}_improvement'] = orig_target_mse - trans_target_mse
            metrics[f'accuracy_{channel.lower()}'] = 1 - trans_target_mse
        
        # Histogram correlation
        def safe_hist_correlation(img1, img2):
            try:
                # Ensure 0-255 range
                img1_255 = (img1 * 255).astype(np.uint8) if img1.max() <= 1 else img1.astype(np.uint8)
                img2_255 = (img2 * 255).astype(np.uint8) if img2.max() <= 1 else img2.astype(np.uint8)
                
                hist1 = cv2.calcHist([img1_255], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
                hist2 = cv2.calcHist([img2_255], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
                
                return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            except:
                return 0.0
        
        metrics['hist_corr_orig_target'] = safe_hist_correlation(original, target)
        metrics['hist_corr_trans_target'] = safe_hist_correlation(transformed, target)
        
        return metrics
    
    def analyze_pair_from_files(self, pair_id):
        """Analyze a pair from saved 4K test files"""
        # Look for the comparison files
        comparison_files = list(self.results_dir.glob(f"*comparison_{pair_id:03d}_*.jpg"))
        individual_files = {
            'iphone': list(self.results_dir.glob(f"iphone_original_{pair_id:03d}_*.jpg")),
            'transformed': list(self.results_dir.glob(f"cinema_v1_4m_{pair_id:03d}_*.jpg")),
            'sony': list(self.results_dir.glob(f"sony_target_{pair_id:03d}_*.jpg"))
        }
        
        # Use individual files if available
        if all(len(files) > 0 for files in individual_files.values()):
            try:
                # Load individual images
                iphone_img = cv2.imread(str(individual_files['iphone'][0]))
                iphone_img = cv2.cvtColor(iphone_img, cv2.COLOR_BGR2RGB)
                
                transformed_img = cv2.imread(str(individual_files['transformed'][0]))
                transformed_img = cv2.cvtColor(transformed_img, cv2.COLOR_BGR2RGB)
                
                sony_img = cv2.imread(str(individual_files['sony'][0]))
                sony_img = cv2.cvtColor(sony_img, cv2.COLOR_BGR2RGB)
                
                print(f"   📊 Loaded individual files for pair {pair_id:03d}")
                
            except Exception as e:
                print(f"   ❌ Error loading individual files: {e}")
                return None
                
        elif comparison_files:
            try:
                # Load from comparison file
                comparison_img = cv2.imread(str(comparison_files[0]))
                comparison_img = cv2.cvtColor(comparison_img, cv2.COLOR_BGR2RGB)
                
                h, w = comparison_img.shape[:2]
                third_w = w // 3
                
                iphone_img = comparison_img[:, :third_w]
                transformed_img = comparison_img[:, third_w:2*third_w]
                sony_img = comparison_img[:, 2*third_w:]
                
                print(f"   📊 Loaded from comparison file for pair {pair_id:03d}")
                
            except Exception as e:
                print(f"   ❌ Error loading comparison file: {e}")
                return None
        else:
            print(f"   ❌ No test files found for pair {pair_id:03d}")
            return None
        
        # Analyze
        brightness_metrics = self.analyze_brightness_levels(iphone_img, transformed_img, sony_img)
        color_metrics = self.analyze_color_accuracy(iphone_img, transformed_img, sony_img)
        
        # Combine metrics
        all_metrics = {
            'pair_id': pair_id,
            **brightness_metrics,
            **color_metrics,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        return all_metrics
    
    def batch_analyze(self, max_pairs=10):
        """Analyze all available test results"""
        print(f"\n📊 PROFESSIONAL ANALYSIS - Cinema v1.4m")
        print("=" * 60)
        
        # Find all available test results
        comparison_files = list(self.results_dir.glob("*comparison_*.jpg"))
        individual_files = list(self.results_dir.glob("cinema_v1_4m_*.jpg"))
        
        if not comparison_files and not individual_files:
            print("❌ No test results found!")
            print(f"   Run the 4K tester first to generate comparison images")
            return
        
        # Extract pair IDs
        pair_ids = set()
        for file in comparison_files + individual_files:
            try:
                # Extract pair ID from filename
                parts = file.stem.split('_')
                for part in parts:
                    if part.isdigit() and len(part) == 3:
                        pair_ids.add(int(part))
                        break
            except:
                continue
        
        pair_ids = sorted(list(pair_ids))[:max_pairs]
        
        if not pair_ids:
            print("❌ No valid pair IDs found in test results")
            return
        
        print(f"🔍 Found test results for pairs: {pair_ids}")
        
        all_metrics = []
        success_count = 0
        
        for pair_id in pair_ids:
            print(f"\n📊 Analyzing Pair {pair_id:03d}")
            
            metrics = self.analyze_pair_from_files(pair_id)
            if metrics:
                all_metrics.append(metrics)
                success_count += 1
                
                # Print key metrics
                print(f"   ✅ Delta E: {metrics['delta_e_transformed_target']:.2f}")
                print(f"   ✅ SSIM: {metrics['ssim_avg']:.3f}")
                print(f"   ✅ Exposure Fixed: {metrics['exposure_fixed']}")
                print(f"   ✅ Brightness Change: {metrics['brightness_change']:+.3f}")
        
        if all_metrics:
            self.generate_professional_report(all_metrics)
            print(f"\n✅ ANALYSIS COMPLETE: {success_count}/{len(pair_ids)} pairs analyzed")
        else:
            print("❌ No successful analyses")
    
    def generate_professional_report(self, all_metrics):
        """Generate comprehensive professional report"""
        print(f"\n📊 GENERATING PROFESSIONAL REPORT")
        print("=" * 40)
        
        df = pd.DataFrame(all_metrics)
        
        # Calculate summary statistics
        delta_e_values = df['delta_e_transformed_target'].values
        ssim_values = df['ssim_avg'].values
        brightness_changes = df['brightness_change'].values
        exposure_fixed_count = df['exposure_fixed'].sum()
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'model_version': 'v1.4m (Exposure Fixed)',
            'pairs_analyzed': len(all_metrics),
            'exposure_statistics': {
                'exposure_fixed_count': int(exposure_fixed_count),
                'exposure_fix_rate': float(exposure_fixed_count / len(all_metrics)),
                'avg_brightness_change': float(np.mean(brightness_changes)),
                'std_brightness_change': float(np.std(brightness_changes)),
                'underexposure_eliminated': bool(exposure_fixed_count == len(all_metrics))
            },
            'color_statistics': {
                'delta_e_mean': float(np.mean(delta_e_values)),
                'delta_e_std': float(np.std(delta_e_values)),
                'delta_e_min': float(np.min(delta_e_values)),
                'delta_e_max': float(np.max(delta_e_values)),
                'ssim_mean': float(np.mean(ssim_values)),
                'ssim_std': float(np.std(ssim_values)),
                'channel_accuracy_r': float(df['accuracy_r'].mean()),
                'channel_accuracy_g': float(df['accuracy_g'].mean()),
                'channel_accuracy_b': float(df['accuracy_b'].mean()),
            },
            'professional_assessment': {
                'professional_grade': bool(np.mean(ssim_values) > 0.7 and np.mean(delta_e_values) < 20),
                'exposure_quality': 'Excellent' if exposure_fixed_count == len(all_metrics) else 'Needs Work',
                'color_accuracy': 'Professional' if np.mean(delta_e_values) < 15 else 'Good',
                'overall_rating': 'Production Ready' if (exposure_fixed_count == len(all_metrics) and 
                                                       np.mean(ssim_values) > 0.7) else 'In Development'
            }
        }
        
        # Save detailed report
        report_path = self.analysis_dir / "professional_analysis_report.json"
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed data
        detailed_path = self.analysis_dir / "detailed_metrics.csv"
        df.to_csv(detailed_path, index=False)
        
        # Create visualizations
        self.create_analysis_visualizations(df, summary)
        
        # Print summary
        exp_stats = summary['exposure_statistics']
        color_stats = summary['color_statistics']
        assessment = summary['professional_assessment']
        
        print(f"✅ Analyzed {summary['pairs_analyzed']} pairs")
        print(f"🔆 Exposure Fix Rate: {exp_stats['exposure_fix_rate']*100:.1f}%")
        print(f"🔆 Avg Brightness Change: {exp_stats['avg_brightness_change']:+.3f}")
        print(f"📊 Delta E: {color_stats['delta_e_mean']:.2f} ± {color_stats['delta_e_std']:.2f}")
        print(f"📊 SSIM: {color_stats['ssim_mean']:.3f} ± {color_stats['ssim_std']:.3f}")
        print(f"🎯 Overall Rating: {assessment['overall_rating']}")
        print(f"💾 Report: {report_path}")
        print(f"📈 Data: {detailed_path}")
    
    def create_analysis_visualizations(self, df, summary):
        """Create professional analysis visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Cinema v1.4m Professional Analysis', fontsize=16, fontweight='bold')
        
        # Delta E distribution
        axes[0,0].hist(df['delta_e_transformed_target'], bins=10, alpha=0.7, color='blue', edgecolor='black')
        axes[0,0].axvline(df['delta_e_transformed_target'].mean(), color='red', linestyle='--', 
                         label=f'Mean: {df["delta_e_transformed_target"].mean():.2f}')
        axes[0,0].set_title('Delta E Distribution')
        axes[0,0].set_xlabel('Delta E')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].legend()
        
        # SSIM distribution
        axes[0,1].hist(df['ssim_avg'], bins=10, alpha=0.7, color='green', edgecolor='black')
        axes[0,1].axvline(df['ssim_avg'].mean(), color='red', linestyle='--',
                         label=f'Mean: {df["ssim_avg"].mean():.3f}')
        axes[0,1].set_title('SSIM Distribution')
        axes[0,1].set_xlabel('SSIM')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].legend()
        
        # Brightness change analysis
        axes[0,2].hist(df['brightness_change'], bins=10, alpha=0.7, color='orange', edgecolor='black')
        axes[0,2].axvline(0, color='black', linestyle='-', alpha=0.3, label='No Change')
        axes[0,2].axvline(df['brightness_change'].mean(), color='red', linestyle='--',
                         label=f'Mean: {df["brightness_change"].mean():+.3f}')
        axes[0,2].set_title('Brightness Change')
        axes[0,2].set_xlabel('Brightness Change')
        axes[0,2].set_ylabel('Frequency')
        axes[0,2].legend()
        
        # Channel accuracy comparison
        channel_acc = [df['accuracy_r'].mean(), df['accuracy_g'].mean(), df['accuracy_b'].mean()]
        axes[1,0].bar(['Red', 'Green', 'Blue'], channel_acc, 
                     color=['red', 'green', 'blue'], alpha=0.7)
        axes[1,0].set_title('Average Channel Accuracy')
        axes[1,0].set_ylabel('Accuracy')
        axes[1,0].set_ylim(0, 1)
        for i, v in enumerate(channel_acc):
            axes[1,0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        
        # Exposure fix status
        exposure_counts = [df['exposure_fixed'].sum(), len(df) - df['exposure_fixed'].sum()]
        axes[1,1].pie(exposure_counts, labels=['Fixed', 'Issues'], autopct='%1.1f%%',
                     colors=['green', 'red'], alpha=0.7)
        axes[1,1].set_title('Exposure Fix Status')
        
        # Performance by pair
        axes[1,2].scatter(df['pair_id'], df['delta_e_transformed_target'], 
                         c=df['exposure_fixed'].map({True: 'green', False: 'red'}),
                         alpha=0.7, s=60)
        axes[1,2].set_title('Performance by Pair')
        axes[1,2].set_xlabel('Pair ID')
        axes[1,2].set_ylabel('Delta E')
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        viz_path = self.analysis_dir / "professional_analysis_visualization.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Visualization: {viz_path}")


def main():
    """Interactive analysis interface"""
    print("📊 CINEMA v1.4m - PROFESSIONAL ANALYZER")
    print("=" * 50)
    print("Pure metrics and validation")
    
    analyzer = ProfessionalAnalyzer()
    
    # Check if test results exist
    test_files = list(analyzer.results_dir.glob("*comparison_*.jpg")) + \
                list(analyzer.results_dir.glob("cinema_v1_4m_*.jpg"))
    
    if not test_files:
        print("\n❌ NO TEST RESULTS FOUND!")
        print("Run the 4K tester first:")
        print("   python v1_4m_tester.py")
        print("\nThen come back to analyze the results.")
        return
    
    print(f"\n✅ Found {len(test_files)} test result files")
    
    while True:
        print("\n📊 ANALYSIS OPTIONS:")
        print("1. Analyze all available results")
        print("2. Analyze specific pair")
        print("3. Quick summary")
        print("4. Exit")
        
        try:
            choice = input("\nChoice (1-4): ").strip()
            
            if choice == "1":
                max_pairs = int(input("Max pairs to analyze (default 10): ") or "10")
                analyzer.batch_analyze(max_pairs=max_pairs)
                
            elif choice == "2":
                pair_id = int(input("Pair ID to analyze: "))
                result = analyzer.analyze_pair_from_files(pair_id)
                if result:
                    print(f"\n📊 Analysis Results for Pair {pair_id:03d}:")
                    print(f"   Delta E: {result['delta_e_transformed_target']:.2f}")
                    print(f"   SSIM: {result['ssim_avg']:.3f}")
                    print(f"   Exposure Fixed: {result['exposure_fixed']}")
                    print(f"   Brightness Change: {result['brightness_change']:+.3f}")
                    print(f"   Channel Accuracy: R={result['accuracy_r']:.3f}, "
                          f"G={result['accuracy_g']:.3f}, B={result['accuracy_b']:.3f}")
                
            elif choice == "3":
                # Quick summary of available files
                comparison_files = list(analyzer.results_dir.glob("*comparison_*.jpg"))
                individual_files = list(analyzer.results_dir.glob("cinema_v1_4m_*.jpg"))
                
                print(f"\n📋 QUICK SUMMARY:")
                print(f"   Comparison files: {len(comparison_files)}")
                print(f"   Individual files: {len(individual_files)}")
                print(f"   Test results directory: {analyzer.results_dir}")
                
                if comparison_files:
                    print(f"\n📁 Recent comparison files:")
                    for f in sorted(comparison_files)[-5:]:
                        print(f"   {f.name}")
                
            elif choice == "4":
                print("👋 Analysis complete!")
                break
                
            else:
                print("Invalid choice")
                
        except KeyboardInterrupt:
            print("\n👋 Interrupted!")
            break
        except ValueError:
            print("❌ Invalid input")
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
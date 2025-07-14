#!/usr/bin/env python3
"""
Fixed Scientific ML Analysis Tool
Empirical validation of transformation results with quantitative metrics
"""

import cv2
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity
from scipy import stats

def main():
   """Main analysis pipeline"""
   results_dir = Path("data/results/perceptual_test")  # Updated for perceptual results
   
   if not results_dir.exists():
       print("❌ No transformation results found!")
       print("Run the perceptual training pipeline first.")
       return
   
   print("🔬 RUNNING SCIENTIFIC ANALYSIS")
   print("=" * 50)
   
   # Find transformation result images
   transform_files = list(results_dir.glob("perceptual_transform_*.jpg"))  # Updated pattern
   
   if not transform_files:
       print("❌ No transformation results found!")
       return
   
   print(f"📊 Found {len(transform_files)} transformation results")
   
   # Create analysis output directory
   analysis_output = results_dir / "scientific_analysis"
   analysis_output.mkdir(parents=True, exist_ok=True)
   
   results = []
   
   for transform_file in transform_files:
       print(f"\n🔍 Analyzing: {transform_file.name}")
       
       # Load transformation comparison image
       comparison_img = cv2.imread(str(transform_file))
       if comparison_img is None:
           print(f"❌ Could not load {transform_file}")
           continue
       
       h, w = comparison_img.shape[:2]
       
       # Split into original and transformed
       original = comparison_img[:, :w//2]
       transformed = comparison_img[:, w//2:]
       
       # Compute metrics
       metrics = analyze_transformation(original, transformed, transform_file.stem)
       results.append(metrics)
       
       print(f"  📊 SSIM: {metrics['ssim_mean']:.3f}")
       print(f"  📊 Delta E: {metrics['delta_e']:.2f}")
       print(f"  📊 Edge preservation: {metrics['edge_preservation']:.3f}")
       print(f"  🎨 Green bias: {metrics['g_mean_shift']:.2f}")  # Check green bias
   
   if results:
       # Generate report
       df = pd.DataFrame(results)
       generate_report(df, analysis_output)
       create_visualizations(df, analysis_output)
       
       print(f"\n✅ ANALYSIS COMPLETE!")
       print(f"📁 Results saved to: {analysis_output}")
   else:
       print("❌ No valid results to analyze!")

def analyze_transformation(original, transformed, sample_id):
   """Analyze a single transformation"""
   
   # Convert to different color spaces
   orig_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
   trans_lab = cv2.cvtColor(transformed, cv2.COLOR_BGR2LAB)
   
   orig_hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
   trans_hsv = cv2.cvtColor(transformed, cv2.COLOR_BGR2HSV)
   
   metrics = {'sample_id': sample_id}
   
   # 1. Pixel-level differences
   metrics['mse'] = mean_squared_error(original.flatten(), transformed.flatten())
   metrics['mae'] = np.mean(np.abs(original.astype(float) - transformed.astype(float)))
   
   # 2. Structural similarity (perceptual quality)
   ssim_scores = []
   for c in range(3):
       ssim = structural_similarity(original[:,:,c], transformed[:,:,c], data_range=255)
       ssim_scores.append(ssim)
   metrics['ssim_mean'] = np.mean(ssim_scores)
   metrics['ssim_std'] = np.std(ssim_scores)
   
   # 3. Color distribution analysis
   for i, color in enumerate(['B', 'G', 'R']):
       orig_hist = cv2.calcHist([original], [i], None, [256], [0, 256])
       trans_hist = cv2.calcHist([transformed], [i], None, [256], [0, 256])
       
       # Histogram correlation
       metrics[f'{color.lower()}_hist_corr'] = cv2.compareHist(orig_hist, trans_hist, cv2.HISTCMP_CORREL)
       
       # Mean shift
       metrics[f'{color.lower()}_mean_shift'] = np.mean(transformed[:,:,i]) - np.mean(original[:,:,i])
       
       # Std change (contrast change)
       metrics[f'{color.lower()}_std_ratio'] = np.std(transformed[:,:,i]) / (np.std(original[:,:,i]) + 1e-6)
   
   # 4. LAB color space analysis (perceptually uniform)
   metrics['delta_e'] = np.mean(np.sqrt(np.sum((orig_lab.astype(float) - trans_lab.astype(float))**2, axis=2)))
   
   # L*a*b* channel analysis
   metrics['lightness_shift'] = np.mean(trans_lab[:,:,0]) - np.mean(orig_lab[:,:,0])
   metrics['a_shift'] = np.mean(trans_lab[:,:,1]) - np.mean(orig_lab[:,:,1])  # Green-Red
   metrics['b_shift'] = np.mean(trans_lab[:,:,2]) - np.mean(orig_lab[:,:,2])  # Blue-Yellow
   
   # 5. HSV analysis
   metrics['hue_shift'] = np.mean(trans_hsv[:,:,0].astype(float)) - np.mean(orig_hsv[:,:,0].astype(float))
   metrics['saturation_ratio'] = np.mean(trans_hsv[:,:,1]) / (np.mean(orig_hsv[:,:,1]) + 1e-6)
   metrics['value_shift'] = np.mean(trans_hsv[:,:,2]) - np.mean(orig_hsv[:,:,2])
   
   # 6. Edge preservation
   orig_edges = cv2.Canny(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), 50, 150)
   trans_edges = cv2.Canny(cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY), 50, 150)
   
   edge_overlap = np.sum((orig_edges > 0) & (trans_edges > 0))
   edge_union = np.sum((orig_edges > 0) | (trans_edges > 0))
   metrics['edge_preservation'] = edge_overlap / (edge_union + 1e-6)
   
   # 7. Green tint analysis
   metrics['green_dominance'] = metrics['g_mean_shift'] - (metrics['r_mean_shift'] + metrics['b_mean_shift']) / 2
   metrics['color_balance_error'] = np.std([metrics['r_mean_shift'], metrics['g_mean_shift'], metrics['b_mean_shift']])
   
   return metrics

def generate_report(df, output_dir):
   """Generate statistical report"""
   print(f"\n📊 STATISTICAL ANALYSIS REPORT")
   print("=" * 50)
   
   # Transformation effectiveness
   print(f"\n🎯 TRANSFORMATION EFFECTIVENESS:")
   print(f"Mean pixel difference (MAE): {df['mae'].mean():.2f} ± {df['mae'].std():.2f}")
   print(f"Structural similarity (SSIM): {df['ssim_mean'].mean():.3f} ± {df['ssim_mean'].std():.3f}")
   print(f"Edge preservation: {df['edge_preservation'].mean():.3f} ± {df['edge_preservation'].std():.3f}")
   
   # Color science analysis
   print(f"\n🎨 COLOR SCIENCE ANALYSIS:")
   print(f"Delta E (perceptual color diff): {df['delta_e'].mean():.2f} ± {df['delta_e'].std():.2f}")
   print(f"Lightness shift: {df['lightness_shift'].mean():.2f} ± {df['lightness_shift'].std():.2f}")
   print(f"Saturation ratio: {df['saturation_ratio'].mean():.3f} ± {df['saturation_ratio'].std():.3f}")
   
   # Channel analysis
   print(f"\n📈 CHANNEL ANALYSIS:")
   for color in ['r', 'g', 'b']:
       mean_shift = df[f'{color}_mean_shift'].mean()
       std_ratio = df[f'{color}_std_ratio'].mean()
       print(f"{color.upper()} channel: shift={mean_shift:.2f}, contrast ratio={std_ratio:.3f}")
   
   # Green tint analysis
   print(f"\n🟢 GREEN TINT ANALYSIS:")
   print(f"Green dominance: {df['green_dominance'].mean():.2f} ± {df['green_dominance'].std():.2f}")
   print(f"Color balance error: {df['color_balance_error'].mean():.2f} ± {df['color_balance_error'].std():.2f}")
   
   if df['green_dominance'].mean() > 5:
       print("  ❌ SIGNIFICANT GREEN TINT DETECTED")
   elif df['green_dominance'].mean() > 2:
       print("  ⚠️ MILD GREEN TINT DETECTED")
   else:
       print("  ✅ COLOR BALANCE ACCEPTABLE")
   
   # Quality assessment
   print(f"\n✅ QUALITY ASSESSMENT:")
   high_quality = (df['ssim_mean'] > 0.9) & (df['edge_preservation'] > 0.7)
   medium_quality = (df['ssim_mean'] > 0.8) & (df['edge_preservation'] > 0.5)
   
   print(f"High quality transformations: {high_quality.sum()}/{len(df)} ({100*high_quality.mean():.1f}%)")
   print(f"Medium+ quality transformations: {medium_quality.sum()}/{len(df)} ({100*medium_quality.mean():.1f}%)")
   
   # Save report
   report = {
       'summary_stats': df.describe().to_dict(),
       'quality_metrics': {
           'high_quality_count': int(high_quality.sum()),
           'medium_quality_count': int(medium_quality.sum()),
           'total_samples': len(df),
           'mae_mean': float(df['mae'].mean()),
           'ssim_mean': float(df['ssim_mean'].mean()),
           'edge_preservation_mean': float(df['edge_preservation'].mean()),
           'green_dominance': float(df['green_dominance'].mean()),
           'color_balance_error': float(df['color_balance_error'].mean())
       },
       'timestamp': datetime.now().isoformat()
   }
   
   with open(output_dir / "statistical_report.json", 'w') as f:
       json.dump(report, f, indent=2)

def create_visualizations(df, output_dir):
   """Create visualizations"""
   print(f"\n📊 Creating visualizations...")
   
   # Set style
   plt.style.use('default')
   
   # Quality metrics distribution
   fig, axes = plt.subplots(2, 2, figsize=(12, 10))
   
   # MAE distribution
   axes[0,0].hist(df['mae'], bins=15, alpha=0.7, color='blue')
   axes[0,0].set_title('Mean Absolute Error Distribution')
   axes[0,0].set_xlabel('MAE')
   axes[0,0].set_ylabel('Frequency')
   
   # SSIM distribution
   axes[0,1].hist(df['ssim_mean'], bins=15, alpha=0.7, color='green')
   axes[0,1].set_title('Structural Similarity Distribution')
   axes[0,1].set_xlabel('SSIM')
   axes[0,1].set_ylabel('Frequency')
   
   # Delta E distribution
   axes[1,0].hist(df['delta_e'], bins=15, alpha=0.7, color='red')
   axes[1,0].set_title('Perceptual Color Difference (ΔE)')
   axes[1,0].set_xlabel('Delta E')
   axes[1,0].set_ylabel('Frequency')
   
   # Edge preservation
   axes[1,1].hist(df['edge_preservation'], bins=15, alpha=0.7, color='orange')
   axes[1,1].set_title('Edge Preservation')
   axes[1,1].set_xlabel('Edge Preservation Score')
   axes[1,1].set_ylabel('Frequency')
   
   plt.tight_layout()
   plt.savefig(output_dir / 'quality_distributions.png', dpi=300, bbox_inches='tight')
   plt.close()
   
   # Color analysis with green bias
   fig, axes = plt.subplots(1, 4, figsize=(20, 5))
   
   channels = ['r', 'g', 'b']
   colors = ['red', 'green', 'blue']
   
   for i, (channel, color) in enumerate(zip(channels, colors)):
       shifts = df[f'{channel}_mean_shift']
       axes[i].bar(range(len(shifts)), shifts, color=color, alpha=0.7)
       axes[i].set_title(f'{channel.upper()} Channel Shift')
       axes[i].set_xlabel('Sample')
       axes[i].set_ylabel('Mean Shift')
       axes[i].axhline(y=0, color='black', linestyle='--', alpha=0.5)
   
   # Green dominance plot
   green_dom = df['green_dominance']
   axes[3].bar(range(len(green_dom)), green_dom, color='lime', alpha=0.7)
   axes[3].set_title('Green Dominance')
   axes[3].set_xlabel('Sample')
   axes[3].set_ylabel('Green Bias')
   axes[3].axhline(y=0, color='black', linestyle='--', alpha=0.5)
   axes[3].axhline(y=5, color='red', linestyle='--', alpha=0.5, label='Problem threshold')
   
   plt.tight_layout()
   plt.savefig(output_dir / 'color_analysis.png', dpi=300, bbox_inches='tight')
   plt.close()
   
   print(f"✅ Visualizations saved to: {output_dir}")

if __name__ == "__main__":
   main()
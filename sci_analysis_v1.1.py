#!/usr/bin/env python3
"""
Scientific Analysis v2 - Enhanced for Hybrid Cinema Model v1.1
More comprehensive metrics including color vibrancy and local contrast
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
from skimage import color

def analyze_v1_1_results():
    """Analyze v1.1 model results with enhanced metrics"""
    results_dir = Path("data/results/hybrid_v1_1_test")
    
    if not results_dir.exists():
        print("❌ No v1.1 results found! Run the test first.")
        return
    
    print("🔬 SCIENTIFIC ANALYSIS V2")
    print("=" * 50)
    print("Enhanced metrics for color vibrancy and local contrast")
    
    transform_files = list(results_dir.glob("hybrid_v1_1_*.jpg"))
    
    if not transform_files:
        print("❌ No transformation results found!")
        return
    
    print(f"📊 Found {len(transform_files)} transformation results")
    
    analysis_output = results_dir / "scientific_analysis_v2"
    analysis_output.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for transform_file in transform_files:
        print(f"\n🔍 Analyzing: {transform_file.name}")
        
        comparison_img = cv2.imread(str(transform_file))
        if comparison_img is None:
            continue
        
        h, w = comparison_img.shape[:2]
        
        # Split into original and transformed
        original = comparison_img[:, :w//2]
        transformed = comparison_img[:, w//2:]
        
        # Enhanced metrics
        metrics = analyze_transformation_v2(original, transformed, transform_file.stem)
        results.append(metrics)
        
        # Quick summary
        print(f"  📊 SSIM: {metrics['ssim_mean']:.3f}")
        print(f"  🎨 Delta E: {metrics['delta_e']:.2f}")
        print(f"  🌈 Color Vibrancy: {metrics['vibrancy_ratio']:.3f}")
        print(f"  📈 Local Contrast: {metrics['local_contrast_ratio']:.3f}")
        print(f"  🎯 Overall Quality: {metrics['quality_score']:.3f}")
    
    if results:
        df = pd.DataFrame(results)
        generate_report_v2(df, analysis_output)
        create_visualizations_v2(df, analysis_output)
        
        print(f"\n✅ ANALYSIS COMPLETE!")
        print(f"📁 Results saved to: {analysis_output}")

def analyze_transformation_v2(original, transformed, sample_id):
    """Enhanced analysis with new metrics"""
    
    # Convert to different color spaces
    orig_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
    trans_lab = cv2.cvtColor(transformed, cv2.COLOR_BGR2LAB)
    
    orig_hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    trans_hsv = cv2.cvtColor(transformed, cv2.COLOR_BGR2HSV)
    
    metrics = {'sample_id': sample_id}
    
    # Standard metrics
    metrics['mse'] = mean_squared_error(original.flatten(), transformed.flatten())
    metrics['mae'] = np.mean(np.abs(original.astype(float) - transformed.astype(float)))
    
    # SSIM per channel and overall
    ssim_scores = []
    for c in range(3):
        ssim = structural_similarity(original[:,:,c], transformed[:,:,c], data_range=255)
        ssim_scores.append(ssim)
    metrics['ssim_mean'] = np.mean(ssim_scores)
    metrics['ssim_std'] = np.std(ssim_scores)
    
    # Color metrics
    for i, color in enumerate(['B', 'G', 'R']):
        orig_hist = cv2.calcHist([original], [i], None, [256], [0, 256])
        trans_hist = cv2.calcHist([transformed], [i], None, [256], [0, 256])
        
        metrics[f'{color.lower()}_hist_corr'] = cv2.compareHist(orig_hist, trans_hist, cv2.HISTCMP_CORREL)
        metrics[f'{color.lower()}_mean_shift'] = np.mean(transformed[:,:,i]) - np.mean(original[:,:,i])
        metrics[f'{color.lower()}_std_ratio'] = np.std(transformed[:,:,i]) / (np.std(original[:,:,i]) + 1e-6)
    
    # Delta E (CIE2000 for better accuracy)
    metrics['delta_e'] = np.mean(color.deltaE_ciede2000(orig_lab, trans_lab))
    
    # LAB analysis
    metrics['lightness_shift'] = np.mean(trans_lab[:,:,0]) - np.mean(orig_lab[:,:,0])
    metrics['a_shift'] = np.mean(trans_lab[:,:,1]) - np.mean(orig_lab[:,:,1])
    metrics['b_shift'] = np.mean(trans_lab[:,:,2]) - np.mean(orig_lab[:,:,2])
    
    # HSV analysis
    metrics['hue_shift'] = np.mean(trans_hsv[:,:,0].astype(float)) - np.mean(orig_hsv[:,:,0].astype(float))
    metrics['saturation_ratio'] = np.mean(trans_hsv[:,:,1]) / (np.mean(orig_hsv[:,:,1]) + 1e-6)
    metrics['value_shift'] = np.mean(trans_hsv[:,:,2]) - np.mean(orig_hsv[:,:,2])
    
    # NEW: Color Vibrancy (how much color "pops")
    orig_vibrancy = np.std(orig_hsv[:,:,1]) * np.mean(orig_hsv[:,:,1])
    trans_vibrancy = np.std(trans_hsv[:,:,1]) * np.mean(trans_hsv[:,:,1])
    metrics['vibrancy_ratio'] = trans_vibrancy / (orig_vibrancy + 1e-6)
    
    # NEW: Local Contrast (using Laplacian variance)
    orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    trans_gray = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)
    
    orig_laplacian = cv2.Laplacian(orig_gray, cv2.CV_64F)
    trans_laplacian = cv2.Laplacian(trans_gray, cv2.CV_64F)
    
    metrics['local_contrast_ratio'] = np.var(trans_laplacian) / (np.var(orig_laplacian) + 1e-6)
    
    # NEW: Highlight/Shadow Detail
    orig_highlights = original[orig_gray > 200]
    trans_highlights = transformed[trans_gray > 200]
    if len(orig_highlights) > 0 and len(trans_highlights) > 0:
        metrics['highlight_detail'] = np.std(trans_highlights) / (np.std(orig_highlights) + 1e-6)
    else:
        metrics['highlight_detail'] = 1.0
    
    orig_shadows = original[orig_gray < 50]
    trans_shadows = transformed[trans_gray < 50]
    if len(orig_shadows) > 0 and len(trans_shadows) > 0:
        metrics['shadow_detail'] = np.std(trans_shadows) / (np.std(orig_shadows) + 1e-6)
    else:
        metrics['shadow_detail'] = 1.0
    
    # NEW: Perceptual Sharpness (using gradient magnitude)
    sobelx = cv2.Sobel(orig_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(orig_gray, cv2.CV_64F, 0, 1, ksize=3)
    orig_sharpness = np.mean(np.sqrt(sobelx**2 + sobely**2))
    
    sobelx = cv2.Sobel(trans_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(trans_gray, cv2.CV_64F, 0, 1, ksize=3)
    trans_sharpness = np.mean(np.sqrt(sobelx**2 + sobely**2))
    
    metrics['sharpness_ratio'] = trans_sharpness / (orig_sharpness + 1e-6)
    
    # Edge preservation
    orig_edges = cv2.Canny(orig_gray, 50, 150)
    trans_edges = cv2.Canny(trans_gray, 50, 150)
    
    edge_overlap = np.sum((orig_edges > 0) & (trans_edges > 0))
    edge_union = np.sum((orig_edges > 0) | (trans_edges > 0))
    metrics['edge_preservation'] = edge_overlap / (edge_union + 1e-6)
    
    # Color balance analysis
    metrics['green_dominance'] = metrics['g_mean_shift'] - (metrics['r_mean_shift'] + metrics['b_mean_shift']) / 2
    metrics['color_balance_error'] = np.std([metrics['r_mean_shift'], metrics['g_mean_shift'], metrics['b_mean_shift']])
    
    # NEW: Overall Quality Score (weighted combination)
    quality_score = (
        0.25 * metrics['ssim_mean'] +                    # Structure
        0.20 * (1 - min(metrics['delta_e'] / 50, 1)) +  # Color accuracy (normalized)
        0.15 * metrics['edge_preservation'] +            # Detail preservation
        0.15 * min(metrics['vibrancy_ratio'], 2) / 2 +  # Vibrancy (capped)
        0.15 * min(metrics['local_contrast_ratio'], 2) / 2 + # Contrast (capped)
        0.10 * min(metrics['sharpness_ratio'], 2) / 2   # Sharpness (capped)
    )
    metrics['quality_score'] = quality_score
    
    return metrics

def generate_report_v2(df, output_dir):
    """Generate enhanced statistical report"""
    print(f"\n📊 ENHANCED STATISTICAL ANALYSIS REPORT")
    print("=" * 50)
    
    # Overall Performance
    print(f"\n🎯 OVERALL PERFORMANCE:")
    print(f"Quality Score: {df['quality_score'].mean():.3f} ± {df['quality_score'].std():.3f}")
    print(f"Best sample: {df.loc[df['quality_score'].idxmax(), 'sample_id']} (score: {df['quality_score'].max():.3f})")
    
    # Transformation effectiveness
    print(f"\n📈 TRANSFORMATION EFFECTIVENESS:")
    print(f"Structural similarity (SSIM): {df['ssim_mean'].mean():.3f} ± {df['ssim_mean'].std():.3f}")
    print(f"Edge preservation: {df['edge_preservation'].mean():.3f} ± {df['edge_preservation'].std():.3f}")
    print(f"Sharpness ratio: {df['sharpness_ratio'].mean():.3f} ± {df['sharpness_ratio'].std():.3f}")
    
    # Color science analysis
    print(f"\n🎨 COLOR SCIENCE ANALYSIS:")
    print(f"Delta E (CIE2000): {df['delta_e'].mean():.2f} ± {df['delta_e'].std():.2f}")
    print(f"Lightness shift: {df['lightness_shift'].mean():.2f} ± {df['lightness_shift'].std():.2f}")
    print(f"Saturation ratio: {df['saturation_ratio'].mean():.3f} ± {df['saturation_ratio'].std():.3f}")
    print(f"Color vibrancy ratio: {df['vibrancy_ratio'].mean():.3f} ± {df['vibrancy_ratio'].std():.3f}")
    
    # Channel analysis
    print(f"\n📊 CHANNEL ANALYSIS:")
    for color in ['r', 'g', 'b']:
        mean_shift = df[f'{color}_mean_shift'].mean()
        std_ratio = df[f'{color}_std_ratio'].mean()
        print(f"{color.upper()} channel: shift={mean_shift:.2f}, contrast={std_ratio:.3f}")
    
    # Green tint check
    print(f"\n🟢 COLOR BALANCE CHECK:")
    print(f"Green dominance: {df['green_dominance'].mean():.2f} ± {df['green_dominance'].std():.2f}")
    print(f"Color balance error: {df['color_balance_error'].mean():.2f}")
    
    if abs(df['green_dominance'].mean()) < 2:
        print("  ✅ COLOR BALANCE EXCELLENT")
    elif abs(df['green_dominance'].mean()) < 5:
        print("  ⚠️ SLIGHT COLOR IMBALANCE")
    else:
        print("  ❌ SIGNIFICANT COLOR IMBALANCE")
    
    # Contrast and detail
    print(f"\n💡 CONTRAST & DETAIL:")
    print(f"Local contrast ratio: {df['local_contrast_ratio'].mean():.3f} ± {df['local_contrast_ratio'].std():.3f}")
    print(f"Highlight detail: {df['highlight_detail'].mean():.3f}")
    print(f"Shadow detail: {df['shadow_detail'].mean():.3f}")
    
    # Quality thresholds
    print(f"\n✅ QUALITY ASSESSMENT:")
    excellent = df['quality_score'] > 0.8
    good = df['quality_score'] > 0.7
    acceptable = df['quality_score'] > 0.6
    
    print(f"Excellent quality: {excellent.sum()}/{len(df)} ({100*excellent.mean():.1f}%)")
    print(f"Good quality: {good.sum()}/{len(df)} ({100*good.mean():.1f}%)")
    print(f"Acceptable quality: {acceptable.sum()}/{len(df)} ({100*acceptable.mean():.1f}%)")
    
    # Save detailed report
    report = {
        'summary_stats': df.describe().to_dict(),
        'quality_metrics': {
            'overall_quality': float(df['quality_score'].mean()),
            'best_quality': float(df['quality_score'].max()),
            'worst_quality': float(df['quality_score'].min()),
            'consistency': float(df['quality_score'].std()),
            'excellent_count': int(excellent.sum()),
            'good_count': int(good.sum()),
            'total_samples': len(df)
        },
        'color_metrics': {
            'delta_e_mean': float(df['delta_e'].mean()),
            'vibrancy_ratio': float(df['vibrancy_ratio'].mean()),
            'saturation_ratio': float(df['saturation_ratio'].mean()),
            'green_dominance': float(df['green_dominance'].mean()),
            'color_balance_error': float(df['color_balance_error'].mean())
        },
        'detail_metrics': {
            'ssim_mean': float(df['ssim_mean'].mean()),
            'edge_preservation': float(df['edge_preservation'].mean()),
            'local_contrast': float(df['local_contrast_ratio'].mean()),
            'sharpness': float(df['sharpness_ratio'].mean())
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_dir / "analysis_report_v2.json", 'w') as f:
        json.dump(report, f, indent=2)

def create_visualizations_v2(df, output_dir):
    """Create enhanced visualizations"""
    print(f"\n📊 Creating enhanced visualizations...")
    
    plt.style.use('default')
    
    # 1. Quality Score Distribution
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.bar(range(len(df)), df['quality_score'], color='steelblue', alpha=0.8)
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Excellent (>0.8)')
    ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Good (>0.7)')
    ax.axhline(y=0.6, color='red', linestyle='--', alpha=0.5, label='Acceptable (>0.6)')
    
    ax.set_xlabel('Sample')
    ax.set_ylabel('Quality Score')
    ax.set_title('Overall Transformation Quality')
    ax.legend()
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'quality_scores.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Multi-metric comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # SSIM vs Edge Preservation
    axes[0,0].scatter(df['ssim_mean'], df['edge_preservation'], alpha=0.6, s=100)
    axes[0,0].set_xlabel('SSIM')
    axes[0,0].set_ylabel('Edge Preservation')
    axes[0,0].set_title('Structure Preservation')
    
    # Delta E vs Vibrancy
    axes[0,1].scatter(df['delta_e'], df['vibrancy_ratio'], alpha=0.6, s=100, c='green')
    axes[0,1].set_xlabel('Delta E')
    axes[0,1].set_ylabel('Vibrancy Ratio')
    axes[0,1].set_title('Color Transformation')
    
    # Local Contrast vs Sharpness
    axes[0,2].scatter(df['local_contrast_ratio'], df['sharpness_ratio'], alpha=0.6, s=100, c='orange')
    axes[0,2].set_xlabel('Local Contrast Ratio')
    axes[0,2].set_ylabel('Sharpness Ratio')
    axes[0,2].set_title('Detail Enhancement')
    
    # Channel shifts comparison
    channels = ['r', 'g', 'b']
    colors = ['red', 'green', 'blue']
    x = np.arange(len(df))
    width = 0.25
    
    for i, (channel, color) in enumerate(zip(channels, colors)):
        shifts = df[f'{channel}_mean_shift']
        axes[1,0].bar(x + i*width, shifts, width, label=channel.upper(), color=color, alpha=0.7)
    
    axes[1,0].set_xlabel('Sample')
    axes[1,0].set_ylabel('Mean Shift')
    axes[1,0].set_title('Color Channel Shifts')
    axes[1,0].legend()
    axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Saturation vs Lightness changes
    axes[1,1].scatter(df['lightness_shift'], df['saturation_ratio'], alpha=0.6, s=100, c='purple')
    axes[1,1].set_xlabel('Lightness Shift')
    axes[1,1].set_ylabel('Saturation Ratio')
    axes[1,1].set_title('LAB Color Changes')
    axes[1,1].axhline(y=1, color='black', linestyle='--', alpha=0.5)
    axes[1,1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # Quality components radar
    quality_components = ['ssim_mean', 'edge_preservation', 'vibrancy_ratio', 
                         'local_contrast_ratio', 'sharpness_ratio']
    
    radar_data = df[quality_components].mean()
    radar_data = radar_data / radar_data.max()  # Normalize
    
    angles = np.linspace(0, 2*np.pi, len(quality_components), endpoint=False)
    radar_data = np.concatenate((radar_data, [radar_data[0]]))  # Complete the circle
    angles = np.concatenate((angles, [angles[0]]))
    
    axes[1,2].plot(angles, radar_data, 'o-', linewidth=2, color='darkblue')
    axes[1,2].fill(angles, radar_data, alpha=0.25, color='darkblue')
    axes[1,2].set_xticks(angles[:-1])
    axes[1,2].set_xticklabels(['SSIM', 'Edge', 'Vibrancy', 'Contrast', 'Sharp'], size=8)
    axes[1,2].set_ylim(0, 1)
    axes[1,2].set_title('Average Quality Components')
    axes[1,2].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'detailed_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Before/After histogram comparison (for first sample)
    # This would require access to actual image data, so we'll skip for now
    
    print(f"✅ Visualizations saved to: {output_dir}")

if __name__ == "__main__":
    analyze_v1_1_results()
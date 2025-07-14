#!/usr/bin/env python3
"""
Scientific Analysis v1.1 Fixed - For Stable Model Results
Complete working version with all syntax errors fixed
"""

import cv2
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity
from scipy import stats

def analyze_stable_results():
    """Analyze stable model results"""
    results_dir = Path("data/results/hybrid_v1_1_stable")
    
    if not results_dir.exists():
        print("❌ No stable model results found! Run the test first.")
        return
    
    print("🔬 SCIENTIFIC ANALYSIS - STABLE MODEL")
    print("=" * 50)
    
    transform_files = list(results_dir.glob("stable_*.jpg"))
    
    if not transform_files:
        print("❌ No transformation results found!")
        return
    
    print(f"📊 Found {len(transform_files)} transformation results")
    
    analysis_output = results_dir / "scientific_analysis"
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
        
        # Analyze
        metrics = analyze_transformation(original, transformed, transform_file.stem)
        results.append(metrics)
        
        # Summary
        print(f"  📊 SSIM: {metrics['ssim_mean']:.3f}")
        print(f"  🎨 Delta E: {metrics['delta_e']:.2f}")
        print(f"  📈 Quality Score: {metrics['quality_score']:.3f}")
    
    if results:
        df = pd.DataFrame(results)
        generate_report(df, analysis_output)
        create_visualizations(df, analysis_output)
        
        print(f"\n✅ ANALYSIS COMPLETE!")
        print(f"📁 Results saved to: {analysis_output}")

def analyze_transformation(original, transformed, sample_id):
    """Analyze a single transformation"""
    
    # Convert to different color spaces
    orig_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
    trans_lab = cv2.cvtColor(transformed, cv2.COLOR_BGR2LAB)
    
    orig_hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    trans_hsv = cv2.cvtColor(transformed, cv2.COLOR_BGR2HSV)
    
    orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    trans_gray = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)
    
    metrics = {'sample_id': sample_id}
    
    # Basic metrics
    metrics['mse'] = mean_squared_error(original.flatten(), transformed.flatten())
    metrics['mae'] = np.mean(np.abs(original.astype(float) - transformed.astype(float)))
    
    # SSIM
    ssim_scores = []
    for c in range(3):
        ssim = structural_similarity(original[:,:,c], transformed[:,:,c], data_range=255)
        ssim_scores.append(ssim)
    metrics['ssim_mean'] = np.mean(ssim_scores)
    metrics['ssim_std'] = np.std(ssim_scores)
    
    # Color channel analysis
    for i, color in enumerate(['B', 'G', 'R']):
        orig_hist = cv2.calcHist([original], [i], None, [256], [0, 256])
        trans_hist = cv2.calcHist([transformed], [i], None, [256], [0, 256])
        
        metrics[f'{color.lower()}_hist_corr'] = cv2.compareHist(orig_hist, trans_hist, cv2.HISTCMP_CORREL)
        metrics[f'{color.lower()}_mean_shift'] = np.mean(transformed[:,:,i]) - np.mean(original[:,:,i])
        metrics[f'{color.lower()}_std_ratio'] = np.std(transformed[:,:,i]) / (np.std(original[:,:,i]) + 1e-6)
    
    # Delta E (perceptual color difference)
    metrics['delta_e'] = np.mean(np.sqrt(np.sum((orig_lab.astype(float) - trans_lab.astype(float))**2, axis=2)))
    
    # LAB channel shifts
    metrics['lightness_shift'] = np.mean(trans_lab[:,:,0]) - np.mean(orig_lab[:,:,0])
    metrics['a_shift'] = np.mean(trans_lab[:,:,1]) - np.mean(orig_lab[:,:,1])
    metrics['b_shift'] = np.mean(trans_lab[:,:,2]) - np.mean(orig_lab[:,:,2])
    
    # HSV analysis
    metrics['hue_shift'] = np.mean(trans_hsv[:,:,0].astype(float)) - np.mean(orig_hsv[:,:,0].astype(float))
    metrics['saturation_ratio'] = np.mean(trans_hsv[:,:,1]) / (np.mean(orig_hsv[:,:,1]) + 1e-6)
    metrics['value_shift'] = np.mean(trans_hsv[:,:,2]) - np.mean(orig_hsv[:,:,2])
    
    # Color vibrancy
    orig_vibrancy = np.std(orig_hsv[:,:,1]) * np.mean(orig_hsv[:,:,1])
    trans_vibrancy = np.std(trans_hsv[:,:,1]) * np.mean(trans_hsv[:,:,1])
    metrics['vibrancy_ratio'] = trans_vibrancy / (orig_vibrancy + 1e-6)
    
    # Local contrast (Laplacian variance)
    orig_laplacian = cv2.Laplacian(orig_gray, cv2.CV_64F)
    trans_laplacian = cv2.Laplacian(trans_gray, cv2.CV_64F)
    metrics['local_contrast_ratio'] = np.var(trans_laplacian) / (np.var(orig_laplacian) + 1e-6)
    
    # Highlight/Shadow detail
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
    
    # Edge preservation
    orig_edges = cv2.Canny(orig_gray, 50, 150)
    trans_edges = cv2.Canny(trans_gray, 50, 150)
    
    edge_overlap = np.sum((orig_edges > 0) & (trans_edges > 0))
    edge_union = np.sum((orig_edges > 0) | (trans_edges > 0))
    metrics['edge_preservation'] = edge_overlap / (edge_union + 1e-6)
    
    # Color balance check
    metrics['green_dominance'] = metrics['g_mean_shift'] - (metrics['r_mean_shift'] + metrics['b_mean_shift']) / 2
    metrics['color_balance_error'] = np.std([metrics['r_mean_shift'], metrics['g_mean_shift'], metrics['b_mean_shift']])
    
    # Overall quality score
    quality_score = (
        0.3 * metrics['ssim_mean'] +
        0.2 * (1 - min(metrics['delta_e'] / 50, 1)) +
        0.2 * metrics['edge_preservation'] +
        0.15 * min(metrics['vibrancy_ratio'], 2) / 2 +
        0.15 * min(metrics['local_contrast_ratio'], 2) / 2
    )
    metrics['quality_score'] = quality_score
    
    return metrics

def generate_report(df, output_dir):
    """Generate analysis report"""
    print(f"\n📊 ANALYSIS REPORT")
    print("=" * 50)
    
    # Overall quality
    print(f"\n🎯 OVERALL QUALITY:")
    print(f"Average Quality Score: {df['quality_score'].mean():.3f} ± {df['quality_score'].std():.3f}")
    print(f"Best: {df['quality_score'].max():.3f}, Worst: {df['quality_score'].min():.3f}")
    
    # Structure preservation
    print(f"\n📈 STRUCTURE PRESERVATION:")
    print(f"SSIM: {df['ssim_mean'].mean():.3f} ± {df['ssim_mean'].std():.3f}")
    print(f"Edge preservation: {df['edge_preservation'].mean():.3f} ± {df['edge_preservation'].std():.3f}")
    
    # Color analysis
    print(f"\n🎨 COLOR ANALYSIS:")
    print(f"Delta E: {df['delta_e'].mean():.2f} ± {df['delta_e'].std():.2f}")
    print(f"Saturation ratio: {df['saturation_ratio'].mean():.3f}")
    print(f"Vibrancy ratio: {df['vibrancy_ratio'].mean():.3f}")
    
    # Channel shifts
    print(f"\n📊 CHANNEL SHIFTS:")
    for color in ['r', 'g', 'b']:
        shift = df[f'{color}_mean_shift'].mean()
        print(f"{color.upper()}: {shift:+.2f}")
    
    # Color balance
    print(f"\n🟢 COLOR BALANCE:")
    green_dom = df['green_dominance'].mean()
    print(f"Green dominance: {green_dom:.2f}")
    if abs(green_dom) < 2:
        print("  ✅ Excellent color balance")
    elif abs(green_dom) < 5:
        print("  ⚠️ Slight color imbalance")
    else:
        print("  ❌ Significant color imbalance")
    
    # Contrast and detail
    print(f"\n💡 CONTRAST & DETAIL:")
    print(f"Local contrast ratio: {df['local_contrast_ratio'].mean():.3f}")
    print(f"Highlight detail: {df['highlight_detail'].mean():.3f}")
    print(f"Shadow detail: {df['shadow_detail'].mean():.3f}")
    
    # Save report
    report = {
        'summary': {
            'quality_score_mean': float(df['quality_score'].mean()),
            'quality_score_std': float(df['quality_score'].std()),
            'ssim_mean': float(df['ssim_mean'].mean()),
            'delta_e_mean': float(df['delta_e'].mean()),
            'vibrancy_ratio': float(df['vibrancy_ratio'].mean()),
            'green_dominance': float(df['green_dominance'].mean()),
            'samples': len(df)
        },
        'detailed_stats': df.describe().to_dict(),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_dir / "analysis_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save CSV for further analysis
    df.to_csv(output_dir / "detailed_metrics.csv", index=False)

def create_visualizations(df, output_dir):
    """Create visualization plots"""
    print(f"\n📊 Creating visualizations...")
    
    plt.style.use('default')
    
    # 1. Quality metrics overview
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Quality scores
    axes[0,0].bar(range(len(df)), df['quality_score'], color='steelblue', alpha=0.7)
    axes[0,0].axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Good (>0.7)')
    axes[0,0].set_xlabel('Sample')
    axes[0,0].set_ylabel('Quality Score')
    axes[0,0].set_title('Overall Quality Scores')
    axes[0,0].legend()
    
    # SSIM distribution
    axes[0,1].hist(df['ssim_mean'], bins=10, alpha=0.7, color='green')
    axes[0,1].set_xlabel('SSIM')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('Structural Similarity Distribution')
    
    # Delta E distribution
    axes[1,0].hist(df['delta_e'], bins=10, alpha=0.7, color='red')
    axes[1,0].set_xlabel('Delta E')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('Perceptual Color Difference')
    
    # Vibrancy vs Contrast
    axes[1,1].scatter(df['vibrancy_ratio'], df['local_contrast_ratio'], alpha=0.6, s=100)
    axes[1,1].set_xlabel('Vibrancy Ratio')
    axes[1,1].set_ylabel('Local Contrast Ratio')
    axes[1,1].set_title('Vibrancy vs Contrast')
    axes[1,1].axhline(y=1, color='black', linestyle='--', alpha=0.3)
    axes[1,1].axvline(x=1, color='black', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'quality_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Color channel analysis
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Channel shifts
    channels = ['r', 'g', 'b']
    colors = ['red', 'green', 'blue']
    x = np.arange(len(df))
    width = 0.25
    
    for i, (channel, color) in enumerate(zip(channels, colors)):
        shifts = df[f'{channel}_mean_shift']
        axes[0].bar(x + i*width, shifts, width, label=channel.upper(), color=color, alpha=0.7)
    
    axes[0].set_xlabel('Sample')
    axes[0].set_ylabel('Mean Shift')
    axes[0].set_title('Color Channel Shifts')
    axes[0].legend()
    axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Color balance
    axes[1].bar(range(len(df)), df['green_dominance'], color='green', alpha=0.7)
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1].axhline(y=2, color='orange', linestyle='--', alpha=0.5, label='Threshold')
    axes[1].axhline(y=-2, color='orange', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Sample')
    axes[1].set_ylabel('Green Dominance')
    axes[1].set_title('Color Balance Check')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'color_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualizations saved to: {output_dir}")

if __name__ == "__main__":
    analyze_stable_results()
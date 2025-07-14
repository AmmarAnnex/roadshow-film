#!/usr/bin/env python3
"""
Quick 4K Analysis - Fixed Syntax Errors
Simple analysis tool for 4K cinema model results
"""

import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import pandas as pd
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

def analyze_4k_results():
    """Quick analysis of 4K results"""
    results_dir = Path("data/results/cinema_v1_3b_4k_test")
    
    if not results_dir.exists():
        print("❌ No 4K results found!")
        return
    
    print("🔬 QUICK 4K ANALYSIS")
    print("=" * 50)
    
    # Find comparison files
    comparison_files = list(results_dir.glob("*_comparison.jpg"))
    
    if not comparison_files:
        print("❌ No comparison files found!")
        return
    
    print(f"📊 Found {len(comparison_files)} results")
    
    results = []
    
    for comp_file in comparison_files:
        print(f"\n🔍 Analyzing: {comp_file.name}")
        
        # Load and split image
        img = cv2.imread(str(comp_file))
        if img is None:
            continue
            
        h, w = img.shape[:2]
        original = img[:, :w//2]
        transformed = img[:, w//2:]
        
        # Basic metrics
        mse = mean_squared_error(original.flatten(), transformed.flatten())
        mae = np.mean(np.abs(original.astype(float) - transformed.astype(float)))
        psnr = peak_signal_noise_ratio(original, transformed, data_range=255)
        
        # SSIM per channel
        ssim_scores = []
        for c in range(3):
            ssim = structural_similarity(original[:,:,c], transformed[:,:,c], data_range=255)
            ssim_scores.append(ssim)
        ssim_mean = np.mean(ssim_scores)
        
        # Color difference (simple)
        orig_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
        trans_lab = cv2.cvtColor(transformed, cv2.COLOR_BGR2LAB)
        delta_e = np.mean(np.sqrt(np.sum((orig_lab.astype(float) - trans_lab.astype(float))**2, axis=2)))
        
        # Color shifts
        orig_hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
        trans_hsv = cv2.cvtColor(transformed, cv2.COLOR_BGR2HSV)
        
        sat_ratio = np.mean(trans_hsv[:,:,1]) / (np.mean(orig_hsv[:,:,1]) + 1e-6)
        hue_shift = np.mean(trans_hsv[:,:,0].astype(float)) - np.mean(orig_hsv[:,:,0].astype(float))
        
        result = {
            'sample': comp_file.stem,
            'resolution': f"{w//2}x{h}",
            'mse': mse,
            'mae': mae,
            'psnr': psnr,
            'ssim': ssim_mean,
            'delta_e': delta_e,
            'saturation_ratio': sat_ratio,
            'hue_shift': hue_shift
        }
        
        results.append(result)
        
        print(f"  📊 SSIM: {ssim_mean:.3f}")
        print(f"  🎨 Delta E: {delta_e:.2f}")
        print(f"  📈 PSNR: {psnr:.1f} dB")
        print(f"  🌈 Saturation: {sat_ratio:.3f}")
    
    if not results:
        print("❌ No valid results to analyze!")
        return
    
    # Summary analysis
    df = pd.DataFrame(results)
    
    print(f"\n📊 SUMMARY ANALYSIS")
    print("=" * 50)
    
    print(f"🎯 QUALITY METRICS:")
    print(f"Average SSIM: {df['ssim'].mean():.3f} ± {df['ssim'].std():.3f}")
    print(f"Average PSNR: {df['psnr'].mean():.1f} ± {df['psnr'].std():.1f} dB")
    print(f"Average Delta E: {df['delta_e'].mean():.2f} ± {df['delta_e'].std():.2f}")
    
    print(f"\n🎨 COLOR CHARACTERISTICS:")
    print(f"Saturation Enhancement: {((df['saturation_ratio'].mean() - 1) * 100):+.1f}%")
    print(f"Hue Shift: {df['hue_shift'].mean():+.1f}°")
    
    print(f"\n📈 QUALITY ASSESSMENT:")
    excellent = (df['ssim'] >= 0.9).sum()
    good = ((df['ssim'] >= 0.8) & (df['ssim'] < 0.9)).sum()
    needs_work = (df['ssim'] < 0.8).sum()
    
    print(f"Excellent (SSIM ≥0.9): {excellent}/{len(df)} ({100*excellent/len(df):.1f}%)")
    print(f"Good (SSIM 0.8-0.9): {good}/{len(df)} ({100*good/len(df):.1f}%)")
    print(f"Needs Work (SSIM <0.8): {needs_work}/{len(df)} ({100*needs_work/len(df):.1f}%)")
    
    # Save results
    analysis_dir = results_dir / "quick_analysis"
    analysis_dir.mkdir(exist_ok=True)
    
    df.to_csv(analysis_dir / "4k_results.csv", index=False)
    
    # Create simple report
    report = {
        'analysis_date': datetime.now().isoformat(),
        'model_version': 'v1.3b_4k',
        'samples_analyzed': len(df),
        'average_metrics': {
            'ssim': float(df['ssim'].mean()),
            'psnr': float(df['psnr'].mean()),
            'delta_e': float(df['delta_e'].mean()),
            'saturation_enhancement_percent': float((df['saturation_ratio'].mean() - 1) * 100)
        },
        'quality_distribution': {
            'excellent': int(excellent),
            'good': int(good),
            'needs_work': int(needs_work)
        }
    }
    
    with open(analysis_dir / "quick_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Quick analysis complete!")
    print(f"📁 Results saved to: {analysis_dir}")
    print(f"📊 CSV: 4k_results.csv")
    print(f"📋 Report: quick_report.json")

if __name__ == "__main__":
    analyze_4k_results()
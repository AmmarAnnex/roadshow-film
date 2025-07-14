#!/usr/bin/env python3
"""
Scientific ML Analysis Tool
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
from sklearn.metrics import mean_squared_error, structural_similarity
from scipy import stats
import rawpy

class ScientificAnalyzer:
    """Scientific analysis of ML transformation results"""
    
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.analysis_output = self.results_dir / "scientific_analysis"
        self.analysis_output.mkdir(parents=True, exist_ok=True)
        
        # Load training metadata
        self.training_metadata = self.load_training_metadata()
        
    def load_training_metadata(self):
        """Load original training metadata for comparison"""
        metadata_file = Path("data/results/simple_depth_analysis/depth_metadata.json")
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        return []
    
    def load_and_analyze_transformations(self):
        """Load transformation results and analyze scientifically"""
        print("🔬 SCIENTIFIC ANALYSIS OF TRANSFORMATIONS")
        print("=" * 50)
        
        # Find transformation result images
        transform_files = list(self.results_dir.glob("color_science_*.jpg"))
        
        if not transform_files:
            print("❌ No transformation results found!")
            return None
        
        print(f"📊 Found {len(transform_files)} transformation results")
        
        results = []
        
        for transform_file in transform_files:
            result = self.analyze_single_transformation(transform_file)
            if result:
                results.append(result)
        
        return results
    
    def analyze_single_transformation(self, transform_file: Path):
        """Analyze a single transformation scientifically"""
        # Parse filename to get source info
        filename = transform_file.stem
        parts = filename.split('_')
        if len(parts) >= 3:
            iphone_id = parts[2]  # e.g., "iphone_001"
        else:
            return None
        
        print(f"\n🔍 Analyzing: {iphone_id}")
        
        # Load transformation comparison image
        comparison_img = cv2.imread(str(transform_file))
        if comparison_img is None:
            print(f"❌ Could not load {transform_file}")
            return None
        
        h, w = comparison_img.shape[:2]
        
        # Split into original and transformed
        original = comparison_img[:, :w//2]
        transformed = comparison_img[:, w//2:]
        
        # Find corresponding training metadata
        training_data = self.find_training_metadata(iphone_id)
        
        # Compute comprehensive metrics
        metrics = self.compute_transformation_metrics(original, transformed, training_data)
        metrics['sample_id'] = iphone_id
        metrics['filename'] = str(transform_file)
        
        return metrics
    
    def find_training_metadata(self, iphone_id):
        """Find corresponding training metadata"""
        iphone_num = iphone_id.split('_')[-1]
        
        for metadata in self.training_metadata:
            if f"iphone_{iphone_num}.dng" in metadata.get('iphone_file', ''):
                return metadata
        
        return None
    
    def compute_transformation_metrics(self, original, transformed, training_data):
        """Compute comprehensive transformation metrics"""
        
        # Convert to different color spaces for analysis
        orig_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
        trans_lab = cv2.cvtColor(transformed, cv2.COLOR_BGR2LAB)
        
        orig_hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
        trans_hsv = cv2.cvtColor(transformed, cv2.COLOR_BGR2HSV)
        
        metrics = {}
        
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
        
        # 5. HSV analysis (hue, saturation, value)
        metrics['hue_shift'] = np.mean(trans_hsv[:,:,0].astype(float)) - np.mean(orig_hsv[:,:,0].astype(float))
        metrics['saturation_ratio'] = np.mean(trans_hsv[:,:,1]) / (np.mean(orig_hsv[:,:,1]) + 1e-6)
        metrics['value_shift'] = np.mean(trans_hsv[:,:,2]) - np.mean(orig_hsv[:,:,2])
        
        # 6. Edge preservation (detail retention)
        orig_edges = cv2.Canny(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), 50, 150)
        trans_edges = cv2.Canny(cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY), 50, 150)
        
        edge_overlap = np.sum((orig_edges > 0) & (trans_edges > 0))
        edge_union = np.sum((orig_edges > 0) | (trans_edges > 0))
        metrics['edge_preservation'] = edge_overlap / (edge_union + 1e-6)
        
        # 7. Training data correlation (if available)
        if training_data:
            analysis = training_data.get('analysis', {})
            metrics['expected_color_diff'] = analysis.get('color_difference', 0)
            metrics['expected_depth_diff'] = analysis.get('depth_difference', 0)
            metrics['expected_sharpness_diff'] = analysis.get('sharpness_difference', 0)
            
            # Camera settings
            iphone_meta = training_data.get('iphone_metadata', {})
            sony_meta = training_data.get('sony_metadata', {})
            
            metrics['iphone_iso'] = iphone_meta.get('iso', 'Unknown')
            metrics['sony_iso'] = sony_meta.get('iso', 'Unknown')
            metrics['aperture_diff'] = iphone_meta.get('aperture', 'Unknown')
        
        return metrics
    
    def generate_statistical_report(self, results):
        """Generate comprehensive statistical analysis"""
        if not results:
            print("❌ No results to analyze!")
            return
        
        df = pd.DataFrame(results)
        
        print(f"\n📊 STATISTICAL ANALYSIS REPORT")
        print("=" * 50)
        
        # 1. Transformation effectiveness
        print(f"\n🎯 TRANSFORMATION EFFECTIVENESS:")
        print(f"Mean pixel difference (MAE): {df['mae'].mean():.2f} ± {df['mae'].std():.2f}")
        print(f"Structural similarity (SSIM): {df['ssim_mean'].mean():.3f} ± {df['ssim_mean'].std():.3f}")
        print(f"Edge preservation: {df['edge_preservation'].mean():.3f} ± {df['edge_preservation'].std():.3f}")
        
        # 2. Color science analysis
        print(f"\n🎨 COLOR SCIENCE ANALYSIS:")
        print(f"Delta E (perceptual color diff): {df['delta_e'].mean():.2f} ± {df['delta_e'].std():.2f}")
        print(f"Lightness shift: {df['lightness_shift'].mean():.2f} ± {df['lightness_shift'].std():.2f}")
        print(f"Saturation ratio: {df['saturation_ratio'].mean():.3f} ± {df['saturation_ratio'].std():.3f}")
        
        # 3. Channel-specific analysis
        print(f"\n📈 CHANNEL ANALYSIS:")
        for color in ['r', 'g', 'b']:
            mean_shift = df[f'{color}_mean_shift'].mean()
            std_ratio = df[f'{color}_std_ratio'].mean()
            print(f"{color.upper()} channel: shift={mean_shift:.2f}, contrast ratio={std_ratio:.3f}")
        
        # 4. Consistency analysis
        print(f"\n📏 CONSISTENCY ANALYSIS:")
        mae_cv = df['mae'].std() / df['mae'].mean()  # Coefficient of variation
        ssim_cv = df['ssim_mean'].std() / df['ssim_mean'].mean()
        print(f"MAE consistency (lower=better): {mae_cv:.3f}")
        print(f"SSIM consistency (lower=better): {ssim_cv:.3f}")
        
        # 5. Quality thresholds
        print(f"\n✅ QUALITY ASSESSMENT:")
        high_quality = (df['ssim_mean'] > 0.9) & (df['edge_preservation'] > 0.7)
        medium_quality = (df['ssim_mean'] > 0.8) & (df['edge_preservation'] > 0.5)
        
        print(f"High quality transformations: {high_quality.sum()}/{len(df)} ({100*high_quality.mean():.1f}%)")
        print(f"Medium+ quality transformations: {medium_quality.sum()}/{len(df)} ({100*medium_quality.mean():.1f}%)")
        
        # Save detailed report
        report_path = self.analysis_output / "statistical_report.json"
        with open(report_path, 'w') as f:
            json.dump({
                'summary_stats': df.describe().to_dict(),
                'quality_metrics': {
                    'high_quality_count': int(high_quality.sum()),
                    'medium_quality_count': int(medium_quality.sum()),
                    'total_samples': len(df),
                    'mae_mean': float(df['mae'].mean()),
                    'ssim_mean': float(df['ssim_mean'].mean()),
                    'edge_preservation_mean': float(df['edge_preservation'].mean())
                },
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        return df
    
    def create_visualizations(self, df):
        """Create scientific visualizations"""
        print(f"\n📊 CREATING VISUALIZATIONS...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Quality metrics distribution
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
        plt.savefig(self.analysis_output / 'quality_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Color analysis
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # RGB channel shifts
        channels = ['r', 'g', 'b']
        colors = ['red', 'green', 'blue']
        
        for i, (channel, color) in enumerate(zip(channels, colors)):
            shifts = df[f'{channel}_mean_shift']
            axes[i].bar(range(len(shifts)), shifts, color=color, alpha=0.7)
            axes[i].set_title(f'{channel.upper()} Channel Shift')
            axes[i].set_xlabel('Sample')
            axes[i].set_ylabel('Mean Shift')
            axes[i].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(self.analysis_output / 'color_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Quality correlation matrix
        quality_cols = ['mae', 'ssim_mean', 'delta_e', 'edge_preservation', 
                       'lightness_shift', 'saturation_ratio']
        
        if all(col in df.columns for col in quality_cols):
            correlation_matrix = df[quality_cols].corr()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Quality Metrics Correlation Matrix')
            plt.tight_layout()
            plt.savefig(self.analysis_output / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✅ Visualizations saved to: {self.analysis_output}")
    
    def run_complete_analysis(self):
        """Run complete scientific analysis"""
        print("🔬 RUNNING COMPLETE SCIENTIFIC ANALYSIS")
        print("=" * 60)
        
        # Load and analyze transformations
        results = self.load_and_analyze_transformations()
        
        if not results:
            print("❌ No valid results found!")
            return
        
        # Generate statistical report
        df = self.generate_statistical_report(results)
        
        # Create visualizations
        if df is not None:
            self.create_visualizations(df)
        
        # Save raw data
        raw_data_path = self.analysis_output / "raw_analysis_data.json"
        with open(raw_data_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ COMPLETE ANALYSIS FINISHED")
        print(f"📁 Results saved to: {self.analysis_output}")
        print(f"📊 Key files:")
        print(f"   - statistical_report.json (summary statistics)")
        print(f"   - quality_distributions.png (quality metrics)")
        print(f"   - color_analysis.png (color channel analysis)")
        print(f"   - correlation_matrix.png (metric correlations)")
        print(f"   - raw_analysis_data.json (complete raw data)")
        
        return results

def main():
    """Main analysis pipeline"""
    results_dir = Path("data/results/color_science_test")
    
    if not results_dir.exists():
        print("❌ No transformation results found!")
        print("Run the transformation pipeline first.")
        return
    
    analyzer = ScientificAnalyzer(results_dir)
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
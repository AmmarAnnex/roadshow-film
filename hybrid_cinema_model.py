#!/usr/bin/env python3
"""
Hybrid Classical + ML Cinema Model
80% classical color science + 20% ML enhancement
"""

import cv2
import numpy as np
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import rawpy
from datetime import datetime

class SmallMLEnhancement(nn.Module):
    """Small ML model for fine-tuning classical results"""
    
    def __init__(self):
        super(SmallMLEnhancement, self).__init__()
        
        # Very small network - just refinement
        self.enhancement = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Tanh()
        )
        
        # Tiny residual strength
        self.residual_strength = nn.Parameter(torch.tensor(0.05))
    
    def forward(self, x):
        # Small learned enhancement
        enhancement = self.enhancement(x)
        
        # Very small residual correction
        output = x + self.residual_strength * enhancement
        
        return torch.clamp(output, 0, 1)

class HybridCinemaTransform(nn.Module):
    """Hybrid classical + ML transformation"""
    
    def __init__(self):
        super(HybridCinemaTransform, self).__init__()
        
        # Classical color science parameters (learnable but constrained)
        self.color_matrix = nn.Parameter(torch.eye(3))
        self.color_bias = nn.Parameter(torch.zeros(3))
        
        # Cinema tone curve parameters
        self.shadows = nn.Parameter(torch.tensor(0.0))
        self.mids = nn.Parameter(torch.tensor(1.0))
        self.highlights = nn.Parameter(torch.tensor(1.0))
        
        # Color grading controls
        self.contrast = nn.Parameter(torch.tensor(1.0))
        self.saturation = nn.Parameter(torch.tensor(1.0))
        self.warmth = nn.Parameter(torch.tensor(0.0))  # Color temperature
        
        # Small ML enhancement on top
        self.ml_enhancement = SmallMLEnhancement()
    
    def apply_3d_lut_approximation(self, x):
        """Approximate 3D LUT with learned parameters"""
        # This could be replaced with actual LUT loading later
        return x
    
    def apply_color_matrix(self, x):
        """Apply constrained color matrix"""
        # Constrain matrix to prevent extreme values
        matrix_constrained = torch.clamp(self.color_matrix, 0.7, 1.3)
        bias_constrained = torch.clamp(self.color_bias, -0.1, 0.1)
        
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1)
        
        # Apply color matrix
        transformed = torch.bmm(matrix_constrained.unsqueeze(0).expand(b, -1, -1), x_flat)
        transformed = transformed + bias_constrained.view(1, 3, 1)
        
        return transformed.view(b, c, h, w)
    
    def apply_cinema_tone_curve(self, x):
        """Apply cinematic tone curve"""
        # Constrain tone curve parameters
        shadows_c = torch.clamp(self.shadows, -0.2, 0.2)
        mids_c = torch.clamp(self.mids, 0.8, 1.2)
        highlights_c = torch.clamp(self.highlights, 0.8, 1.2)
        
        # 3-point tone curve
        shadows_mask = (x < 0.33).float()
        mids_mask = ((x >= 0.33) & (x < 0.66)).float()
        highlights_mask = (x >= 0.66).float()
        
        result = (shadows_mask * x * (1 + shadows_c) +
                 mids_mask * x * mids_c +
                 highlights_mask * x * highlights_c)
        
        return result
    
    def apply_color_grading(self, x):
        """Apply basic color grading"""
        # Contrast (constrained)
        contrast_c = torch.clamp(self.contrast, 0.8, 1.2)
        x = x * contrast_c
        
        # Saturation (constrained)
        saturation_c = torch.clamp(self.saturation, 0.7, 1.3)
        gray = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
        x = gray + saturation_c * (x - gray)
        
        # Warmth (color temperature)
        warmth_c = torch.clamp(self.warmth, -0.1, 0.1)
        x[:, 0, :, :] = x[:, 0, :, :] + warmth_c  # Red
        x[:, 2, :, :] = x[:, 2, :, :] - warmth_c  # Blue
        
        return x
    
    def white_balance_constraint(self, x):
        """Ensure gray stays gray"""
        # Sample middle gray
        gray_input = torch.ones_like(x) * 0.5
        gray_output = self.apply_classical_pipeline(gray_input)
        
        # Should remain close to 0.5
        wb_loss = F.mse_loss(gray_output, gray_input)
        return wb_loss
    
    def apply_classical_pipeline(self, x):
        """Classical color science pipeline (80% of the work)"""
        # Step 1: 3D LUT approximation
        x = self.apply_3d_lut_approximation(x)
        
        # Step 2: Color matrix correction
        x = self.apply_color_matrix(x)
        
        # Step 3: Cinema tone curve
        x = self.apply_cinema_tone_curve(x)
        
        # Step 4: Color grading
        x = self.apply_color_grading(x)
        
        return torch.clamp(x, 0, 1)
    
    def forward(self, x):
        # 80% - Classical pipeline
        classical_result = self.apply_classical_pipeline(x)
        
        # 20% - ML enhancement (very small)
        final_result = self.ml_enhancement(classical_result)
        
        return final_result

class HybridDataset(Dataset):
    """Dataset for hybrid training"""
    
    def __init__(self, training_data_path: Path):
        self.data_path = training_data_path
        self.pairs = self.load_pairs()
        
    def load_pairs(self):
        metadata_file = self.data_path / "depth_metadata.json"
        with open(metadata_file, 'r') as f:
            pairs = json.load(f)
        return pairs
    
    def process_image_clean(self, file_path: str, size: int = 256, test_mode: bool = False):
        """Process image cleanly - high res for testing"""
        try:
            with rawpy.imread(file_path) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    output_bps=16,
                    no_auto_bright=True,
                    user_flip=0
                )
            
            rgb_norm = rgb.astype(np.float32) / 65535.0
            
            if test_mode:
                # Use higher resolution for testing (max 1024 to fit in memory)
                h, w = rgb_norm.shape[:2]
                max_dim = 1024
                if max(h, w) > max_dim:
                    scale = max_dim / max(h, w)
                    new_h, new_w = int(h * scale), int(w * scale)
                    rgb_resized = cv2.resize(rgb_norm, (new_w, new_h))
                else:
                    rgb_resized = rgb_norm
            else:
                # Training mode - use smaller size
                rgb_resized = cv2.resize(rgb_norm, (size, size))
            
            return np.transpose(rgb_resized, (2, 0, 1))
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        iphone_img = self.process_image_clean(pair['iphone_file'])
        sony_img = self.process_image_clean(pair['sony_file'])
        
        if iphone_img is None or sony_img is None:
            return self.__getitem__((idx + 1) % len(self.pairs))
        
        return {
            'iphone': torch.FloatTensor(iphone_img),
            'sony': torch.FloatTensor(sony_img),
            'pair_id': idx
        }

def white_balance_loss(predicted_batch):
    """White balance constraint loss"""
    # Create gray test input
    gray_test = torch.ones_like(predicted_batch) * 0.5
    
    # The prediction should keep gray as gray
    gray_pred = predicted_batch.mean(dim=(2, 3), keepdim=True).expand_as(predicted_batch)
    
    # Gray pixels should have equal R, G, B
    gray_variance = torch.var(predicted_batch.mean(dim=(2, 3)), dim=1)
    
    return gray_variance.mean()

def histogram_loss(pred, target, bins=64):
    """Histogram matching loss"""
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    pred_hist = torch.histc(pred_flat, bins=bins, min=0, max=1)
    target_hist = torch.histc(target_flat, bins=bins, min=0, max=1)
    
    pred_hist = pred_hist / (pred_hist.sum() + 1e-8)
    target_hist = target_hist / (target_hist.sum() + 1e-8)
    
    return F.mse_loss(pred_hist, target_hist)

def train_hybrid_model():
    """Train hybrid classical + ML model"""
    print("🎬 TRAINING HYBRID CLASSICAL + ML MODEL")
    print("=" * 50)
    print("80% Classical Color Science + 20% ML Enhancement")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    data_path = Path("data/results/simple_depth_analysis")
    
    if not (data_path / "depth_metadata.json").exists():
        print("❌ Training data not found!")
        return
    
    # Initialize dataset and model
    dataset = HybridDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    
    model = HybridCinemaTransform().to(device)
    
    # Conservative optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    
    print(f"📊 Dataset size: {len(dataset)} pairs")
    print(f"🔧 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training with balanced losses
    num_epochs = 30
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_mse = 0
        total_hist = 0
        total_wb = 0
        num_batches = 0
        
        for batch in dataloader:
            iphone_imgs = batch['iphone'].to(device)
            sony_imgs = batch['sony'].to(device)
            
            optimizer.zero_grad()
            predicted = model(iphone_imgs)
            
            # Balanced loss function
            mse_loss = F.mse_loss(predicted, sony_imgs)
            hist_loss = histogram_loss(predicted, sony_imgs)
            wb_loss = white_balance_loss(predicted)
            
            # Weighted combination (emphasis on stability)
            total_loss_batch = (0.6 * mse_loss +    # Primary objective
                              0.2 * hist_loss +     # Color distribution
                              0.2 * wb_loss)        # Color balance
            
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            
            total_loss += total_loss_batch.item()
            total_mse += mse_loss.item()
            total_hist += hist_loss.item()
            total_wb += wb_loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_mse = total_mse / max(num_batches, 1)
        avg_hist = total_hist / max(num_batches, 1)
        avg_wb = total_wb / max(num_batches, 1)
        
        # Print progress every 5 epochs
        if epoch % 5 == 0:
            print(f"\n📊 Epoch {epoch + 1}/{num_epochs}:")
            print(f"     Total: {avg_loss:.6f}, MSE: {avg_mse:.6f}")
            print(f"     Histogram: {avg_hist:.6f}, White Balance: {avg_wb:.6f}")
            
            # Print learned parameters
            with torch.no_grad():
                print(f"     Color Matrix diagonal: {torch.diag(model.color_matrix).data.cpu().numpy()}")
                print(f"     Contrast: {model.contrast.item():.3f}, Saturation: {model.saturation.item():.3f}")
                print(f"     Warmth: {model.warmth.item():.3f}")
                print(f"     ML Enhancement strength: {model.ml_enhancement.residual_strength.item():.3f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_loss,
                'mse_loss': avg_mse,
                'hist_loss': avg_hist,
                'wb_loss': avg_wb
            }, Path("data/hybrid_cinema_model.pth"))
    
    print(f"\n✅ Hybrid training complete!")
    print(f"🎯 Best loss: {best_loss:.6f}")
    print("🎬 Classical foundation + ML enhancement ready!")
    
    return model

def test_hybrid_model():
    """Test the hybrid model"""
    print("\n🧪 TESTING HYBRID CINEMA MODEL")
    print("=" * 50)
    
    model_path = Path("data/hybrid_cinema_model.pth")
    if not model_path.exists():
        print("❌ No hybrid model found. Train first!")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridCinemaTransform().to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded (epoch {checkpoint['epoch']})")
    print(f"📊 White balance loss: {checkpoint['wb_loss']:.6f}")
    
    # Test on training images
    training_pairs_dir = Path("data/training_pairs")
    iphone_files = list(training_pairs_dir.glob("iphone_*.dng"))[:5]
    
    results_dir = Path("data/results/hybrid_test")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for iphone_file in iphone_files:
        print(f"\n🎯 Processing: {iphone_file.name}")
        
        try:
            dataset = HybridDataset(Path("data/results/simple_depth_analysis"))
            processed_img = dataset.process_image_clean(str(iphone_file), test_mode=True)  # High res
            
            if processed_img is None:
                continue
                
            rgb_tensor = torch.FloatTensor(processed_img).unsqueeze(0).to(device)
            
            # Transform
            with torch.no_grad():
                transformed = model(rgb_tensor)
                transformed_np = transformed.cpu().squeeze(0).numpy()
                transformed_np = np.transpose(transformed_np, (1, 2, 0))
            
            # Original for comparison
            original_np = np.transpose(processed_img, (1, 2, 0))
            
            # Convert to display format
            original_display = (original_np * 255).astype(np.uint8)
            transformed_display = (transformed_np * 255).astype(np.uint8)
            
            # Create comparison (handle different sizes)
            h, w = original_display.shape[:2]
            comparison = np.hstack([
                cv2.cvtColor(original_display, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(transformed_display, cv2.COLOR_RGB2BGR)
            ])
            
            # Add labels (scale font for image size)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.6, min(2.0, h / 512))  # Scale font with image size
            thickness = max(1, int(h / 512))
            cv2.putText(comparison, "iPhone Original", (10, int(30 * font_scale)), font, font_scale, (0, 255, 0), thickness)
            cv2.putText(comparison, "Hybrid Cinema Transform", (w + 10, int(30 * font_scale)), font, font_scale, (0, 255, 0), thickness)
            
            # Save
            output_path = results_dir / f"hybrid_transform_{iphone_file.stem}.jpg"
            cv2.imwrite(str(output_path), comparison)
            
            print(f"  ✅ Saved: {output_path}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n✅ Hybrid test complete! Check: {results_dir}")

def analyze_hybrid_model():
    """Analyze what the hybrid model learned"""
    print("\n📊 ANALYZING HYBRID MODEL")
    print("=" * 50)
    
    model_path = Path("data/hybrid_cinema_model.pth")
    if not model_path.exists():
        print("❌ No model found!")
        return
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model = HybridCinemaTransform()
    model.load_state_dict(checkpoint['model_state_dict'])
    
    with torch.no_grad():
        print("🎨 CLASSICAL PARAMETERS:")
        matrix = model.color_matrix.data.numpy()
        bias = model.color_bias.data.numpy()
        
        print(f"  Color Matrix diagonal: R={matrix[0,0]:.3f}, G={matrix[1,1]:.3f}, B={matrix[2,2]:.3f}")
        print(f"  Color Bias: R={bias[0]:.3f}, G={bias[1]:.3f}, B={bias[2]:.3f}")
        print(f"  Tone: Shadows={model.shadows.item():.3f}, Mids={model.mids.item():.3f}, Highlights={model.highlights.item():.3f}")
        print(f"  Grading: Contrast={model.contrast.item():.3f}, Saturation={model.saturation.item():.3f}, Warmth={model.warmth.item():.3f}")
        
        print("\n🤖 ML ENHANCEMENT:")
        ml_strength = model.ml_enhancement.residual_strength.item()
        print(f"  ML residual strength: {ml_strength:.3f}")
        
        if ml_strength < 0.02:
            print("  ✅ Conservative ML enhancement")
        elif ml_strength < 0.05:
            print("  ⚠️ Moderate ML enhancement")
        else:
            print("  ❌ Aggressive ML enhancement")

def main():
    """Main hybrid training pipeline"""
    print("🎬 HYBRID CLASSICAL + ML CINEMA PIPELINE")
    print("Building reliable cinematic transformations")
    print("80% Classical + 20% ML from day one")
    print("\nChoose option:")
    print("1. Train hybrid model")
    print("2. Test hybrid model")
    print("3. Analyze hybrid model")
    print("4. Full pipeline")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice in ["1", "4"]:
        train_hybrid_model()
    
    if choice in ["2", "4"]:
        test_hybrid_model()
    
    if choice in ["3", "4"]:
        analyze_hybrid_model()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Hybrid Cinema Model v1.1 Stable - Fixed numerical stability issues
Prevents NaN values during training
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
import matplotlib.pyplot as plt

class StableMLComponent(nn.Module):
    """ML component with numerical stability improvements"""
    
    def __init__(self):
        super(StableMLComponent, self).__init__()
        
        # Network with batch normalization for stability
        self.color_enhance = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Tanh()
        )
        
        # Conservative initial residual strength
        self.residual_strength = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x):
        enhancement = self.color_enhance(x)
        # Clamp to prevent extreme values
        strength = torch.clamp(self.residual_strength, 0.05, 0.25)
        output = x + strength * enhancement
        return torch.clamp(output, 0, 1)

class StableHybridCinemaTransform(nn.Module):
    """Stable hybrid transformation with NaN prevention"""
    
    def __init__(self):
        super(StableHybridCinemaTransform, self).__init__()
        
        # Initialize closer to identity for stability
        self.color_matrix = nn.Parameter(torch.eye(3) * 0.95 + 0.05)
        self.color_bias = nn.Parameter(torch.zeros(3) * 0.01)
        
        # Conservative initial tone curve
        self.shadows = nn.Parameter(torch.tensor(0.0))
        self.mids = nn.Parameter(torch.tensor(1.0))
        self.highlights = nn.Parameter(torch.tensor(1.0))
        
        # Conservative grading parameters
        self.contrast = nn.Parameter(torch.tensor(1.0))
        self.saturation = nn.Parameter(torch.tensor(1.0))
        self.vibrance = nn.Parameter(torch.tensor(0.0))
        self.warmth = nn.Parameter(torch.tensor(0.0))
        
        # Stable ML component
        self.ml_enhancement = StableMLComponent()
        
        # Fixed blend ratio initially
        self.ml_blend = nn.Parameter(torch.tensor(0.2))
    
    def apply_color_matrix(self, x):
        """Numerically stable color matrix"""
        # Ensure matrix stays reasonable
        matrix = torch.clamp(self.color_matrix, 0.3, 2.0)
        bias = torch.clamp(self.color_bias, -0.3, 0.3)
        
        # Add small epsilon for stability
        matrix = matrix + torch.eye(3).to(matrix.device) * 1e-6
        
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1)
        
        transformed = torch.bmm(matrix.unsqueeze(0).expand(b, -1, -1), x_flat)
        transformed = transformed + bias.view(1, 3, 1)
        
        return transformed.view(b, c, h, w)
    
    def apply_cinema_tone_curve(self, x):
        """Stable tone curve application"""
        # Conservative clamps
        shadows_c = torch.clamp(self.shadows, -0.2, 0.2)
        mids_c = torch.clamp(self.mids, 0.8, 1.2)
        highlights_c = torch.clamp(self.highlights, 0.8, 1.2)
        
        # Prevent division by zero
        x_safe = torch.clamp(x, 1e-6, 1.0 - 1e-6)
        
        # Smooth curve
        x_adjusted = x_safe.clone()
        
        # Shadow adjustment
        shadow_mask = torch.pow(1 - x_safe, 2)
        x_adjusted = x_adjusted + shadows_c * shadow_mask * 0.5
        
        # Midtone adjustment
        mid_mask = 4 * x_safe * (1 - x_safe)
        x_adjusted = x_adjusted * (1 + (mids_c - 1) * mid_mask * 0.5)
        
        # Highlight adjustment
        highlight_mask = torch.pow(x_safe, 2)
        x_adjusted = x_adjusted * (1 + (highlights_c - 1) * highlight_mask * 0.5)
        
        return torch.clamp(x_adjusted, 0, 1)
    
    def apply_color_grading(self, x):
        """Stable color grading"""
        # Conservative limits
        contrast_c = torch.clamp(self.contrast, 0.8, 1.3)
        saturation_c = torch.clamp(self.saturation, 0.7, 1.3)
        vibrance_c = torch.clamp(self.vibrance, -0.2, 0.2)
        warmth_c = torch.clamp(self.warmth, -0.1, 0.1)
        
        # Stable contrast
        x_safe = torch.clamp(x, 1e-6, 1.0 - 1e-6)
        x_contrasted = torch.pow(x_safe, 1.0 / contrast_c)
        
        # Luminance calculation
        gray = 0.299 * x_contrasted[:, 0:1, :, :] + \
               0.587 * x_contrasted[:, 1:2, :, :] + \
               0.114 * x_contrasted[:, 2:3, :, :]
        
        # Saturation
        x_saturated = gray + saturation_c * (x_contrasted - gray)
        
        # Vibrance (with stability)
        sat_diff = torch.abs(x_saturated - gray)
        sat_mask = torch.exp(-sat_diff * 2)  # Smooth mask
        x_saturated = x_saturated + vibrance_c * sat_mask * (x_saturated - gray) * 0.5
        
        # Warmth
        x_final = x_saturated.clone()
        x_final[:, 0, :, :] = torch.clamp(x_final[:, 0, :, :] + warmth_c, 0, 1)
        x_final[:, 2, :, :] = torch.clamp(x_final[:, 2, :, :] - warmth_c * 0.7, 0, 1)
        
        return torch.clamp(x_final, 0, 1)
    
    def forward(self, x):
        # Add small epsilon to input for stability
        x = torch.clamp(x, 1e-6, 1.0 - 1e-6)
        
        # Classical pipeline
        x_classical = self.apply_color_matrix(x)
        x_classical = self.apply_cinema_tone_curve(x_classical)
        x_classical = self.apply_color_grading(x_classical)
        
        # ML enhancement
        x_ml = self.ml_enhancement(x_classical)
        
        # Stable blending
        blend_ratio = torch.clamp(self.ml_blend, 0.1, 0.3)
        output = (1 - blend_ratio) * x_classical + blend_ratio * x_ml
        
        return torch.clamp(output, 0, 1)

def stable_histogram_loss(pred, target, bins=64):
    """Numerically stable histogram loss"""
    loss = 0
    eps = 1e-8
    
    for c in range(3):
        # Add small noise to prevent empty bins
        pred_c = pred[:, c, :, :] + torch.randn_like(pred[:, c, :, :]) * 1e-4
        target_c = target[:, c, :, :] + torch.randn_like(target[:, c, :, :]) * 1e-4
        
        pred_hist = torch.histc(pred_c, bins=bins, min=0, max=1)
        target_hist = torch.histc(target_c, bins=bins, min=0, max=1)
        
        # Normalize with epsilon
        pred_hist = pred_hist / (pred_hist.sum() + eps)
        target_hist = target_hist / (target_hist.sum() + eps)
        
        # Add small epsilon to prevent log(0)
        pred_hist = pred_hist + eps
        target_hist = target_hist + eps
        
        # KL divergence (more stable than MSE for histograms)
        kl_div = torch.sum(target_hist * torch.log(target_hist / pred_hist))
        loss += kl_div
    
    return loss / 3

def stable_color_loss(pred, target):
    """Stable color consistency loss"""
    eps = 1e-6
    
    # Channel ratios with stability
    pred_r = torch.clamp(pred[:, 0, :, :], eps, 1)
    pred_g = torch.clamp(pred[:, 1, :, :], eps, 1)
    pred_b = torch.clamp(pred[:, 2, :, :], eps, 1)
    
    target_r = torch.clamp(target[:, 0, :, :], eps, 1)
    target_g = torch.clamp(target[:, 1, :, :], eps, 1)
    target_b = torch.clamp(target[:, 2, :, :], eps, 1)
    
    # Log ratios (more stable)
    pred_rg = torch.log(pred_r / pred_g)
    pred_rb = torch.log(pred_r / pred_b)
    
    target_rg = torch.log(target_r / target_g)
    target_rb = torch.log(target_r / target_b)
    
    return F.l1_loss(pred_rg, target_rg) + F.l1_loss(pred_rb, target_rb)

class StableDataset(Dataset):
    """Dataset with stable preprocessing"""
    
    def __init__(self, training_data_path: Path, size: int = 384):
        self.data_path = training_data_path
        self.size = size
        self.pairs = self.load_pairs()
    
    def load_pairs(self):
        metadata_file = self.data_path / "depth_metadata.json"
        with open(metadata_file, 'r') as f:
            pairs = json.load(f)
        return pairs
    
    def process_image_stable(self, file_path: str):
        """Stable image processing"""
        try:
            with rawpy.imread(file_path) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    output_bps=16,
                    no_auto_bright=True,
                    user_flip=0
                )
            
            # Normalize with clipping
            rgb_norm = np.clip(rgb.astype(np.float32) / 65535.0, 0, 1)
            
            # Add small epsilon to prevent pure black
            rgb_norm = rgb_norm + 1e-6
            
            # Resize
            h, w = rgb_norm.shape[:2]
            if min(h, w) > self.size:
                # Center crop
                y = (h - self.size) // 2
                x = (w - self.size) // 2
                rgb_crop = rgb_norm[y:y+self.size, x:x+self.size]
            else:
                rgb_crop = cv2.resize(rgb_norm, (self.size, self.size))
            
            return np.transpose(rgb_crop, (2, 0, 1))
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        iphone_img = self.process_image_stable(pair['iphone_file'])
        sony_img = self.process_image_stable(pair['sony_file'])
        
        if iphone_img is None or sony_img is None:
            return self.__getitem__((idx + 1) % len(self.pairs))
        
        return {
            'iphone': torch.FloatTensor(iphone_img),
            'sony': torch.FloatTensor(sony_img),
            'pair_id': idx
        }

def train_stable_model():
    """Train with numerical stability"""
    print("🎬 TRAINING STABLE HYBRID CINEMA MODEL V1.1")
    print("=" * 50)
    print("Fixed numerical stability issues")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    data_path = Path("data/results/simple_depth_analysis")
    
    # Stable dataset
    dataset = StableDataset(data_path, size=384)  # Slightly smaller for stability
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    
    model = StableHybridCinemaTransform().to(device)
    
    # Lower learning rate for stability
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    print(f"📊 Dataset size: {len(dataset)} pairs")
    print(f"🔧 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"📐 Training resolution: 384x384")
    
    num_epochs = 40
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        valid_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            iphone_imgs = batch['iphone'].to(device)
            sony_imgs = batch['sony'].to(device)
            
            # Check for NaN in inputs
            if torch.isnan(iphone_imgs).any() or torch.isnan(sony_imgs).any():
                print(f"Warning: NaN in input batch {batch_idx}, skipping")
                continue
            
            optimizer.zero_grad()
            
            # Forward pass with gradient checking
            predicted = model(iphone_imgs)
            
            # Check for NaN in output
            if torch.isnan(predicted).any():
                print(f"Warning: NaN in output batch {batch_idx}, skipping")
                continue
            
            # Stable loss computation
            mse_loss = F.mse_loss(predicted, sony_imgs)
            hist_loss = stable_histogram_loss(predicted, sony_imgs)
            color_loss = stable_color_loss(predicted, sony_imgs)
            
            # Check for NaN in losses
            if torch.isnan(mse_loss) or torch.isnan(hist_loss) or torch.isnan(color_loss):
                print(f"Warning: NaN in loss batch {batch_idx}, skipping")
                continue
            
            # Conservative weighting
            total_loss_batch = (
                0.6 * mse_loss +
                0.2 * hist_loss +
                0.2 * color_loss
            )
            
            # Gradient clipping before backward
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            # Check for NaN in gradients
            has_nan_grad = False
            for param in model.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                print(f"Warning: NaN in gradients batch {batch_idx}, skipping")
                optimizer.zero_grad()
                continue
            
            optimizer.step()
            
            total_loss += total_loss_batch.item()
            valid_batches += 1
        
        if valid_batches == 0:
            print(f"Epoch {epoch + 1}: No valid batches!")
            continue
        
        avg_loss = total_loss / valid_batches
        scheduler.step(avg_loss)
        
        if epoch % 5 == 0:
            print(f"\n📊 Epoch {epoch + 1}/{num_epochs}:")
            print(f"     Avg Loss: {avg_loss:.6f} (from {valid_batches} valid batches)")
            
            # Print parameters (check for NaN)
            with torch.no_grad():
                matrix_diag = torch.diag(model.color_matrix).cpu().numpy()
                if not np.isnan(matrix_diag).any():
                    print(f"     Color Matrix: {matrix_diag}")
                    print(f"     Grading: C={model.contrast.item():.3f}, S={model.saturation.item():.3f}")
                else:
                    print("     Parameters contain NaN!")
        
        # Save if better
        if avg_loss < best_loss and not np.isnan(avg_loss):
            best_loss = avg_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_loss
            }, Path("data/hybrid_cinema_v1_1_stable.pth"))
        else:
            patience_counter += 1
            if patience_counter > 10:
                print("Early stopping due to no improvement")
                break
    
    print(f"\n✅ Training complete!")
    print(f"🎯 Best loss: {best_loss:.6f}")
    return model

def test_stable_model():
    """Test the stable model"""
    print("\n🧪 TESTING STABLE MODEL")
    print("=" * 50)
    
    model_path = Path("data/hybrid_cinema_v1_1_stable.pth")
    if not model_path.exists():
        print("❌ No stable model found. Train first!")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StableHybridCinemaTransform().to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded (epoch {checkpoint['epoch']})")
    
    # Test on images
    training_pairs_dir = Path("data/training_pairs")
    iphone_files = list(training_pairs_dir.glob("iphone_*.dng"))[:5]
    
    results_dir = Path("data/results/hybrid_v1_1_stable")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = StableDataset(Path("data/results/simple_depth_analysis"))
    
    for iphone_file in iphone_files:
        print(f"\n🎯 Processing: {iphone_file.name}")
        
        try:
            processed_img = dataset.process_image_stable(str(iphone_file))
            
            if processed_img is None:
                continue
            
            # Process at higher resolution for testing
            with rawpy.imread(str(iphone_file)) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    output_bps=16,
                    no_auto_bright=True,
                    user_flip=0
                )
            
            rgb_norm = np.clip(rgb.astype(np.float32) / 65535.0, 0, 1) + 1e-6
            h, w = rgb_norm.shape[:2]
            max_dim = 1536
            
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                rgb_resized = cv2.resize(rgb_norm, (new_w, new_h))
            else:
                rgb_resized = rgb_norm
            
            rgb_tensor = torch.FloatTensor(np.transpose(rgb_resized, (2, 0, 1))).unsqueeze(0).to(device)
            
            # Transform
            with torch.no_grad():
                transformed = model(rgb_tensor)
                transformed_np = transformed.cpu().squeeze(0).numpy()
                transformed_np = np.transpose(transformed_np, (1, 2, 0))
            
            # Save result
            original_display = (rgb_resized * 255).astype(np.uint8)
            transformed_display = (transformed_np * 255).astype(np.uint8)
            
            comparison = np.hstack([
                cv2.cvtColor(original_display, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(transformed_display, cv2.COLOR_RGB2BGR)
            ])
            
            output_path = results_dir / f"stable_{iphone_file.stem}.jpg"
            cv2.imwrite(str(output_path), comparison, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            print(f"  ✅ Saved: {output_path}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n✅ Test complete! Check: {results_dir}")

def main():
    """Main entry point"""
    print("🎬 STABLE HYBRID CINEMA MODEL V1.1")
    print("Fixed numerical stability issues")
    print("\nChoose option:")
    print("1. Train stable model")
    print("2. Test stable model")
    print("3. Both")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice in ["1", "3"]:
        train_stable_model()
    
    if choice in ["2", "3"]:
        test_stable_model()

if __name__ == "__main__":
    main()
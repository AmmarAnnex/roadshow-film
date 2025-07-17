#!/usr/bin/env python3
"""
Cinema Model v1.4m Improved - Complete Training Script
Conservative exposure fix that targets Sony accuracy, not brightness
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

class ImprovedExposureFixedColorTransform(nn.Module):
    """Improved v1.4m with much more conservative parameters"""
    
    def __init__(self):
        super(ImprovedExposureFixedColorTransform, self).__init__()
        
        # Same CNN architecture but conservative
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv_out = nn.Conv2d(16, 3, 3, padding=1)
        
        # Professional color matrix (much smaller initial values)
        self.color_matrix = nn.Parameter(torch.eye(3) + torch.randn(3, 3) * 0.002)  # 5x smaller
        self.color_bias = nn.Parameter(torch.randn(3) * 0.001)  # 5x smaller
        
        # FIXED: Much more conservative exposure parameters
        self.global_exposure = nn.Parameter(torch.tensor(1.01))   # Was 1.05 -> 1.01 (tiny boost)
        self.shadows = nn.Parameter(torch.tensor(0.01))           # Was 0.05 -> 0.01 (gentle lift)
        self.mids = nn.Parameter(torch.tensor(1.0))               # Neutral
        self.highlights = nn.Parameter(torch.tensor(0.98))        # Slight protection
        
        # Much more conservative color grading
        self.contrast = nn.Parameter(torch.tensor(1.01))          # Was 1.03 -> 1.01
        self.saturation = nn.Parameter(torch.tensor(1.02))        # Was 1.08 -> 1.02
        self.warmth = nn.Parameter(torch.tensor(0.002))           # Was 0.01 -> 0.002
        
        # Much smaller residual strength (like v1.4 4K)
        self.residual_strength = nn.Parameter(torch.tensor(0.02)) # Was 0.05 -> 0.02
        
    def apply_exposure_correction(self, x):
        """MUCH more conservative exposure handling"""
        exposure_factor = torch.clamp(self.global_exposure, 0.98, 1.03)  # Tight bounds
        return x * exposure_factor
    
    def apply_color_matrix(self, x):
        """Apply professional 3x3 color matrix with tighter bounds"""
        matrix = torch.clamp(self.color_matrix, 0.95, 1.05)    # Much tighter bounds
        bias = torch.clamp(self.color_bias, -0.005, 0.005)     # Much smaller bias
        
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1)
        
        # Matrix multiplication
        transformed = torch.bmm(matrix.unsqueeze(0).expand(b, -1, -1), x_flat)
        
        # Add bias
        bias_expanded = bias.view(1, 3, 1).expand(b, -1, x_flat.size(2))
        transformed = transformed + bias_expanded
        
        return transformed.view(b, c, h, w)
    
    def apply_professional_tone_curve(self, x):
        """Much more gentle tone curve"""
        shadows_c = torch.clamp(self.shadows, 0.0, 0.02)        # Much smaller range
        mids_c = torch.clamp(self.mids, 0.98, 1.02)             # Very tight
        highlights_c = torch.clamp(self.highlights, 0.95, 1.0)  # Protect highlights
        
        # Calculate luminance for masking
        luma = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        
        # Create smooth masks
        shadow_mask = (1 - luma).clamp(0, 1) ** 1.5
        highlight_mask = luma.clamp(0, 1) ** 1.5
        mid_mask = 1 - shadow_mask - highlight_mask
        
        # Much gentler adjustments
        shadow_adj = shadows_c * shadow_mask * 0.02           # Was 0.05 -> 0.02
        mid_adj = (mids_c - 1.0) * mid_mask * 0.2             # Was 0.5 -> 0.2
        highlight_adj = (highlights_c - 1.0) * highlight_mask * 0.1  # Was 0.3 -> 0.1
        
        return x + shadow_adj + mid_adj + highlight_adj
    
    def apply_color_grading(self, x):
        """Much more conservative color grading"""
        contrast_c = torch.clamp(self.contrast, 0.98, 1.05)      # Tighter bounds
        saturation_c = torch.clamp(self.saturation, 0.95, 1.1)   # Much smaller range
        warmth_c = torch.clamp(self.warmth, -0.005, 0.005)       # Tiny warmth
        
        # Contrast adjustment around 0.5 middle gray
        x_contrast = (x - 0.5) * contrast_c + 0.5
        
        # Saturation adjustment
        luma = 0.299 * x_contrast[:, 0:1] + 0.587 * x_contrast[:, 1:2] + 0.114 * x_contrast[:, 2:3]
        x_saturated = luma + (x_contrast - luma) * saturation_c
        
        # Warmth adjustment (minimal)
        warmth_matrix = torch.tensor([
            [1.0 + warmth_c, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0 - warmth_c]
        ], device=x.device, dtype=x.dtype)
        
        b, c, h, w = x_saturated.shape
        x_flat = x_saturated.view(b, c, -1)
        x_warm = torch.bmm(warmth_matrix.unsqueeze(0).expand(b, -1, -1), x_flat)
        
        return x_warm.view(b, c, h, w)
    
    def forward(self, x):
        # Store original for residual
        x_original = x
        
        # Apply GENTLE exposure correction
        x = self.apply_exposure_correction(x)
        
        # Apply color matrix
        x = self.apply_color_matrix(x)
        
        # Professional tone curve (much gentler)
        x = self.apply_professional_tone_curve(x)
        
        # Color grading (much more conservative)
        x = self.apply_color_grading(x)
        
        # CNN residual refinement (smaller strength)
        residual = F.relu(self.conv1(x_original))
        residual = F.relu(self.conv2(residual))
        residual = F.relu(self.conv3(residual))
        residual = F.relu(self.conv4(residual))
        residual = torch.tanh(self.conv_out(residual))
        
        # Apply residual with much smaller strength
        strength = torch.clamp(self.residual_strength, 0.01, 0.03)  # Much smaller
        x_final = x + strength * residual
        
        return torch.clamp(x_final, 0, 1)

# Import dataset from existing modules
class AdvancedDataset(Dataset):
    """Enhanced dataset supporting 79 pairs with professional color handling"""
    
    def __init__(self, data_path, target_size=768, max_pairs=79):
        self.data_path = Path(data_path)
        self.target_size = target_size
        self.max_pairs = max_pairs
        
        # Load metadata
        metadata_file = self.data_path / "depth_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = []
        
        # Filter valid pairs (limit to max_pairs for consistent training)
        self.valid_pairs = []
        training_pairs_dir = Path("data/training_pairs")
        
        for i in range(1, self.max_pairs + 1):
            iphone_file = training_pairs_dir / f"iphone_{i:03d}.dng"
            sony_file = training_pairs_dir / f"sony_{i:03d}.arw"
            
            if iphone_file.exists() and sony_file.exists():
                self.valid_pairs.append({
                    'iphone': iphone_file,
                    'sony': sony_file,
                    'pair_id': i
                })
        
        print(f"📊 Dataset: {len(self.valid_pairs)} valid pairs (max {max_pairs})")
    
    def load_and_process_raw(self, file_path, target_size):
        """Load RAW with professional color handling"""
        try:
            with rawpy.imread(str(file_path)) as raw:
                # Professional RAW processing
                rgb = raw.postprocess(
                    demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                    half_size=False,
                    four_color_rgb=False,
                    dcb_iterations=0,
                    dcb_enhance=False,
                    fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode.Off,
                    noise_thr=None,
                    median_filter_passes=0,
                    use_camera_wb=True,
                    use_auto_wb=False,
                    user_wb=None,
                    output_color=rawpy.ColorSpace.sRGB,
                    output_bps=16,
                    bright=1.0,
                    highlight_mode=rawpy.HighlightMode.Clip,
                    exp_shift=None,
                    exp_preserve_highlights=0.0,
                    no_auto_bright=True,
                    auto_bright_thr=None,
                    gamma=(1, 1),  # Linear gamma for professional workflow
                    user_flip=None,
                    user_black=None,
                    user_sat=None
                )
            
            # Convert to float and normalize to [0,1]
            rgb_float = rgb.astype(np.float32) / 65535.0
            
            # Intelligent crop to square (center + subject detection)
            h, w = rgb_float.shape[:2]
            size = min(h, w)
            
            # Center crop
            start_y = (h - size) // 2
            start_x = (w - size) // 2
            rgb_crop = rgb_float[start_y:start_y + size, start_x:start_x + size]
            
            # Resize to target
            rgb_resized = cv2.resize(rgb_crop, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
            
            return rgb_resized
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return np.zeros((target_size, target_size, 3), dtype=np.float32)
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        pair = self.valid_pairs[idx]
        
        # Load images
        iphone_img = self.load_and_process_raw(pair['iphone'], self.target_size)
        sony_img = self.load_and_process_raw(pair['sony'], self.target_size)
        
        # Convert to tensors
        iphone_tensor = torch.from_numpy(iphone_img).permute(2, 0, 1)  # CHW
        sony_tensor = torch.from_numpy(sony_img).permute(2, 0, 1)
        
        # Professional validation
        if torch.any(torch.isnan(iphone_tensor)) or torch.any(torch.isnan(sony_tensor)):
            print(f"Warning: NaN detected in pair {pair['pair_id']}")
            # Return zero tensors if corrupted
            return {
                'iphone': torch.zeros_like(iphone_tensor),
                'sony': torch.zeros_like(sony_tensor),
                'pair_id': pair['pair_id']
            }
        
        return {
            'iphone': iphone_tensor,
            'sony': sony_tensor,
            'pair_id': pair['pair_id']
        }

def professional_loss_function(predicted, target):
    """Professional loss function based on perceptual and colorimetric principles"""
    
    # L1 loss for general fidelity
    l1_loss = F.l1_loss(predicted, target)
    
    # Perceptual loss in LAB color space (more perceptually uniform)
    def rgb_to_lab_approx(rgb):
        # Simplified RGB to LAB conversion
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        
        # Approximate XYZ conversion
        x = 0.412453 * r + 0.357580 * g + 0.180423 * b
        y = 0.212671 * r + 0.715160 * g + 0.072169 * b
        z = 0.019334 * r + 0.119193 * g + 0.950227 * b
        
        # Approximate LAB conversion
        l = 116 * torch.pow(y.clamp(min=1e-8), 1/3) - 16
        a = 500 * (torch.pow(x.clamp(min=1e-8), 1/3) - torch.pow(y.clamp(min=1e-8), 1/3))
        b_lab = 200 * (torch.pow(y.clamp(min=1e-8), 1/3) - torch.pow(z.clamp(min=1e-8), 1/3))
        
        return torch.cat([l/100, a/127, b_lab/127], dim=1)  # Normalize
    
    # LAB perceptual loss
    pred_lab = rgb_to_lab_approx(predicted)
    target_lab = rgb_to_lab_approx(target)
    lab_loss = F.mse_loss(pred_lab, target_lab)
    
    # Histogram matching loss (professional color matching)
    def histogram_loss(pred, tgt):
        # Simple histogram matching in 3D
        pred_flat = pred.view(pred.size(0), pred.size(1), -1)
        tgt_flat = tgt.view(tgt.size(0), tgt.size(1), -1)
        
        # Mean and std matching
        pred_mean = pred_flat.mean(dim=2, keepdim=True)
        pred_std = pred_flat.std(dim=2, keepdim=True)
        tgt_mean = tgt_flat.mean(dim=2, keepdim=True)
        tgt_std = tgt_flat.std(dim=2, keepdim=True)
        
        mean_loss = F.mse_loss(pred_mean, tgt_mean)
        std_loss = F.mse_loss(pred_std, tgt_std)
        
        return mean_loss + std_loss
    
    hist_loss = histogram_loss(predicted, target)
    
    # Professional edge preservation (detail retention)
    def edge_loss(pred, tgt):
        # Sobel edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        
        # Calculate edges for each channel
        edge_loss_total = 0
        for i in range(3):
            pred_ch = pred[:, i:i+1, :, :]
            tgt_ch = tgt[:, i:i+1, :, :]
            
            pred_edge_x = F.conv2d(pred_ch, sobel_x, padding=1)
            pred_edge_y = F.conv2d(pred_ch, sobel_y, padding=1)
            tgt_edge_x = F.conv2d(tgt_ch, sobel_x, padding=1)
            tgt_edge_y = F.conv2d(tgt_ch, sobel_y, padding=1)
            
            pred_edge = torch.sqrt(pred_edge_x**2 + pred_edge_y**2 + 1e-8)
            tgt_edge = torch.sqrt(tgt_edge_x**2 + tgt_edge_y**2 + 1e-8)
            
            edge_loss_total += F.mse_loss(pred_edge, tgt_edge)
        
        return edge_loss_total / 3
    
    edge_preservation_loss = edge_loss(predicted, target)
    
    # Combined professional loss
    total_loss = (0.4 * l1_loss + 
                  0.3 * lab_loss + 
                  0.2 * hist_loss + 
                  0.1 * edge_preservation_loss)
    
    return total_loss, l1_loss, lab_loss, hist_loss, edge_preservation_loss

def train_v14m_improved():
    """Train the improved v1.4m model with conservative parameters"""
    print("🎬 TRAINING IMPROVED CINEMA MODEL V1.4M")
    print("=" * 50)
    print("Conservative exposure fix that targets Sony accuracy")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    # Use consistent training parameters
    target_size = 768
    max_pairs = 79
    
    dataset = AdvancedDataset(
        data_path=Path("data/results/simple_depth_analysis"), 
        target_size=target_size,
        max_pairs=max_pairs
    )
    
    batch_size = 2 if device.type == 'cuda' else 1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Use the improved model
    model = ImprovedExposureFixedColorTransform().to(device)
    
    # Print initial parameters to verify they're conservative
    print(f"📊 Initial Conservative Parameters:")
    print(f"   Global exposure: {model.global_exposure.item():.3f}")
    print(f"   Shadows: {model.shadows.item():.3f}")
    print(f"   Contrast: {model.contrast.item():.3f}")
    print(f"   Saturation: {model.saturation.item():.3f}")
    print(f"   Residual strength: {model.residual_strength.item():.3f}")
    
    # Professional optimizer settings
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    print(f"📊 Training: {len(dataset)} pairs at {target_size}x{target_size}")
    print(f"🔧 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"💾 Batch size: {batch_size}")
    
    # Training loop
    num_epochs = 25
    best_loss = float('inf')
    loss_history = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        epoch_l1 = []
        epoch_lab = []
        epoch_hist = []
        epoch_edge = []
        
        for batch_idx, batch in enumerate(dataloader):
            iphone_imgs = batch['iphone'].to(device)
            sony_imgs = batch['sony'].to(device)
            
            # Validation
            if torch.any(torch.isnan(iphone_imgs)) or torch.any(torch.isnan(sony_imgs)):
                print(f"Warning: NaN in batch {batch_idx}")
                continue
            
            optimizer.zero_grad()
            
            # Forward pass
            predicted = model(iphone_imgs)
            
            # Professional loss computation
            total_loss, l1_loss, lab_loss, hist_loss, edge_loss = professional_loss_function(predicted, sony_imgs)
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Record metrics
            epoch_losses.append(total_loss.item())
            epoch_l1.append(l1_loss.item())
            epoch_lab.append(lab_loss.item())
            epoch_hist.append(hist_loss.item())
            epoch_edge.append(edge_loss.item())
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1:2d}/{num_epochs}, Batch {batch_idx:2d}, "
                      f"Loss: {total_loss.item():.4f}")
        
        # Epoch statistics
        avg_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
        avg_l1 = np.mean(epoch_l1) if epoch_l1 else 0
        avg_lab = np.mean(epoch_lab) if epoch_lab else 0
        avg_hist = np.mean(epoch_hist) if epoch_hist else 0
        avg_edge = np.mean(epoch_edge) if epoch_edge else 0
        
        scheduler.step()
        
        print(f"\nEpoch {epoch+1:2d} Summary:")
        print(f"  Total Loss: {avg_loss:.4f}")
        print(f"  L1 Loss: {avg_l1:.4f}")
        print(f"  LAB Loss: {avg_lab:.4f}")
        print(f"  Histogram: {avg_hist:.4f}")
        print(f"  Edge: {avg_edge:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Print learned parameters for monitoring (should stay conservative)
        with torch.no_grad():
            print(f"  Exposure: {model.global_exposure.item():.3f}")
            print(f"  Shadows: {model.shadows.item():.3f}")
            print(f"  Contrast: {model.contrast.item():.3f}")
            print(f"  Saturation: {model.saturation.item():.3f}")
            print(f"  Residual: {model.residual_strength.item():.3f}")
        
        loss_history.append({
            'epoch': epoch + 1,
            'total_loss': avg_loss,
            'l1_loss': avg_l1,
            'lab_loss': avg_lab,
            'hist_loss': avg_hist,
            'edge_loss': avg_edge
        })
        
        # Save best model
        if avg_loss < best_loss and avg_loss < float('inf'):
            best_loss = avg_loss
            model_path = Path("models/cinema_v1_4m_improved_model.pth")
            model_path.parent.mkdir(exist_ok=True)
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss': best_loss,
                'training_pairs': len(dataset),
                'target_size': target_size,
                'model_type': 'v1.4m_improved'
            }, model_path)
            
            print(f"  ✅ Best model saved: {best_loss:.4f}")
        
        print("-" * 50)
    
    # Save training history
    history_path = Path("data/results/v1_4m_improved_training_history.json")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, 'w') as f:
        json.dump(loss_history, f, indent=2)
    
    print(f"\n🎉 Training Complete!")
    print(f"📊 Best Loss: {best_loss:.4f}")
    print(f"💾 Model: models/cinema_v1_4m_improved_model.pth")
    print(f"📈 History: {history_path}")
    print(f"\n🔬 Next Steps:")
    print(f"   1. Test: python model_comparison_tool.py")
    print(f"   2. Compare with v1.4 4K to see improvement")

if __name__ == "__main__":
    train_v14m_improved()
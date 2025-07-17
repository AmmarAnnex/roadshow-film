#!/usr/bin/env python3
"""
Cinema Model v1.4m - Exposure Fixed
Addresses severe underexposure issues from v1.4l
Professional color science with proper brightness handling
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
import torchvision.transforms as transforms

class ExposureFixedColorTransform(nn.Module):
    """Professional color transform with fixed exposure handling"""
    
    def __init__(self):
        super(ExposureFixedColorTransform, self).__init__()
        
        # Multi-scale color processing
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv_out = nn.Conv2d(16, 3, 3, padding=1)
        
        # Professional color matrix (3x3)
        self.color_matrix = nn.Parameter(torch.eye(3) * 1.0 + torch.randn(3, 3) * 0.01)
        self.color_bias = nn.Parameter(torch.randn(3) * 0.005)
        
        # FIXED: Proper exposure and tone curve parameters
        self.global_exposure = nn.Parameter(torch.tensor(1.05))  # Slight brightness boost
        self.shadows = nn.Parameter(torch.tensor(0.05))          # Lift shadows
        self.mids = nn.Parameter(torch.tensor(1.0))              # Neutral mids
        self.highlights = nn.Parameter(torch.tensor(0.95))       # Protect highlights
        
        # Color grading controls
        self.contrast = nn.Parameter(torch.tensor(1.03))
        self.saturation = nn.Parameter(torch.tensor(1.08))
        self.warmth = nn.Parameter(torch.tensor(0.01))
        
        # FIXED: More conservative residual strength
        self.residual_strength = nn.Parameter(torch.tensor(0.05))
        
    def apply_exposure_correction(self, x):
        """FIXED: Proper exposure handling to prevent underexposure"""
        exposure_factor = torch.clamp(self.global_exposure, 0.9, 1.2)  # Prevent darkening
        return x * exposure_factor
    
    def apply_color_matrix(self, x):
        """Apply professional 3x3 color matrix"""
        matrix = torch.clamp(self.color_matrix, 0.8, 1.2)  # Conservative bounds
        bias = torch.clamp(self.color_bias, -0.02, 0.02)   # Small bias
        
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1)
        
        # Matrix multiplication
        transformed = torch.bmm(matrix.unsqueeze(0).expand(b, -1, -1), x_flat)
        
        # Add bias
        bias_expanded = bias.view(1, 3, 1).expand(b, -1, x_flat.size(2))
        transformed = transformed + bias_expanded
        
        return transformed.view(b, c, h, w)
    
    def apply_professional_tone_curve(self, x):
        """Professional tone curve with fixed exposure"""
        shadows_c = torch.clamp(self.shadows, 0.0, 0.1)      # Only lift, never crush
        mids_c = torch.clamp(self.mids, 0.95, 1.05)          # Conservative mids
        highlights_c = torch.clamp(self.highlights, 0.9, 1.0) # Protect highlights
        
        # Calculate luminance for masking
        luma = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        
        # Create smooth masks
        shadow_mask = (1 - luma).clamp(0, 1) ** 1.5
        highlight_mask = luma.clamp(0, 1) ** 1.5
        mid_mask = 1 - shadow_mask - highlight_mask
        
        # Apply adjustments (more conservative)
        shadow_adj = shadows_c * shadow_mask * 0.05   # Gentle shadow lift
        mid_adj = (mids_c - 1.0) * mid_mask * 0.5     # Conservative mid adjustment
        highlight_adj = (highlights_c - 1.0) * highlight_mask * 0.3  # Protect highlights
        
        return x + shadow_adj + mid_adj + highlight_adj
    
    def apply_color_grading(self, x):
        """Conservative color grading"""
        contrast_c = torch.clamp(self.contrast, 0.95, 1.1)    # Conservative contrast
        saturation_c = torch.clamp(self.saturation, 0.9, 1.2) # Conservative saturation
        warmth_c = torch.clamp(self.warmth, -0.01, 0.02)      # Subtle warmth
        
        # Contrast adjustment around 0.5 middle gray
        x_contrast = (x - 0.5) * contrast_c + 0.5
        
        # Saturation adjustment
        luma = 0.299 * x_contrast[:, 0:1] + 0.587 * x_contrast[:, 1:2] + 0.114 * x_contrast[:, 2:3]
        x_saturated = luma + (x_contrast - luma) * saturation_c
        
        # Warmth adjustment (very subtle)
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
        
        # FIXED: Apply exposure correction first to prevent underexposure
        x = self.apply_exposure_correction(x)
        
        # Apply color matrix
        x = self.apply_color_matrix(x)
        
        # Professional tone curve
        x = self.apply_professional_tone_curve(x)
        
        # Color grading
        x = self.apply_color_grading(x)
        
        # CNN residual refinement (conservative)
        residual = F.relu(self.conv1(x_original))
        residual = F.relu(self.conv2(residual))
        residual = F.relu(self.conv3(residual))
        residual = F.relu(self.conv4(residual))
        residual = torch.tanh(self.conv_out(residual))
        
        # Apply residual with conservative strength
        strength = torch.clamp(self.residual_strength, 0.02, 0.08)
        x_final = x + strength * residual
        
        return torch.clamp(x_final, 0, 1)

# Import dataset and loss functions from v1.4l
from cinema_v1_4l import AdvancedDataset, professional_loss_function

def train_v1_4m():
    """Train v1.4m with fixed exposure handling"""
    print("🎬 TRAINING CINEMA MODEL V1.4M - EXPOSURE FIXED")
    print("=" * 50)
    print("Professional Color Science + Fixed Exposure Handling")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    # Use corrected training pairs
    target_size = 768
    max_pairs = 79
    
    dataset = AdvancedDataset(
        data_path=Path("data/results/simple_depth_analysis"), 
        target_size=target_size,
        max_pairs=max_pairs
    )
    
    batch_size = 2 if device.type == 'cuda' else 1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Use the exposure-fixed model
    model = ExposureFixedColorTransform().to(device)
    
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
        
        # Print learned parameters for monitoring
        with torch.no_grad():
            print(f"  Exposure: {model.global_exposure.item():.3f}")
            print(f"  Shadows: {model.shadows.item():.3f}")
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
            model_path = Path("models/cinema_v1_4m_model.pth")
            model_path.parent.mkdir(exist_ok=True)
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss': best_loss,
                'training_pairs': len(dataset),
                'target_size': target_size
            }, model_path)
            
            print(f"  ✅ Best model saved: {best_loss:.4f}")
        
        print("-" * 50)
    
    # Save training history
    history_path = Path("data/results/v1_4m_training_history.json")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, 'w') as f:
        json.dump(loss_history, f, indent=2)
    
    print(f"\n🎉 Training Complete!")
    print(f"📊 Best Loss: {best_loss:.4f}")
    print(f"💾 Model: models/cinema_v1_4m_model.pth")
    print(f"📈 History: {history_path}")

if __name__ == "__main__":
    train_v1_4m()

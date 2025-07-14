#!/usr/bin/env python3
"""
Hybrid Cinema Model v1.1 - Enhanced Color Science
Fixes grey/desaturated output issues with better parameter ranges
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

class EnhancedMLComponent(nn.Module):
    """Enhanced ML component with better color preservation"""
    
    def __init__(self):
        super(EnhancedMLComponent, self).__init__()
        
        # Larger network for better color learning
        self.color_enhance = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Tanh()
        )
        
        # Adaptive residual strength (can go higher for more impact)
        self.residual_strength = nn.Parameter(torch.tensor(0.15))
    
    def forward(self, x):
        enhancement = self.color_enhance(x)
        # Allow stronger corrections
        output = x + torch.clamp(self.residual_strength, 0.05, 0.3) * enhancement
        return torch.clamp(output, 0, 1)

class HybridCinemaTransformV1_1(nn.Module):
    """Improved hybrid transformation with better color handling"""
    
    def __init__(self):
        super(HybridCinemaTransformV1_1, self).__init__()
        
        # Less constrained color matrix (wider range)
        self.color_matrix = nn.Parameter(torch.eye(3))
        self.color_bias = nn.Parameter(torch.zeros(3))
        
        # Enhanced tone curve parameters
        self.shadows = nn.Parameter(torch.tensor(0.0))
        self.mids = nn.Parameter(torch.tensor(1.0))
        self.highlights = nn.Parameter(torch.tensor(1.0))
        
        # Color grading with wider ranges
        self.contrast = nn.Parameter(torch.tensor(1.0))
        self.saturation = nn.Parameter(torch.tensor(1.0))
        self.vibrance = nn.Parameter(torch.tensor(0.0))  # New parameter
        self.warmth = nn.Parameter(torch.tensor(0.0))
        
        # Enhanced ML component
        self.ml_enhancement = EnhancedMLComponent()
        
        # Learnable classical/ML blend ratio
        self.ml_blend = nn.Parameter(torch.tensor(0.25))  # Start at 25%
    
    def apply_color_matrix(self, x):
        """Less constrained color matrix"""
        # Wider constraints for more flexibility
        matrix_constrained = torch.clamp(self.color_matrix, 0.5, 1.5)
        bias_constrained = torch.clamp(self.color_bias, -0.2, 0.2)
        
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1)
        
        transformed = torch.bmm(matrix_constrained.unsqueeze(0).expand(b, -1, -1), x_flat)
        transformed = transformed + bias_constrained.view(1, 3, 1)
        
        return transformed.view(b, c, h, w)
    
    def apply_cinema_tone_curve(self, x):
        """Enhanced S-curve for cinematic look"""
        # More aggressive parameters allowed
        shadows_c = torch.clamp(self.shadows, -0.3, 0.3)
        mids_c = torch.clamp(self.mids, 0.7, 1.3)
        highlights_c = torch.clamp(self.highlights, 0.7, 1.3)
        
        # Smoother curve application
        x_adjusted = x.clone()
        
        # Shadow lift
        shadow_mask = torch.pow(1 - x, 2)
        x_adjusted = x_adjusted + shadows_c * shadow_mask * (1 - x)
        
        # Midtone adjustment
        mid_mask = 4 * x * (1 - x)  # Peaks at 0.5
        x_adjusted = x_adjusted * (1 + (mids_c - 1) * mid_mask)
        
        # Highlight compression
        highlight_mask = torch.pow(x, 2)
        x_adjusted = x_adjusted * (1 + (highlights_c - 1) * highlight_mask)
        
        return torch.clamp(x_adjusted, 0, 1)
    
    def apply_color_grading(self, x):
        """Enhanced color grading with vibrance"""
        # Contrast with S-curve
        contrast_c = torch.clamp(self.contrast, 0.7, 1.5)
        x_contrasted = torch.pow(x, 1.0 / contrast_c)
        
        # Convert to perceived luminance
        gray = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
        
        # Saturation with protection for skin tones
        saturation_c = torch.clamp(self.saturation, 0.5, 1.5)
        x_saturated = gray + saturation_c * (x_contrasted - gray)
        
        # Vibrance (selective saturation)
        vibrance_c = torch.clamp(self.vibrance, -0.3, 0.3)
        saturation_mask = 1.0 - torch.abs(x_saturated - gray).mean(dim=1, keepdim=True)
        x_saturated = x_saturated + vibrance_c * saturation_mask * (x_saturated - gray)
        
        # Warmth adjustment
        warmth_c = torch.clamp(self.warmth, -0.15, 0.15)
        x_final = x_saturated.clone()
        x_final[:, 0, :, :] = x_final[:, 0, :, :] + warmth_c  # Red
        x_final[:, 2, :, :] = x_final[:, 2, :, :] - warmth_c * 0.7  # Blue
        
        return torch.clamp(x_final, 0, 1)
    
    def forward(self, x):
        # Classical pipeline (70-80%)
        x_classical = self.apply_color_matrix(x)
        x_classical = self.apply_cinema_tone_curve(x_classical)
        x_classical = self.apply_color_grading(x_classical)
        
        # ML enhancement (20-30%)
        x_ml = self.ml_enhancement(x_classical)
        
        # Adaptive blending
        blend_ratio = torch.clamp(self.ml_blend, 0.15, 0.35)
        output = (1 - blend_ratio) * x_classical + blend_ratio * x_ml
        
        return torch.clamp(output, 0, 1)

class ImprovedDataset(Dataset):
    """Dataset with higher resolution support"""
    
    def __init__(self, training_data_path: Path, train_size: int = 512, test_size: int = 1024):
        self.data_path = training_data_path
        self.train_size = train_size
        self.test_size = test_size
        self.pairs = self.load_pairs()
    
    def load_pairs(self):
        metadata_file = self.data_path / "depth_metadata.json"
        with open(metadata_file, 'r') as f:
            pairs = json.load(f)
        return pairs
    
    def process_image_multiscale(self, file_path: str, size: int):
        """Process at multiple scales for better learning"""
        try:
            with rawpy.imread(file_path) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    output_bps=16,
                    no_auto_bright=True,
                    user_flip=0
                )
            
            # Higher bit depth preservation
            rgb_norm = rgb.astype(np.float32) / 65535.0
            
            # Multi-scale processing
            h, w = rgb_norm.shape[:2]
            if min(h, w) > size:
                # Random crop for training
                y = np.random.randint(0, h - size)
                x = np.random.randint(0, w - size)
                rgb_crop = rgb_norm[y:y+size, x:x+size]
            else:
                # Resize if too small
                rgb_crop = cv2.resize(rgb_norm, (size, size))
            
            return np.transpose(rgb_crop, (2, 0, 1))
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        iphone_img = self.process_image_multiscale(pair['iphone_file'], self.train_size)
        sony_img = self.process_image_multiscale(pair['sony_file'], self.train_size)
        
        if iphone_img is None or sony_img is None:
            return self.__getitem__((idx + 1) % len(self.pairs))
        
        return {
            'iphone': torch.FloatTensor(iphone_img),
            'sony': torch.FloatTensor(sony_img),
            'pair_id': idx
        }

def perceptual_loss(pred, target):
    """Simple perceptual loss based on gradients"""
    # Edge detection
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    
    pred_edges_x = F.conv2d(pred.mean(dim=1, keepdim=True), sobel_x.to(pred.device), padding=1)
    pred_edges_y = F.conv2d(pred.mean(dim=1, keepdim=True), sobel_y.to(pred.device), padding=1)
    pred_edges = torch.sqrt(pred_edges_x**2 + pred_edges_y**2)
    
    target_edges_x = F.conv2d(target.mean(dim=1, keepdim=True), sobel_x.to(target.device), padding=1)
    target_edges_y = F.conv2d(target.mean(dim=1, keepdim=True), sobel_y.to(target.device), padding=1)
    target_edges = torch.sqrt(target_edges_x**2 + target_edges_y**2)
    
    return F.mse_loss(pred_edges, target_edges)

def color_consistency_loss(pred, target):
    """Ensure color relationships are preserved"""
    # Channel ratios
    pred_rg = pred[:, 0, :, :] / (pred[:, 1, :, :] + 1e-6)
    pred_rb = pred[:, 0, :, :] / (pred[:, 2, :, :] + 1e-6)
    
    target_rg = target[:, 0, :, :] / (target[:, 1, :, :] + 1e-6)
    target_rb = target[:, 0, :, :] / (target[:, 2, :, :] + 1e-6)
    
    return F.l1_loss(pred_rg, target_rg) + F.l1_loss(pred_rb, target_rb)

def train_hybrid_v1_1():
    """Train improved hybrid model"""
    print("🎬 TRAINING HYBRID CINEMA MODEL V1.1")
    print("=" * 50)
    print("Enhanced color science with better parameter ranges")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    data_path = Path("data/results/simple_depth_analysis")
    
    # Higher resolution dataset
    dataset = ImprovedDataset(data_path, train_size=512)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    
    model = HybridCinemaTransformV1_1().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.00001)
    
    print(f"📊 Dataset size: {len(dataset)} pairs")
    print(f"🔧 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"📐 Training resolution: 512x512")
    
    num_epochs = 50
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        losses = {'mse': 0, 'hist': 0, 'percep': 0, 'color': 0}
        num_batches = 0
        
        for batch in dataloader:
            iphone_imgs = batch['iphone'].to(device)
            sony_imgs = batch['sony'].to(device)
            
            optimizer.zero_grad()
            predicted = model(iphone_imgs)
            
            # Multi-component loss
            mse_loss = F.mse_loss(predicted, sony_imgs)
            hist_loss = histogram_loss(predicted, sony_imgs, bins=128)  # More bins
            percep_loss = perceptual_loss(predicted, sony_imgs)
            color_loss = color_consistency_loss(predicted, sony_imgs)
            
            # Balanced weighting
            total_loss_batch = (
                0.4 * mse_loss +      # Primary fidelity
                0.2 * hist_loss +     # Color distribution
                0.2 * percep_loss +   # Edge preservation
                0.2 * color_loss      # Color relationships
            )
            
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += total_loss_batch.item()
            losses['mse'] += mse_loss.item()
            losses['hist'] += hist_loss.item()
            losses['percep'] += percep_loss.item()
            losses['color'] += color_loss.item()
            num_batches += 1
        
        scheduler.step()
        
        avg_loss = total_loss / max(num_batches, 1)
        
        if epoch % 5 == 0:
            print(f"\n📊 Epoch {epoch + 1}/{num_epochs}:")
            print(f"     Total Loss: {avg_loss:.6f}")
            print(f"     Components: MSE={losses['mse']/num_batches:.4f}, Hist={losses['hist']/num_batches:.4f}")
            print(f"     Perceptual={losses['percep']/num_batches:.4f}, Color={losses['color']/num_batches:.4f}")
            
            # Print learned parameters
            with torch.no_grad():
                print(f"     Color Matrix diagonal: {torch.diag(model.color_matrix).cpu().numpy()}")
                print(f"     Grading: Contrast={model.contrast.item():.3f}, Saturation={model.saturation.item():.3f}")
                print(f"     Vibrance={model.vibrance.item():.3f}, Warmth={model.warmth.item():.3f}")
                print(f"     ML Blend: {model.ml_blend.item():.3f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_loss,
                'losses': losses,
                'resolution': 512
            }, Path("data/hybrid_cinema_v1_1.pth"))
    
    print(f"\n✅ Training complete!")
    print(f"🎯 Best loss: {best_loss:.6f}")
    return model

def histogram_loss(pred, target, bins=128):
    """Enhanced histogram loss with more bins"""
    loss = 0
    for c in range(3):
        pred_hist = torch.histc(pred[:, c, :, :], bins=bins, min=0, max=1)
        target_hist = torch.histc(target[:, c, :, :], bins=bins, min=0, max=1)
        
        pred_hist = pred_hist / (pred_hist.sum() + 1e-8)
        target_hist = target_hist / (target_hist.sum() + 1e-8)
        
        loss += F.mse_loss(pred_hist, target_hist)
    
    return loss / 3

def test_model_v1_1():
    """Test the improved model with higher resolution"""
    print("\n🧪 TESTING HYBRID CINEMA MODEL V1.1")
    print("=" * 50)
    
    model_path = Path("data/hybrid_cinema_v1_1.pth")
    if not model_path.exists():
        print("❌ No model found. Train first!")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridCinemaTransformV1_1().to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded (epoch {checkpoint['epoch']})")
    
    # Test on high resolution
    training_pairs_dir = Path("data/training_pairs")
    iphone_files = list(training_pairs_dir.glob("iphone_*.dng"))[:5]
    
    results_dir = Path("data/results/hybrid_v1_1_test")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = ImprovedDataset(Path("data/results/simple_depth_analysis"), test_size=1024)
    
    for iphone_file in iphone_files:
        print(f"\n🎯 Processing: {iphone_file.name}")
        
        try:
            # Process at high resolution
            with rawpy.imread(str(iphone_file)) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    output_bps=16,
                    no_auto_bright=True,
                    user_flip=0
                )
            
            # Keep high resolution (up to 2048)
            rgb_norm = rgb.astype(np.float32) / 65535.0
            h, w = rgb_norm.shape[:2]
            max_dim = 2048
            
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
            
            # Save high quality result
            original_display = (rgb_resized * 255).astype(np.uint8)
            transformed_display = (transformed_np * 255).astype(np.uint8)
            
            # Create comparison
            comparison = np.hstack([
                cv2.cvtColor(original_display, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(transformed_display, cv2.COLOR_RGB2BGR)
            ])
            
            # Save at high quality
            output_path = results_dir / f"hybrid_v1_1_{iphone_file.stem}.jpg"
            cv2.imwrite(str(output_path), comparison, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            print(f"  ✅ Saved: {output_path}")
            print(f"  📐 Resolution: {rgb_resized.shape[1]}x{rgb_resized.shape[0]}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n✅ Test complete! Check: {results_dir}")

def main():
    """Main entry point"""
    print("🎬 HYBRID CINEMA MODEL V1.1")
    print("Enhanced color science for better results")
    print("\nChoose option:")
    print("1. Train improved model")
    print("2. Test improved model")
    print("3. Both")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice in ["1", "3"]:
        train_hybrid_v1_1()
    
    if choice in ["2", "3"]:
        test_model_v1_1()

if __name__ == "__main__":
    main()
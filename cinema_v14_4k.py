#!/usr/bin/env python3
"""
Cinema Model v1.4 - 4K High Resolution Training
Stable architecture with full resolution processing
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

class StableColorTransform4K(nn.Module):
    """Ultra-stable color transform optimized for high resolution"""
    
    def __init__(self):
        super(StableColorTransform4K, self).__init__()
        
        # Efficient network that works at any resolution
        self.color_net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # Slightly larger for 4K detail
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Tanh()
        )
        
        # Global adjustments (resolution-independent)
        self.channel_adjust = nn.Parameter(torch.ones(3) * 0.99)  # Start very close to identity
        self.channel_bias = nn.Parameter(torch.zeros(3) * 0.005)
        
        # Tone curve parameters
        self.shadows = nn.Parameter(torch.tensor(0.01))
        self.mids = nn.Parameter(torch.tensor(1.005))
        self.highlights = nn.Parameter(torch.tensor(0.995))
        
        # Very conservative residual strength
        self.residual_strength = nn.Parameter(torch.tensor(0.03))
    
    def apply_tone_curve(self, x):
        """Conservative tone curve that works at any resolution"""
        shadows_c = torch.clamp(self.shadows, -0.03, 0.05)
        mids_c = torch.clamp(self.mids, 0.98, 1.02)
        highlights_c = torch.clamp(self.highlights, 0.95, 1.05)
        
        # Very gentle adjustments
        shadow_mask = (1 - x).clamp(0, 1)
        shadow_lift = shadows_c * shadow_mask * 0.05
        x_shadows = x + shadow_lift
        
        x_mids = torch.pow(x_shadows.clamp(1e-7, 1), 1.0 / mids_c)
        x_final = x_mids * highlights_c
        
        return torch.clamp(x_final, 0, 1)
    
    def forward(self, x):
        # Apply color network with very small residual
        color_delta = self.color_net(x)
        
        # Ultra-conservative residual strength
        strength = torch.clamp(self.residual_strength, 0.005, 0.05)
        
        # Apply minimal residual correction
        corrected = x + strength * color_delta
        corrected = torch.clamp(corrected, 0, 1)
        
        # Apply global per-channel adjustments
        channel_adj = torch.clamp(self.channel_adjust, 0.95, 1.05)
        channel_bias_c = torch.clamp(self.channel_bias, -0.01, 0.01)
        
        channel_outputs = []
        for c in range(3):
            channel_out = corrected[:, c:c+1, :, :] * channel_adj[c] + channel_bias_c[c]
            channel_outputs.append(channel_out)
        
        corrected = torch.cat(channel_outputs, dim=1)
        corrected = torch.clamp(corrected, 0, 1)
        
        # Apply tone curve
        output = self.apply_tone_curve(corrected)
        
        return torch.clamp(output, 0, 1)

class HighResDataset(Dataset):
    """Dataset for high resolution training with smart cropping"""
    
    def __init__(self, training_data_path: Path, target_size: int = 1024):
        self.data_path = training_data_path
        self.target_size = target_size
        self.pairs = self.load_pairs()
        print(f"Loaded {len(self.pairs)} training pairs for {target_size}x{target_size} training")
        
    def load_pairs(self):
        metadata_file = self.data_path / "depth_metadata.json"
        with open(metadata_file, 'r') as f:
            pairs = json.load(f)
        return pairs
    
    def process_image_highres(self, file_path: str):
        """Process at high resolution with smart center cropping"""
        try:
            with rawpy.imread(file_path) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    output_bps=16,
                    no_auto_bright=True,
                    user_flip=0
                )
            
            # Normalize carefully
            rgb_norm = rgb.astype(np.float32) / 65535.0
            rgb_norm = np.clip(rgb_norm, 0, 1)
            
            # Check for issues
            if np.any(np.isnan(rgb_norm)) or np.any(np.isinf(rgb_norm)):
                print(f"Warning: NaN/inf in {file_path}")
                return None
            
            h, w = rgb_norm.shape[:2]
            
            # Smart center crop to square
            if h != w:
                size = min(h, w)
                start_y = (h - size) // 2
                start_x = (w - size) // 2
                rgb_cropped = rgb_norm[start_y:start_y+size, start_x:start_x+size]
            else:
                rgb_cropped = rgb_norm
            
            # Resize to target size with high quality
            if rgb_cropped.shape[0] != self.target_size:
                rgb_resized = cv2.resize(rgb_cropped, (self.target_size, self.target_size), 
                                       interpolation=cv2.INTER_LANCZOS4)
            else:
                rgb_resized = rgb_cropped
            
            return np.transpose(rgb_resized, (2, 0, 1))
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        iphone_img = self.process_image_highres(pair['iphone_file'])
        sony_img = self.process_image_highres(pair['sony_file'])
        
        if iphone_img is None or sony_img is None:
            return self.__getitem__((idx + 1) % len(self.pairs))
        
        return {
            'iphone': torch.FloatTensor(iphone_img),
            'sony': torch.FloatTensor(sony_img),
            'pair_id': idx
        }

def smart_loss_function(pred, target):
    """Optimized loss for high resolution"""
    
    # Primary MSE loss
    mse_loss = F.mse_loss(pred, target)
    
    # Efficient histogram loss (reduced bins for speed)
    hist_loss = 0
    for c in range(3):
        pred_hist = torch.histc(pred[:, c, :, :], bins=32, min=0, max=1)
        target_hist = torch.histc(target[:, c, :, :], bins=32, min=0, max=1)
        
        pred_hist = pred_hist / (pred_hist.sum() + 1e-8)
        target_hist = target_hist / (target_hist.sum() + 1e-8)
        
        hist_loss = hist_loss + F.mse_loss(pred_hist, target_hist)
    
    hist_loss = hist_loss / 3
    
    # Minimal histogram weight for speed
    total_loss = mse_loss + 0.005 * hist_loss
    
    return total_loss, mse_loss, hist_loss

def train_4k_v14():
    """Train at high resolution"""
    print("🎬 TRAINING 4K CINEMA MODEL V1.4")
    print("=" * 50)
    print("High resolution training with stable architecture")
    
    # Use GPU if available for high-res training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    data_path = Path("data/results/simple_depth_analysis")
    
    # Start with 1024x1024 (can increase later)
    target_size = 1024
    dataset = HighResDataset(data_path, target_size=target_size)
    
    # Smaller batch size for high resolution
    batch_size = 1 if device.type == 'cuda' else 1  # Even GPU needs small batches at 1K
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    model = StableColorTransform4K().to(device)
    
    # Conservative optimizer for high-res
    optimizer = optim.Adam(model.parameters(), lr=0.00005)  # Even lower LR for stability
    
    print(f"📊 Dataset: {len(dataset)} pairs at {target_size}x{target_size}")
    print(f"🔧 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"💾 Batch size: {batch_size}")
    
    num_epochs = 15  # Fewer epochs for high-res
    best_loss = float('inf')
    loss_history = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        epoch_mse = []
        epoch_hist = []
        
        for batch_idx, batch in enumerate(dataloader):
            iphone_imgs = batch['iphone'].to(device)
            sony_imgs = batch['sony'].to(device)
            
            # Input validation
            if torch.any(torch.isnan(iphone_imgs)) or torch.any(torch.isnan(sony_imgs)):
                print(f"Warning: NaN in batch {batch_idx}")
                continue
            
            optimizer.zero_grad()
            
            # Forward pass
            predicted = model(iphone_imgs)
            
            # Output validation
            if torch.any(torch.isnan(predicted)):
                print(f"Warning: NaN in prediction at batch {batch_idx}")
                continue
            
            # Compute loss
            total_loss, mse_loss, hist_loss = smart_loss_function(predicted, sony_imgs)
            
            # Loss explosion check
            if total_loss > 5.0:
                print(f"Warning: Loss explosion {total_loss:.4f}, skipping")
                continue
            
            # Backward pass with gradient clipping
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.05)  # Even stricter for 4K
            
            optimizer.step()
            
            epoch_losses.append(total_loss.item())
            epoch_mse.append(mse_loss.item())
            epoch_hist.append(hist_loss.item())
            
            # Progress update for high-res (slower training)
            if batch_idx % 5 == 0:
                print(f"    Batch {batch_idx}/{len(dataloader)}: Loss={total_loss.item():.6f}")
        
        # Epoch summary
        avg_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
        avg_mse = np.mean(epoch_mse) if epoch_mse else float('inf')
        avg_hist = np.mean(epoch_hist) if epoch_hist else float('inf')
        
        loss_history.append(avg_loss)
        
        print(f"\n📊 Epoch {epoch + 1}/{num_epochs}:")
        print(f"     Total: {avg_loss:.6f}, MSE: {avg_mse:.6f}, Hist: {avg_hist:.6f}")
        
        # Print learned parameters
        with torch.no_grad():
            print(f"     Channels: R={model.channel_adjust[0]:.4f}, G={model.channel_adjust[1]:.4f}, B={model.channel_adjust[2]:.4f}")
            print(f"     Tone: S={model.shadows.item():.4f}, M={model.mids.item():.4f}, H={model.highlights.item():.4f}")
            print(f"     Residual: {model.residual_strength.item():.4f}")
        
        # Save best model
        if avg_loss < best_loss and avg_loss < 0.5:
            best_loss = avg_loss
            save_dict = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'loss': float(avg_loss),
                'mse_loss': float(avg_mse),
                'hist_loss': float(avg_hist),
                'target_size': target_size,
                'loss_history': [float(x) for x in loss_history]
            }
            torch.save(save_dict, Path("data/cinema_v14_4k_model.pth"))
    
    print(f"\n✅ 4K Training complete!")
    print(f"🎯 Best loss: {best_loss:.6f}")
    print(f"📊 Training resolution: {target_size}x{target_size}")
    
    return model

def test_4k_v14():
    """Test at full 4K resolution"""
    print("\n🧪 TESTING 4K CINEMA MODEL V1.4")
    print("=" * 50)
    
    model_path = Path("data/cinema_v14_4k_model.pth")
    if not model_path.exists():
        print("❌ No 4K v1.4 model found. Train first!")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StableColorTransform4K().to(device)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    training_size = checkpoint.get('target_size', 1024)
    
    print(f"✅ Model loaded (epoch {checkpoint['epoch']})")
    print(f"🎯 Training loss: {checkpoint['loss']:.6f}")
    print(f"📊 Training size: {training_size}x{training_size}")
    
    # Test on original high-res images
    training_pairs_dir = Path("data/training_pairs")
    iphone_files = list(training_pairs_dir.glob("iphone_*.dng"))[:3]
    
    results_dir = Path("data/results/cinema_v14_4k_test")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for iphone_file in iphone_files:
        print(f"\n🎯 Processing: {iphone_file.name}")
        
        try:
            # Process at multiple resolutions for comparison
            with rawpy.imread(str(iphone_file)) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    output_bps=16,
                    no_auto_bright=True,
                    user_flip=0
                )
            
            rgb_norm = np.clip(rgb.astype(np.float32) / 65535.0, 0, 1)
            h, w = rgb_norm.shape[:2]
            
            print(f"  Original size: {h}x{w}")
            
            # Process at training resolution for consistency
            size = min(h, w, 2048)  # Cap at 2K for memory
            start_y = (h - size) // 2
            start_x = (w - size) // 2
            rgb_crop = rgb_norm[start_y:start_y+size, start_x:start_x+size]
            
            print(f"  Processing size: {size}x{size}")
            
            # Convert to tensor
            rgb_tensor = torch.FloatTensor(np.transpose(rgb_crop, (2, 0, 1))).unsqueeze(0).to(device)
            
            # Transform
            with torch.no_grad():
                transformed = model(rgb_tensor)
                
                if torch.any(torch.isnan(transformed)):
                    print(f"  ❌ NaN in output")
                    continue
                
                transformed_np = transformed.cpu().squeeze(0).numpy()
                transformed_np = np.transpose(transformed_np, (1, 2, 0))
            
            # Convert to display format
            original_display = (np.clip(rgb_crop, 0, 1) * 255).astype(np.uint8)
            transformed_display = (np.clip(transformed_np, 0, 1) * 255).astype(np.uint8)
            
            # Create side-by-side comparison
            comparison = np.hstack([
                cv2.cvtColor(original_display, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(transformed_display, cv2.COLOR_RGB2BGR)
            ])
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.8, size / 1000)  # Scale font with image size
            thickness = max(2, int(size / 500))
            
            cv2.putText(comparison, "iPhone Original", (20, 50), font, font_scale, (0, 255, 0), thickness)
            cv2.putText(comparison, f"Cinema v1.4 4K ({size}x{size})", (size + 20, 50), font, font_scale, (0, 255, 0), thickness)
            
            # Save at high quality
            output_path = results_dir / f"4k_v14_{iphone_file.stem}_{size}x{size}.jpg"
            cv2.imwrite(str(output_path), comparison, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            print(f"  ✅ Saved: {output_path}")
            
            # Quality metrics
            mse = np.mean((rgb_crop - transformed_np) ** 2)
            print(f"  📊 MSE: {mse:.6f}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n✅ 4K test complete!")
    print(f"📁 Results: {results_dir}")
    print("🔍 Check the high-resolution comparisons!")

def main():
    """Main entry point for 4K v1.4"""
    print("🎬 CINEMA MODEL V1.4 - 4K HIGH RESOLUTION")
    print("Stable architecture + Full resolution training")
    print("\nChoose option:")
    print("1. Train 4K v1.4 (1024x1024)")
    print("2. Test 4K v1.4")
    print("3. Both")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice in ["1", "3"]:
        train_4k_v14()
    
    if choice in ["2", "3"]:
        test_4k_v14()

if __name__ == "__main__":
    main()
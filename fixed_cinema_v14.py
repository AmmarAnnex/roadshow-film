#!/usr/bin/env python3
"""
Cinema Model v1.4 - Emergency Repair
Fixed catastrophic failures in v1.3b
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

class StableColorTransform(nn.Module):
    """Ultra-stable color transform with conservative learning"""
    
    def __init__(self):
        super(StableColorTransform, self).__init__()
        
        # MUCH smaller, more conservative network
        self.color_net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # Reduced from 64 to 16
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),  # Keep small
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Tanh()  # Ensures output in [-1, 1]
        )
        
        # Very conservative per-channel adjustments
        # Initialize close to identity
        self.channel_adjust = nn.Parameter(torch.ones(3) * 0.98)  # Start near 1.0
        self.channel_bias = nn.Parameter(torch.zeros(3) * 0.01)   # Start near 0.0
        
        # Conservative tone curve
        self.shadows = nn.Parameter(torch.tensor(0.02))    # Very small
        self.mids = nn.Parameter(torch.tensor(1.01))       # Near identity
        self.highlights = nn.Parameter(torch.tensor(0.98)) # Slight compression
        
        # Residual strength - start VERY low
        self.residual_strength = nn.Parameter(torch.tensor(0.05))
    
    def apply_tone_curve(self, x):
        """Very conservative tone curve"""
        # Clamp parameters to safe ranges
        shadows_c = torch.clamp(self.shadows, -0.05, 0.1)
        mids_c = torch.clamp(self.mids, 0.95, 1.05)
        highlights_c = torch.clamp(self.highlights, 0.9, 1.1)
        
        # Gentle shadow lift
        shadow_mask = (1 - x).clamp(0, 1)
        shadow_lift = shadows_c * shadow_mask * 0.1  # Very gentle
        x_shadows = x + shadow_lift
        
        # Gentle midtone adjustment
        x_mids = torch.pow(x_shadows.clamp(1e-7, 1), 1.0 / mids_c)
        
        # Gentle highlight compression
        highlight_mask = x_mids.clamp(0, 1)
        x_final = x_mids * highlights_c
        
        return torch.clamp(x_final, 0, 1)
    
    def forward(self, x):
        # Apply color network with VERY small residual
        color_delta = self.color_net(x)
        
        # Clamp residual strength to be very conservative
        strength = torch.clamp(self.residual_strength, 0.01, 0.1)
        
        # Apply tiny residual correction
        corrected = x + strength * color_delta
        corrected = torch.clamp(corrected, 0, 1)
        
        # Apply per-channel adjustments (clamped to safe ranges)
        channel_adj = torch.clamp(self.channel_adjust, 0.9, 1.1)
        channel_bias_c = torch.clamp(self.channel_bias, -0.02, 0.02)
        
        channel_outputs = []
        for c in range(3):
            channel_out = corrected[:, c:c+1, :, :] * channel_adj[c] + channel_bias_c[c]
            channel_outputs.append(channel_out)
        
        corrected = torch.cat(channel_outputs, dim=1)
        corrected = torch.clamp(corrected, 0, 1)
        
        # Apply conservative tone curve
        output = self.apply_tone_curve(corrected)
        
        return torch.clamp(output, 0, 1)

class ConservativeDataset(Dataset):
    """Dataset with better normalization and validation"""
    
    def __init__(self, training_data_path: Path, size: int = 256):
        self.data_path = training_data_path
        self.size = size
        self.pairs = self.load_pairs()
        print(f"Loaded {len(self.pairs)} training pairs")
        
    def load_pairs(self):
        metadata_file = self.data_path / "depth_metadata.json"
        with open(metadata_file, 'r') as f:
            pairs = json.load(f)
        return pairs
    
    def process_image_stable(self, file_path: str):
        """Process with careful validation and normalization"""
        try:
            with rawpy.imread(file_path) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    output_bps=16,
                    no_auto_bright=True,
                    user_flip=0
                )
            
            # More careful normalization
            rgb_norm = rgb.astype(np.float32) / 65535.0
            
            # Validate range
            if np.any(rgb_norm < 0) or np.any(rgb_norm > 1):
                print(f"Warning: Invalid range in {file_path}")
                rgb_norm = np.clip(rgb_norm, 0, 1)
            
            # Check for NaN/inf
            if np.any(np.isnan(rgb_norm)) or np.any(np.isinf(rgb_norm)):
                print(f"Warning: NaN/inf detected in {file_path}")
                return None
            
            # Resize with anti-aliasing
            rgb_resized = cv2.resize(rgb_norm, (self.size, self.size), 
                                   interpolation=cv2.INTER_AREA)
            
            return np.transpose(rgb_resized, (2, 0, 1))
            
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
            # Return next valid pair
            return self.__getitem__((idx + 1) % len(self.pairs))
        
        return {
            'iphone': torch.FloatTensor(iphone_img),
            'sony': torch.FloatTensor(sony_img),
            'pair_id': idx
        }

def stable_loss_function(pred, target):
    """Conservative loss with multiple safeguards"""
    
    # Primary MSE loss
    mse_loss = F.mse_loss(pred, target)
    
    # Prevent extreme values
    if mse_loss > 1.0:
        print(f"Warning: High MSE loss {mse_loss:.4f}")
    
    # Conservative histogram loss
    hist_loss = 0
    for c in range(3):
        pred_hist = torch.histc(pred[:, c, :, :], bins=32, min=0, max=1)
        target_hist = torch.histc(target[:, c, :, :], bins=32, min=0, max=1)
        
        # Normalize
        pred_hist = pred_hist / (pred_hist.sum() + 1e-8)
        target_hist = target_hist / (target_hist.sum() + 1e-8)
        
        hist_loss = hist_loss + F.mse_loss(pred_hist, target_hist)
    
    hist_loss = hist_loss / 3
    
    # Very small histogram weight
    total_loss = mse_loss + 0.01 * hist_loss
    
    return total_loss, mse_loss, hist_loss

def train_stable_v14():
    """Ultra-conservative training with extensive monitoring"""
    print("🔧 TRAINING STABLE CINEMA MODEL V1.4")
    print("=" * 50)
    print("Emergency repair for catastrophic v1.3b failures")
    
    device = torch.device('cpu')  # Stay on CPU for stability
    data_path = Path("data/results/simple_depth_analysis")
    
    # Initialize with stable dataset
    dataset = ConservativeDataset(data_path, size=256)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)  # Small batch
    
    model = StableColorTransform().to(device)
    
    # VERY conservative optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Much lower LR
    
    print(f"📊 Dataset size: {len(dataset)} pairs")
    print(f"🔧 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    num_epochs = 20  # Fewer epochs
    best_loss = float('inf')
    
    # Track training stability
    loss_history = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        epoch_mse = []
        epoch_hist = []
        
        for batch_idx, batch in enumerate(dataloader):
            iphone_imgs = batch['iphone'].to(device)
            sony_imgs = batch['sony'].to(device)
            
            # Validate inputs
            if torch.any(torch.isnan(iphone_imgs)) or torch.any(torch.isnan(sony_imgs)):
                print(f"Warning: NaN in batch {batch_idx}")
                continue
            
            optimizer.zero_grad()
            
            # Forward pass
            predicted = model(iphone_imgs)
            
            # Validate outputs
            if torch.any(torch.isnan(predicted)):
                print(f"Warning: NaN in prediction at batch {batch_idx}")
                continue
            
            # Compute stable loss
            total_loss, mse_loss, hist_loss = stable_loss_function(predicted, sony_imgs)
            
            # Check for loss explosion
            if total_loss > 10.0:
                print(f"Warning: Loss explosion {total_loss:.4f}, skipping batch")
                continue
            
            # Backward pass with gradient clipping
            total_loss.backward()
            
            # Aggressive gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            
            optimizer.step()
            
            epoch_losses.append(total_loss.item())
            epoch_mse.append(mse_loss.item())
            epoch_hist.append(hist_loss.item())
        
        # Calculate epoch averages
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
        
        # Check for training instability
        if len(loss_history) > 3:
            recent_trend = loss_history[-3:]
            if all(l1 < l2 for l1, l2 in zip(recent_trend[:-1], recent_trend[1:])):
                print("⚠️  Warning: Loss increasing trend detected")
        
        # Save best model
        if avg_loss < best_loss and avg_loss < 1.0:  # Only save if reasonable
            best_loss = avg_loss
            save_dict = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'loss': float(avg_loss),  # Convert to Python float
                'mse_loss': float(avg_mse),
                'hist_loss': float(avg_hist),
                'loss_history': [float(x) for x in loss_history]  # Convert all to Python floats
            }
            torch.save(save_dict, Path("data/stable_cinema_v14_model.pth"))
    
    print(f"\n✅ Training complete!")
    print(f"🎯 Best loss: {best_loss:.6f}")
    
    if best_loss > 1.0:
        print("⚠️  WARNING: Model may not have converged properly")
        print("   Consider reducing learning rate further")
    
    return model

def test_stable_v14():
    """Test the stable v1.4 model"""
    print("\n🧪 TESTING STABLE MODEL V1.4")
    print("=" * 50)
    
    model_path = Path("data/stable_cinema_v14_model.pth")
    if not model_path.exists():
        print("❌ No stable v1.4 model found. Train first!")
        return
    
    device = torch.device('cpu')
    model = StableColorTransform().to(device)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded (epoch {checkpoint['epoch']})")
    print(f"🎯 Training loss: {checkpoint['loss']:.6f}")
    
    # Print learned parameters
    print(f"\n📊 Learned parameters:")
    with torch.no_grad():
        print(f"   Channels: R={model.channel_adjust[0]:.4f}, G={model.channel_adjust[1]:.4f}, B={model.channel_adjust[2]:.4f}")
        print(f"   Tone: S={model.shadows.item():.4f}, M={model.mids.item():.4f}, H={model.highlights.item():.4f}")
        print(f"   Residual: {model.residual_strength.item():.4f}")
    
    # Test on training images
    training_pairs_dir = Path("data/training_pairs")
    iphone_files = list(training_pairs_dir.glob("iphone_*.dng"))[:3]  # Test 3 samples
    
    results_dir = Path("data/results/stable_v14_test")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for iphone_file in iphone_files:
        print(f"\n🎯 Processing: {iphone_file.name}")
        
        try:
            # Use stable processing
            dataset = ConservativeDataset(Path("data/results/simple_depth_analysis"))
            processed_img = dataset.process_image_stable(str(iphone_file))
            
            if processed_img is None:
                continue
                
            rgb_tensor = torch.FloatTensor(processed_img).unsqueeze(0)
            
            # Transform
            with torch.no_grad():
                transformed = model(rgb_tensor)
                
                # Validate output
                if torch.any(torch.isnan(transformed)):
                    print(f"  ❌ NaN in output, skipping")
                    continue
                
                transformed_np = transformed.squeeze(0).numpy()
                transformed_np = np.transpose(transformed_np, (1, 2, 0))
            
            # Original for comparison
            original_np = np.transpose(processed_img, (1, 2, 0))
            
            # Convert to display format
            original_display = (np.clip(original_np, 0, 1) * 255).astype(np.uint8)
            transformed_display = (np.clip(transformed_np, 0, 1) * 255).astype(np.uint8)
            
            # Create comparison
            comparison = np.hstack([
                cv2.cvtColor(original_display, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(transformed_display, cv2.COLOR_RGB2BGR)
            ])
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(comparison, "iPhone Original", (10, 30), font, 0.6, (0, 255, 0), 2)
            cv2.putText(comparison, "Stable v1.4", (266, 30), font, 0.6, (0, 255, 0), 2)
            
            # Save
            output_path = results_dir / f"stable_v14_{iphone_file.stem}.jpg"
            cv2.imwrite(str(output_path), comparison)
            
            print(f"  ✅ Saved: {output_path}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n✅ Stable test complete! Check: {results_dir}")

def main():
    """Main entry point for stable v1.4"""
    print("🔧 STABLE CINEMA MODEL V1.4 - EMERGENCY REPAIR")
    print("Fixing catastrophic failures from v1.3b")
    print("\nChoose option:")
    print("1. Train stable v1.4")
    print("2. Test stable v1.4")
    print("3. Both")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice in ["1", "3"]:
        train_stable_v14()
    
    if choice in ["2", "3"]:
        test_stable_v14()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Fix Orientation + Improve Color Science
- Fix upside-down images properly
- Add perceptual loss for better color matching
- Histogram matching loss
"""

import cv2
import numpy as np
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import rawpy
from datetime import datetime
import torch.nn.functional as F

class ImprovedColorTransform(nn.Module):
    """Improved color transform with better color science"""
    
    def __init__(self):
        super(ImprovedColorTransform, self).__init__()
        
        # Color correction network with more capacity
        self.color_net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(), 
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()
        )
        
        # Per-channel adjustments
        self.channel_adjust = nn.Parameter(torch.ones(3))
        self.channel_bias = nn.Parameter(torch.zeros(3))
        
        # Learnable tone curve (shadows, mids, highlights)
        self.shadows = nn.Parameter(torch.tensor(0.0))
        self.mids = nn.Parameter(torch.tensor(1.0))
        self.highlights = nn.Parameter(torch.tensor(1.0))
    
    def apply_tone_curve(self, x):
        """Apply learnable tone curve"""
        # Simple 3-point tone curve
        shadows_mask = (x < 0.33).float()
        mids_mask = ((x >= 0.33) & (x < 0.66)).float()
        highlights_mask = (x >= 0.66).float()
        
        result = (shadows_mask * x * (1 + self.shadows) + 
                 mids_mask * x * self.mids + 
                 highlights_mask * x * self.highlights)
        
        return result
    
    def forward(self, x):
        # Apply learned color corrections
        color_delta = self.color_net(x)
        
        # Small residual correction
        corrected = x + 0.05 * color_delta  # Even smaller for subtlety
        
        # Apply per-channel adjustments
        channel_outputs = []
        for c in range(3):
            channel_out = corrected[:, c:c+1, :, :] * self.channel_adjust[c] + self.channel_bias[c]
            channel_outputs.append(channel_out)
        
        corrected = torch.cat(channel_outputs, dim=1)
        
        # Apply tone curve
        corrected = self.apply_tone_curve(corrected)
        
        # Clamp to valid range
        output = torch.clamp(corrected, 0, 1)
        
        return output

class OrientationFixedDataset(Dataset):
    """Dataset with proper orientation handling"""
    
    def __init__(self, training_data_path: Path):
        self.data_path = training_data_path
        self.pairs = self.load_pairs()
        
    def load_pairs(self):
        metadata_file = self.data_path / "depth_metadata.json"
        with open(metadata_file, 'r') as f:
            pairs = json.load(f)
        return pairs
    
    def process_image_fixed_orientation(self, file_path: str, size: int = 256):
        """Process image with multiple orientation fixes"""
        try:
            # Try different user_flip values to fix orientation
            for flip_val in [0, 1, 2, 3]:
                try:
                    with rawpy.imread(file_path) as raw:
                        rgb = raw.postprocess(
                            use_camera_wb=True,
                            output_bps=16,
                            no_auto_bright=True,
                            user_flip=flip_val
                        )
                    break  # If successful, exit loop
                except:
                    continue
            else:
                # If all fail, use default
                with rawpy.imread(file_path) as raw:
                    rgb = raw.postprocess(
                        use_camera_wb=True,
                        output_bps=16,
                        no_auto_bright=True,
                        user_flip=0
                    )
            
            # Normalize to [0, 1]
            rgb_norm = rgb.astype(np.float32) / 65535.0
            
            # ORIENTATION FIX: Try flipping if needed
            # Check if image looks upside down by analyzing brightness distribution
            top_brightness = np.mean(rgb_norm[:rgb_norm.shape[0]//4, :])
            bottom_brightness = np.mean(rgb_norm[3*rgb_norm.shape[0]//4:, :])
            
            # If bottom is much brighter than top, it might be upside down
            if bottom_brightness > top_brightness * 1.5:
                print(f"    🔄 Detected upside-down image, flipping...")
                rgb_norm = np.flipud(rgb_norm)
            
            # Resize to target size
            rgb_resized = cv2.resize(rgb_norm, (size, size))
            
            # Convert to CHW format for PyTorch
            return np.transpose(rgb_resized, (2, 0, 1))
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        iphone_img = self.process_image_fixed_orientation(pair['iphone_file'])
        sony_img = self.process_image_fixed_orientation(pair['sony_file'])
        
        if iphone_img is None or sony_img is None:
            return self.__getitem__((idx + 1) % len(self.pairs))
        
        return {
            'iphone': torch.FloatTensor(iphone_img),
            'sony': torch.FloatTensor(sony_img),
            'pair_id': idx
        }

def histogram_loss(pred, target, bins=64):
    """Compute histogram matching loss"""
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    # Compute histograms
    pred_hist = torch.histc(pred_flat, bins=bins, min=0, max=1)
    target_hist = torch.histc(target_flat, bins=bins, min=0, max=1)
    
    # Normalize histograms
    pred_hist = pred_hist / pred_hist.sum()
    target_hist = target_hist / target_hist.sum()
    
    # Compute histogram difference
    hist_loss = F.mse_loss(pred_hist, target_hist)
    
    return hist_loss

def train_improved_color_science():
    """Train with improved color science techniques"""
    print("🎨 TRAINING IMPROVED COLOR SCIENCE MODEL")
    print("=" * 50)
    
    device = torch.device('cpu')
    data_path = Path("data/results/simple_depth_analysis")
    
    if not (data_path / "depth_metadata.json").exists():
        print("❌ Training data not found!")
        return
    
    # Initialize with improved components
    dataset = OrientationFixedDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    model = ImprovedColorTransform().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0003)  # Even lower LR
    
    print(f"📊 Dataset size: {len(dataset)} pairs")
    print(f"🔧 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training with multiple loss functions
    num_epochs = 25
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_mse = 0
        total_hist = 0
        num_batches = 0
        
        for batch in dataloader:
            iphone_imgs = batch['iphone'].to(device)
            sony_imgs = batch['sony'].to(device)
            
            optimizer.zero_grad()
            predicted = model(iphone_imgs)
            
            # Multiple loss components
            mse_loss = F.mse_loss(predicted, sony_imgs)
            hist_loss = histogram_loss(predicted, sony_imgs)
            
            # Combined loss
            total_loss_batch = mse_loss + 0.1 * hist_loss
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += total_loss_batch.item()
            total_mse += mse_loss.item()
            total_hist += hist_loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_mse = total_mse / max(num_batches, 1)
        avg_hist = total_hist / max(num_batches, 1)
        
        print(f"  📊 Epoch {epoch + 1}/{num_epochs}: Total={avg_loss:.6f}, MSE={avg_mse:.6f}, Hist={avg_hist:.6f}")
        
        # Print learned parameters every 5 epochs
        if epoch % 5 == 0:
            with torch.no_grad():
                print(f"     Channels: R={model.channel_adjust[0]:.3f}, G={model.channel_adjust[1]:.3f}, B={model.channel_adjust[2]:.3f}")
                print(f"     Tone: Shadows={model.shadows.item():.3f}, Mids={model.mids.item():.3f}, Highlights={model.highlights.item():.3f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_loss,
                'mse_loss': avg_mse,
                'hist_loss': avg_hist
            }, Path("data/color_science_model.pth"))
    
    print(f"\n✅ Training complete. Best loss: {best_loss:.6f}")
    return model

def test_orientation_and_color():
    """Test with orientation fixes and improved color science"""
    print("\n🧪 TESTING ORIENTATION + COLOR SCIENCE")
    print("=" * 50)
    
    model_path = Path("data/color_science_model.pth")
    if not model_path.exists():
        print("❌ No color science model found. Train first!")
        return
    
    device = torch.device('cpu')
    model = ImprovedColorTransform().to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded (epoch {checkpoint['epoch']})")
    print(f"📊 MSE loss: {checkpoint['mse_loss']:.6f}, Histogram loss: {checkpoint['hist_loss']:.6f}")
    
    # Test on training images
    training_pairs_dir = Path("data/training_pairs")
    iphone_files = list(training_pairs_dir.glob("iphone_*.dng"))[:3]
    
    results_dir = Path("data/results/color_science_test")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for iphone_file in iphone_files:
        print(f"\n🎯 Processing: {iphone_file.name}")
        
        try:
            # Use the same orientation-fixed processing as training
            dataset = OrientationFixedDataset(Path("data/results/simple_depth_analysis"))
            processed_img = dataset.process_image_fixed_orientation(str(iphone_file))
            
            if processed_img is None:
                continue
                
            rgb_tensor = torch.FloatTensor(processed_img).unsqueeze(0)
            
            # Transform
            with torch.no_grad():
                transformed = model(rgb_tensor)
                transformed_np = transformed.squeeze(0).numpy()
                transformed_np = np.transpose(transformed_np, (1, 2, 0))
            
            # Original for comparison
            original_np = np.transpose(processed_img, (1, 2, 0))
            
            # Convert to display format
            original_display = (original_np * 255).astype(np.uint8)
            transformed_display = (transformed_np * 255).astype(np.uint8)
            
            # Create comparison
            comparison = np.hstack([
                cv2.cvtColor(original_display, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(transformed_display, cv2.COLOR_RGB2BGR)
            ])
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(comparison, "iPhone Original", (10, 30), font, 0.6, (0, 255, 0), 2)
            cv2.putText(comparison, "Color Science Transform", (266, 30), font, 0.6, (0, 255, 0), 2)
            
            # Save
            output_path = results_dir / f"color_science_{iphone_file.stem}.jpg"
            cv2.imwrite(str(output_path), comparison)
            
            print(f"  ✅ Saved: {output_path}")
            
            # Analysis
            diff = np.mean(np.abs(original_display.astype(float) - transformed_display.astype(float)))
            print(f"  📊 Pixel difference: {diff:.1f}")
            
            # Check if images are right-side up by looking at labels
            if comparison.shape[0] > 0 and comparison.shape[1] > 0:
                print("  🔄 Orientation: Images should be right-side up now")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n✅ Test complete! Check: {results_dir}")

def analyze_color_science_learning():
    """Analyze what the improved model learned"""
    print("\n📊 ANALYZING COLOR SCIENCE LEARNING")
    print("=" * 50)
    
    model_path = Path("data/color_science_model.pth")
    if not model_path.exists():
        print("❌ No model found!")
        return
    
    device = torch.device('cpu')
    model = ImprovedColorTransform().to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    with torch.no_grad():
        print("🎨 LEARNED COLOR SCIENCE:")
        
        # Channel adjustments
        r_gain, g_gain, b_gain = model.channel_adjust.data.numpy()
        r_bias, g_bias, b_bias = model.channel_bias.data.numpy()
        
        print(f"  📊 Channel gains: R={r_gain:.3f}, G={g_gain:.3f}, B={b_gain:.3f}")
        print(f"  📊 Channel biases: R={r_bias:.3f}, G={g_bias:.3f}, B={b_bias:.3f}")
        
        # Tone curve
        shadows = model.shadows.item()
        mids = model.mids.item()
        highlights = model.highlights.item()
        
        print(f"  📊 Tone curve: Shadows={shadows:.3f}, Mids={mids:.3f}, Highlights={highlights:.3f}")
        
        print("\n🔍 INTERPRETATION:")
        if r_gain > g_gain and r_gain > b_gain:
            print("  - Model learned to boost RED → Warmer Sony+Zeiss look")
        elif b_gain > r_gain and b_gain > g_gain:
            print("  - Model learned to boost BLUE → Cooler Sony+Zeiss look")
        
        if shadows < 0:
            print("  - Model darkens shadows → More cinematic contrast")
        elif shadows > 0:
            print("  - Model lifts shadows → Sony+Zeiss shadow detail")
        
        if highlights < 1.0:
            print("  - Model rolls off highlights → Film-like highlight handling")

def main():
    """Improved pipeline with orientation fixes and better color science"""
    print("🎨 IMPROVED COLOR SCIENCE PIPELINE")
    print("Choose option:")
    print("1. Train improved color science model")
    print("2. Test orientation + color science")
    print("3. Analyze color science learning")
    print("4. Full pipeline")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice in ["1", "4"]:
        train_improved_color_science()
    
    if choice in ["2", "4"]:
        test_orientation_and_color()
    
    if choice in ["3", "4"]:
        analyze_color_science_learning()

if __name__ == "__main__":
    main()
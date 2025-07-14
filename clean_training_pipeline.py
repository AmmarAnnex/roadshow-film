#!/usr/bin/env python3
"""
Clean Training Pipeline - No unnecessary flipping
For properly oriented images
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
        
        # Color correction network
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
        
        # Learnable tone curve
        self.shadows = nn.Parameter(torch.tensor(0.0))
        self.mids = nn.Parameter(torch.tensor(1.0))
        self.highlights = nn.Parameter(torch.tensor(1.0))
    
    def apply_tone_curve(self, x):
        """Apply learnable tone curve"""
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
        corrected = x + 0.05 * color_delta
        
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

class CleanDataset(Dataset):
    """Clean dataset without unnecessary orientation detection"""
    
    def __init__(self, training_data_path: Path):
        self.data_path = training_data_path
        self.pairs = self.load_pairs()
        
    def load_pairs(self):
        metadata_file = self.data_path / "depth_metadata.json"
        with open(metadata_file, 'r') as f:
            pairs = json.load(f)
        return pairs
    
    def process_image_clean(self, file_path: str, size: int = 256):
        """Process image without orientation detection"""
        try:
            # Simple RAW processing with consistent settings
            with rawpy.imread(file_path) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    output_bps=16,
                    no_auto_bright=True,
                    user_flip=0  # No rotation
                )
            
            # Normalize and resize
            rgb_norm = rgb.astype(np.float32) / 65535.0
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

def histogram_loss(pred, target, bins=64):
    """Compute histogram matching loss"""
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    pred_hist = torch.histc(pred_flat, bins=bins, min=0, max=1)
    target_hist = torch.histc(target_flat, bins=bins, min=0, max=1)
    
    pred_hist = pred_hist / pred_hist.sum()
    target_hist = target_hist / target_hist.sum()
    
    hist_loss = F.mse_loss(pred_hist, target_hist)
    
    return hist_loss

def train_clean_model():
    """Train with clean, fast processing"""
    print("🎨 TRAINING CLEAN COLOR SCIENCE MODEL")
    print("=" * 50)
    
    device = torch.device('cpu')
    data_path = Path("data/results/simple_depth_analysis")
    
    if not (data_path / "depth_metadata.json").exists():
        print("❌ Training data not found!")
        return
    
    # Initialize with clean dataset (no flipping)
    dataset = CleanDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # Larger batch
    
    model = ImprovedColorTransform().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    print(f"📊 Dataset size: {len(dataset)} pairs")
    print(f"🔧 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Faster training
    num_epochs = 30
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
        
        if epoch % 5 == 0:
            print(f"  📊 Epoch {epoch + 1}/{num_epochs}: Total={avg_loss:.6f}, MSE={avg_mse:.6f}, Hist={avg_hist:.6f}")
            
            # Print learned parameters
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
            }, Path("data/clean_color_model.pth"))
    
    print(f"\n✅ Training complete. Best loss: {best_loss:.6f}")
    return model

def test_clean_model():
    """Test the clean model"""
    print("\n🧪 TESTING CLEAN MODEL")
    print("=" * 50)
    
    model_path = Path("data/clean_color_model.pth")
    if not model_path.exists():
        print("❌ No clean model found. Train first!")
        return
    
    device = torch.device('cpu')
    model = ImprovedColorTransform().to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded (epoch {checkpoint['epoch']})")
    
    # Test on training images
    training_pairs_dir = Path("data/training_pairs")
    iphone_files = list(training_pairs_dir.glob("iphone_*.dng"))[:5]  # Test 5 samples
    
    results_dir = Path("data/results/clean_model_test")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for iphone_file in iphone_files:
        print(f"\n🎯 Processing: {iphone_file.name}")
        
        try:
            # Use clean processing (no flipping detection)
            dataset = CleanDataset(Path("data/results/simple_depth_analysis"))
            processed_img = dataset.process_image_clean(str(iphone_file))
            
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
            cv2.putText(comparison, "→ Sony+Zeiss Transform", (266, 30), font, 0.6, (0, 255, 0), 2)
            
            # Save
            output_path = results_dir / f"clean_transform_{iphone_file.stem}.jpg"
            cv2.imwrite(str(output_path), comparison)
            
            print(f"  ✅ Saved: {output_path}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n✅ Clean test complete! Check: {results_dir}")

def main():
    """Clean training pipeline"""
    print("🎨 CLEAN COLOR TRANSFORMATION PIPELINE")
    print("Choose option:")
    print("1. Train clean model (no flipping)")
    print("2. Test clean model")
    print("3. Both")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice in ["1", "3"]:
        train_clean_model()
    
    if choice in ["2", "3"]:
        test_clean_model()

if __name__ == "__main__":
    main()
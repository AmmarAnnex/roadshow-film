#!/usr/bin/env python3
"""
Fix Image Processing Issues
- Fix upside-down images
- Better visualization
- Improved training process
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

class SimpleColorTransform(nn.Module):
    """Fixed simple color transform model"""
    
    def __init__(self):
        super(SimpleColorTransform, self).__init__()
        
        # Simple color correction network
        self.color_net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()
        )
        
        # Per-channel adjustments
        self.channel_adjust = nn.Parameter(torch.ones(3))
        self.channel_bias = nn.Parameter(torch.zeros(3))
        
        # Simple tone curve adjustment
        self.contrast = nn.Parameter(torch.tensor(1.0))
        self.brightness = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, x):
        # Apply learned color corrections
        color_delta = self.color_net(x)
        
        # Small residual correction
        corrected = x + 0.1 * color_delta  # Reduced from 0.2 to 0.1 for subtlety
        
        # Apply per-channel adjustments (fixed - no in-place operations)
        channel_outputs = []
        for c in range(3):
            channel_out = corrected[:, c:c+1, :, :] * self.channel_adjust[c] + self.channel_bias[c]
            channel_outputs.append(channel_out)
        
        corrected = torch.cat(channel_outputs, dim=1)
        
        # Apply tone curve
        corrected = corrected * self.contrast + self.brightness
        
        # Clamp to valid range
        output = torch.clamp(corrected, 0, 1)
        
        return output

class FixedColorDataset(Dataset):
    """Fixed dataset with proper image orientation"""
    
    def __init__(self, training_data_path: Path):
        self.data_path = training_data_path
        self.pairs = self.load_pairs()
        
    def load_pairs(self):
        metadata_file = self.data_path / "depth_metadata.json"
        with open(metadata_file, 'r') as f:
            pairs = json.load(f)
        return pairs
    
    def process_image(self, file_path: str, size: int = 256):
        """Process image with proper orientation"""
        try:
            with rawpy.imread(file_path) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    output_bps=16,
                    no_auto_bright=True,
                    user_flip=0  # No automatic rotation
                )
            
            # Normalize to [0, 1]
            rgb_norm = rgb.astype(np.float32) / 65535.0
            
            # Resize with proper aspect ratio handling
            h, w = rgb_norm.shape[:2]
            if h > w:
                # Portrait - resize based on width
                new_w = size
                new_h = int(size * h / w)
            else:
                # Landscape - resize based on height  
                new_h = size
                new_w = int(size * w / h)
            
            rgb_resized = cv2.resize(rgb_norm, (new_w, new_h))
            
            # Center crop to exact size
            start_y = (new_h - size) // 2 if new_h > size else 0
            start_x = (new_w - size) // 2 if new_w > size else 0
            
            if new_h < size or new_w < size:
                # Pad if image is smaller
                pad_h = max(0, size - new_h)
                pad_w = max(0, size - new_w)
                rgb_resized = np.pad(rgb_resized, 
                                   ((pad_h//2, pad_h - pad_h//2), 
                                    (pad_w//2, pad_w - pad_w//2), 
                                    (0, 0)), 
                                   mode='constant', constant_values=0)
            else:
                # Crop if image is larger
                rgb_resized = rgb_resized[start_y:start_y+size, start_x:start_x+size]
            
            # Convert to CHW format for PyTorch (RGB -> CHW)
            return np.transpose(rgb_resized, (2, 0, 1))
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        iphone_img = self.process_image(pair['iphone_file'])
        sony_img = self.process_image(pair['sony_file'])
        
        if iphone_img is None or sony_img is None:
            return self.__getitem__((idx + 1) % len(self.pairs))
        
        return {
            'iphone': torch.FloatTensor(iphone_img),
            'sony': torch.FloatTensor(sony_img),
            'pair_id': idx
        }

def train_improved_model():
    """Train with better parameters and monitoring"""
    print("🔧 TRAINING IMPROVED COLOR TRANSFORM")
    print("=" * 50)
    
    device = torch.device('cpu')
    data_path = Path("data/results/simple_depth_analysis")
    
    if not (data_path / "depth_metadata.json").exists():
        print("❌ Training data not found!")
        return
    
    # Initialize with fixed dataset
    dataset = FixedColorDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)  # Smaller batch size
    
    model = SimpleColorTransform().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Lower learning rate
    criterion = nn.MSELoss()
    
    print(f"📊 Dataset size: {len(dataset)} pairs")
    print(f"🔧 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training with better monitoring
    num_epochs = 20  # Fewer epochs to start
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            iphone_imgs = batch['iphone'].to(device)
            sony_imgs = batch['sony'].to(device)
            
            optimizer.zero_grad()
            predicted = model(iphone_imgs)
            
            loss = criterion(predicted, sony_imgs)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        print(f"  📊 Epoch {epoch + 1}/{num_epochs}: Loss = {avg_loss:.6f}")
        
        # Print parameter changes to monitor learning
        if epoch % 5 == 0:
            with torch.no_grad():
                print(f"     Channel gains: {model.channel_adjust.data.numpy()}")
                print(f"     Channel biases: {model.channel_bias.data.numpy()}")
                print(f"     Contrast: {model.contrast.item():.3f}")
                print(f"     Brightness: {model.brightness.item():.3f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_loss
            }, Path("data/improved_color_model.pth"))
    
    print(f"\n✅ Training complete. Best loss: {best_loss:.6f}")
    return model

def test_improved_model():
    """Test with fixed image processing"""
    print("\n🧪 TESTING IMPROVED MODEL")
    print("=" * 50)
    
    model_path = Path("data/improved_color_model.pth")
    if not model_path.exists():
        print("❌ No improved model found. Train first!")
        return
    
    device = torch.device('cpu')
    model = SimpleColorTransform().to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded (epoch {checkpoint['epoch']}, loss: {checkpoint['loss']:.6f})")
    
    # Test on training images
    training_pairs_dir = Path("data/training_pairs")
    iphone_files = list(training_pairs_dir.glob("iphone_*.dng"))[:3]
    
    results_dir = Path("data/results/improved_color_test")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for iphone_file in iphone_files:
        print(f"\n🎯 Transforming: {iphone_file.name}")
        
        try:
            # Process iPhone image (same as training)
            with rawpy.imread(str(iphone_file)) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    output_bps=16,
                    no_auto_bright=True,
                    user_flip=0
                )
            
            rgb_norm = rgb.astype(np.float32) / 65535.0
            
            # Resize to 256x256 (same as training)
            rgb_resized = cv2.resize(rgb_norm, (256, 256))
            rgb_tensor = torch.FloatTensor(np.transpose(rgb_resized, (2, 0, 1))).unsqueeze(0)
            
            # Transform
            with torch.no_grad():
                transformed = model(rgb_tensor)
                transformed_np = transformed.squeeze(0).numpy()
                # Convert back: CHW -> HWC
                transformed_np = np.transpose(transformed_np, (1, 2, 0))
            
            # Convert to display format (both should be right-side up now)
            original_display = (rgb_resized * 255).astype(np.uint8)
            transformed_display = (transformed_np * 255).astype(np.uint8)
            
            # Create side-by-side comparison
            comparison = np.hstack([
                cv2.cvtColor(original_display, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(transformed_display, cv2.COLOR_RGB2BGR)
            ])
            
            # Add clear labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(comparison, "Original iPhone", (10, 30), font, 0.6, (0, 255, 0), 2)
            cv2.putText(comparison, "→ Sony+Zeiss Style", (266, 30), font, 0.6, (0, 255, 0), 2)
            
            # Save
            output_path = results_dir / f"improved_transform_{iphone_file.stem}.jpg"
            cv2.imwrite(str(output_path), comparison)
            
            print(f"  ✅ Saved: {output_path}")
            
            # Analyze differences
            diff = np.mean(np.abs(original_display.astype(float) - transformed_display.astype(float)))
            print(f"  📊 Average pixel difference: {diff:.1f}")
            
            # Color analysis
            orig_mean = np.mean(original_display, axis=(0,1))
            trans_mean = np.mean(transformed_display, axis=(0,1))
            print(f"  🎨 Original RGB mean: {orig_mean}")
            print(f"  🎨 Transformed RGB mean: {trans_mean}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n✅ Test complete! Results in: {results_dir}")

def analyze_what_model_learned():
    """Analyze what the model actually learned"""
    print("\n📊 ANALYZING MODEL LEARNING")
    print("=" * 50)
    
    model_path = Path("data/improved_color_model.pth")
    if not model_path.exists():
        print("❌ No model found!")
        return
    
    device = torch.device('cpu')
    model = SimpleColorTransform().to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    with torch.no_grad():
        print("🔍 LEARNED PARAMETERS:")
        print(f"  Channel gains (R,G,B): {model.channel_adjust.data.numpy()}")
        print(f"  Channel biases (R,G,B): {model.channel_bias.data.numpy()}")
        print(f"  Contrast multiplier: {model.contrast.item():.3f}")
        print(f"  Brightness offset: {model.brightness.item():.3f}")
        
        print("\n🎨 WHAT THIS MEANS:")
        gains = model.channel_adjust.data.numpy()
        biases = model.channel_bias.data.numpy()
        
        if gains[0] > gains[1] and gains[0] > gains[2]:
            print("  - Boosting RED channel (warmer look)")
        elif gains[2] > gains[0] and gains[2] > gains[1]:
            print("  - Boosting BLUE channel (cooler look)")
        
        if model.contrast.item() > 1.0:
            print("  - Increasing contrast")
        elif model.contrast.item() < 1.0:
            print("  - Decreasing contrast")
        
        if model.brightness.item() > 0:
            print("  - Making image brighter")
        elif model.brightness.item() < 0:
            print("  - Making image darker")

def main():
    """Improved pipeline with better monitoring"""
    print("🔧 IMPROVED COLOR TRANSFORMATION PIPELINE")
    print("Choose option:")
    print("1. Train improved model")
    print("2. Test improved model")
    print("3. Analyze what model learned")
    print("4. Full pipeline (train + test + analyze)")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice in ["1", "4"]:
        train_improved_model()
    
    if choice in ["2", "4"]:
        test_improved_model()
    
    if choice in ["3", "4"]:
        analyze_what_model_learned()

if __name__ == "__main__":
    main()
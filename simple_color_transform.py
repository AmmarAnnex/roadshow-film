#!/usr/bin/env python3
"""
Simple Color Transform Model
Much simpler approach - just learn color corrections, not full reconstruction
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
    """Simple model that learns color corrections only"""
    
    def __init__(self):
        super(SimpleColorTransform, self).__init__()
        
        # Very simple color correction network
        self.color_net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()  # Output in [-1, 1] range
        )
        
        # Learn per-channel gains and biases
        self.channel_adjust = nn.Parameter(torch.ones(3))
        self.channel_bias = nn.Parameter(torch.zeros(3))
    
    def forward(self, x):
        # Apply learned color corrections
        color_delta = self.color_net(x)
        
        # Small residual correction (max 20% change)
        corrected = x + 0.2 * color_delta
        
        # Apply per-channel adjustments (avoid in-place operations)
        channel_outputs = []
        for c in range(3):
            channel_out = corrected[:, c:c+1, :, :] * self.channel_adjust[c] + self.channel_bias[c]
            channel_outputs.append(channel_out)
        
        # Concatenate channels
        corrected = torch.cat(channel_outputs, dim=1)
        
        # Clamp to valid range
        output = torch.clamp(corrected, 0, 1)
        
        return output

class SimpleColorDataset(Dataset):
    """Simple dataset for color matching"""
    
    def __init__(self, training_data_path: Path):
        self.data_path = training_data_path
        self.pairs = self.load_pairs()
        
    def load_pairs(self):
        metadata_file = self.data_path / "depth_metadata.json"
        with open(metadata_file, 'r') as f:
            pairs = json.load(f)
        return pairs
    
    def process_image(self, file_path: str, size: int = 256):
        """Process image to smaller size for faster training"""
        try:
            with rawpy.imread(file_path) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    output_bps=16,
                    no_auto_bright=True,
                    user_flip=0
                )
            
            # Normalize and resize to smaller size
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
        
        iphone_img = self.process_image(pair['iphone_file'])
        sony_img = self.process_image(pair['sony_file'])
        
        if iphone_img is None or sony_img is None:
            return self.__getitem__((idx + 1) % len(self.pairs))
        
        return {
            'iphone': torch.FloatTensor(iphone_img),
            'sony': torch.FloatTensor(sony_img)
        }

def train_simple_model():
    """Train the simple color transform model"""
    print("🎨 TRAINING SIMPLE COLOR TRANSFORM")
    print("=" * 50)
    
    # Setup
    device = torch.device('cpu')  # Use CPU for simplicity
    data_path = Path("data/results/simple_depth_analysis")
    
    # Check training data
    if not (data_path / "depth_metadata.json").exists():
        print("❌ Training data not found!")
        return
    
    # Initialize
    dataset = SimpleColorDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model = SimpleColorTransform().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print(f"📊 Dataset size: {len(dataset)} pairs")
    print(f"🔧 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            iphone_imgs = batch['iphone'].to(device)
            sony_imgs = batch['sony'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predicted = model(iphone_imgs)
            
            # Simple MSE loss
            loss = criterion(predicted, sony_imgs)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        if epoch % 5 == 0:
            print(f"  📊 Epoch {epoch + 1}/{num_epochs}: Loss = {avg_loss:.6f}")
            
            # Check parameter values
            with torch.no_grad():
                print(f"     Channel gains: {model.channel_adjust.data}")
                print(f"     Channel biases: {model.channel_bias.data}")
    
    # Save model
    save_path = Path("data/simple_color_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': num_epochs,
        'loss': avg_loss
    }, save_path)
    
    print(f"\n✅ Simple model saved: {save_path}")
    return model

def test_simple_model():
    """Test the simple color transform"""
    print("\n🧪 TESTING SIMPLE COLOR TRANSFORM")
    print("=" * 50)
    
    # Load model
    model_path = Path("data/simple_color_model.pth")
    if not model_path.exists():
        print("❌ No simple model found. Train first!")
        return
    
    device = torch.device('cpu')
    model = SimpleColorTransform().to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded (trained for {checkpoint['epoch']} epochs)")
    
    # Test on training images
    training_pairs_dir = Path("data/training_pairs")
    iphone_files = list(training_pairs_dir.glob("iphone_*.dng"))[:3]
    
    results_dir = Path("data/results/simple_color_test")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for iphone_file in iphone_files:
        print(f"\n🎯 Transforming: {iphone_file.name}")
        
        try:
            # Process iPhone image
            with rawpy.imread(str(iphone_file)) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    output_bps=16,
                    no_auto_bright=True,
                    user_flip=0
                )
            
            # Prepare for model
            rgb_norm = rgb.astype(np.float32) / 65535.0
            rgb_small = cv2.resize(rgb_norm, (256, 256))
            rgb_tensor = torch.FloatTensor(np.transpose(rgb_small, (2, 0, 1))).unsqueeze(0)
            
            # Transform
            with torch.no_grad():
                transformed = model(rgb_tensor)
                transformed_np = transformed.squeeze(0).numpy()
                transformed_np = np.transpose(transformed_np, (1, 2, 0))
            
            # Convert to display format
            original_display = (rgb_small * 255).astype(np.uint8)
            transformed_display = (transformed_np * 255).astype(np.uint8)
            
            # Create comparison
            comparison = np.hstack([
                cv2.cvtColor(original_display, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(transformed_display, cv2.COLOR_RGB2BGR)
            ])
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(comparison, "Original iPhone", (10, 30), font, 0.5, (255, 255, 255), 1)
            cv2.putText(comparison, "Simple Color Transform", (266, 30), font, 0.5, (255, 255, 255), 1)
            
            # Save
            output_path = results_dir / f"simple_transform_{iphone_file.stem}.jpg"
            cv2.imwrite(str(output_path), comparison)
            
            print(f"  ✅ Saved: {output_path}")
            
            # Check if transformation is working
            diff = np.mean(np.abs(original_display.astype(float) - transformed_display.astype(float)))
            print(f"  📊 Average difference: {diff:.1f}")
            
            if diff > 5:
                print("  🎯 Visible transformation applied ✅")
            else:
                print("  ⚠️ Minimal transformation detected")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n✅ Test complete! Check results in: {results_dir}")

def main():
    """Main pipeline for simple color transform"""
    print("🎨 SIMPLE COLOR TRANSFORMATION PIPELINE")
    print("Choose option:")
    print("1. Train simple color model")
    print("2. Test simple color model") 
    print("3. Both")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice in ["1", "3"]:
        train_simple_model()
    
    if choice in ["2", "3"]:
        test_simple_model()

if __name__ == "__main__":
    main()
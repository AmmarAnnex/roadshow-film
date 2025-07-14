#!/usr/bin/env python3
"""
Perceptual Loss Cinema Model
Advanced model with VGG-based perceptual loss for high-quality transformations
Building toward genuinely novel ML advancement
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
import torchvision.models as models
import torchvision.transforms as transforms

class VGGPerceptualLoss(nn.Module):
    """VGG-based perceptual loss for preserving visual quality"""
    
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        
        # Use pre-trained VGG19 features
        vgg = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])   # relu1_2
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])  # relu2_2
        self.slice3 = nn.Sequential(*list(vgg.children())[9:18]) # relu3_4
        self.slice4 = nn.Sequential(*list(vgg.children())[18:27])# relu4_4
        
        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x, y):
        """Compute perceptual loss between x and y"""
        # Normalize for VGG (ImageNet preprocessing)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        
        x_norm = (x - mean) / std
        y_norm = (y - mean) / std
        
        # Extract features at multiple levels
        x_relu1_2 = self.slice1(x_norm)
        x_relu2_2 = self.slice2(x_relu1_2)
        x_relu3_4 = self.slice3(x_relu2_2)
        x_relu4_4 = self.slice4(x_relu3_4)
        
        y_relu1_2 = self.slice1(y_norm)
        y_relu2_2 = self.slice2(y_relu1_2)
        y_relu3_4 = self.slice3(y_relu2_2)
        y_relu4_4 = self.slice4(y_relu3_4)
        
        # Compute feature losses at multiple scales
        loss1 = F.mse_loss(x_relu1_2, y_relu1_2)
        loss2 = F.mse_loss(x_relu2_2, y_relu2_2)
        loss3 = F.mse_loss(x_relu3_4, y_relu3_4)
        loss4 = F.mse_loss(x_relu4_4, y_relu4_4)
        
        # Weighted combination (emphasize mid-level features)
        total_loss = 0.1 * loss1 + 0.2 * loss2 + 0.4 * loss3 + 0.3 * loss4
        
        return total_loss

class AdvancedCinemaTransform(nn.Module):
    """Advanced cinema transformation with improved architecture"""
    
    def __init__(self):
        super(AdvancedCinemaTransform, self).__init__()
        
        # Improved color correction network with skip connections
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv5 = nn.Conv2d(16, 3, 3, padding=1)
        
        # Batch normalization for stable training
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(16)
        
        # Per-channel adjustments (learnable color matrix)
        self.color_matrix = nn.Parameter(torch.eye(3))
        self.color_bias = nn.Parameter(torch.zeros(3))
        
        # Advanced tone curve (more control points)
        self.shadows = nn.Parameter(torch.tensor(0.0))
        self.darks = nn.Parameter(torch.tensor(0.0))
        self.mids = nn.Parameter(torch.tensor(1.0))
        self.lights = nn.Parameter(torch.tensor(1.0))
        self.highlights = nn.Parameter(torch.tensor(1.0))
        
        # Contrast and saturation controls
        self.contrast = nn.Parameter(torch.tensor(1.0))
        self.saturation = nn.Parameter(torch.tensor(1.0))
        
        # Edge enhancement (learned detail preservation)
        self.detail_enhance = nn.Parameter(torch.tensor(0.0))
    
    def apply_advanced_tone_curve(self, x):
        """Apply 5-point tone curve for cinematic look"""
        # Create masks for different tonal ranges
        shadows_mask = (x < 0.2).float()
        darks_mask = ((x >= 0.2) & (x < 0.4)).float()
        mids_mask = ((x >= 0.4) & (x < 0.6)).float()
        lights_mask = ((x >= 0.6) & (x < 0.8)).float()
        highlights_mask = (x >= 0.8).float()
        
        # Apply tone adjustments
        result = (shadows_mask * x * (1 + self.shadows) +
                 darks_mask * x * (1 + self.darks) +
                 mids_mask * x * self.mids +
                 lights_mask * x * self.lights +
                 highlights_mask * x * self.highlights)
        
        return result
    
    def apply_color_matrix(self, x):
        """Apply learned color matrix transformation"""
        # Reshape for matrix multiplication
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1)  # [B, 3, H*W]
        
        # Apply color matrix
        transformed = torch.bmm(self.color_matrix.unsqueeze(0).expand(b, -1, -1), x_flat)
        
        # Add bias
        transformed = transformed + self.color_bias.view(1, 3, 1)
        
        # Reshape back
        return transformed.view(b, c, h, w)
    
    def enhance_details(self, x, original):
        """Learned detail enhancement"""
        if self.detail_enhance != 0:
            # Simple unsharp masking
            blur = F.avg_pool2d(x, 3, stride=1, padding=1)
            details = x - blur
            enhanced = x + self.detail_enhance * details
            return enhanced
        return x
    
    def forward(self, x):
        # Store original for skip connection
        identity = x
        
        # Color correction network with skip connections
        h1 = F.relu(self.bn1(self.conv1(x)))
        h2 = F.relu(self.bn2(self.conv2(h1)))
        h3 = F.relu(self.bn3(self.conv3(h2)))
        h4 = F.relu(self.bn4(self.conv4(h3)))
        
        # Output residual correction
        delta = torch.tanh(self.conv5(h4))
        
        # Apply small residual correction (conservative approach)
        corrected = x + 0.02 * delta  # Very small correction
        
        # Apply learned color matrix
        corrected = self.apply_color_matrix(corrected)
        
        # Apply advanced tone curve
        corrected = self.apply_advanced_tone_curve(corrected)
        
        # Apply contrast adjustment
        corrected = corrected * self.contrast
        
        # Apply saturation adjustment (convert to HSV-like)
        gray = 0.299 * corrected[:, 0:1, :, :] + 0.587 * corrected[:, 1:2, :, :] + 0.114 * corrected[:, 2:3, :, :]
        corrected = gray + self.saturation * (corrected - gray)
        
        # Detail enhancement
        corrected = self.enhance_details(corrected, x)
        
        # Clamp to valid range
        output = torch.clamp(corrected, 0, 1)
        
        return output

class PerceptualDataset(Dataset):
    """Dataset with proper preprocessing for perceptual loss"""
    
    def __init__(self, training_data_path: Path):
        self.data_path = training_data_path
        self.pairs = self.load_pairs()
        
    def load_pairs(self):
        metadata_file = self.data_path / "depth_metadata.json"
        with open(metadata_file, 'r') as f:
            pairs = json.load(f)
        return pairs
    
    def process_image_clean(self, file_path: str, size: int = 256):
        """Process image cleanly"""
        try:
            with rawpy.imread(file_path) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    output_bps=16,
                    no_auto_bright=True,
                    user_flip=0
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
    """Histogram matching loss"""
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    pred_hist = torch.histc(pred_flat, bins=bins, min=0, max=1)
    target_hist = torch.histc(target_flat, bins=bins, min=0, max=1)
    
    pred_hist = pred_hist / (pred_hist.sum() + 1e-8)
    target_hist = target_hist / (target_hist.sum() + 1e-8)
    
    return F.mse_loss(pred_hist, target_hist)

def train_perceptual_model():
    """Train with perceptual loss for high quality"""
    print("🎨 TRAINING PERCEPTUAL LOSS CINEMA MODEL")
    print("=" * 50)
    print("Goal: Achieve at least 1 high-quality transformation")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    data_path = Path("data/results/simple_depth_analysis")
    
    if not (data_path / "depth_metadata.json").exists():
        print("❌ Training data not found!")
        return
    
    # Initialize dataset and model
    dataset = PerceptualDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    
    model = AdvancedCinemaTransform().to(device)
    perceptual_loss_fn = VGGPerceptualLoss().to(device)
    
    # Optimizer with different learning rates for different components
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if 'color_matrix' in n], 'lr': 0.0001},
        {'params': [p for n, p in model.named_parameters() if 'color_matrix' not in n], 'lr': 0.0003}
    ]
    optimizer = optim.Adam(param_groups)
    
    print(f"📊 Dataset size: {len(dataset)} pairs")
    print(f"🔧 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training with multiple loss components
    num_epochs = 40
    best_loss = float('inf')
    target_quality_reached = False
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_mse = 0
        total_perceptual = 0
        total_hist = 0
        num_batches = 0
        
        for batch in dataloader:
            iphone_imgs = batch['iphone'].to(device)
            sony_imgs = batch['sony'].to(device)
            
            optimizer.zero_grad()
            predicted = model(iphone_imgs)
            
            # Multiple loss components (DPED-style)
            mse_loss = F.mse_loss(predicted, sony_imgs)
            perceptual_loss = perceptual_loss_fn(predicted, sony_imgs)
            hist_loss = histogram_loss(predicted, sony_imgs)
            
            # Weighted combination (emphasize perceptual quality)
            total_loss_batch = (0.3 * mse_loss + 
                              0.5 * perceptual_loss + 
                              0.2 * hist_loss)
            
            total_loss_batch.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += total_loss_batch.item()
            total_mse += mse_loss.item()
            total_perceptual += perceptual_loss.item()
            total_hist += hist_loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_mse = total_mse / max(num_batches, 1)
        avg_perceptual = total_perceptual / max(num_batches, 1)
        avg_hist = total_hist / max(num_batches, 1)
        
        # Print progress every 5 epochs
        if epoch % 5 == 0:
            print(f"\n📊 Epoch {epoch + 1}/{num_epochs}:")
            print(f"     Total: {avg_loss:.6f}, MSE: {avg_mse:.6f}")
            print(f"     Perceptual: {avg_perceptual:.6f}, Hist: {avg_hist:.6f}")
            
            # Print learned parameters
            with torch.no_grad():
                print(f"     Color Matrix diagonal: {torch.diag(model.color_matrix).data.cpu().numpy()}")
                print(f"     Tone: S={model.shadows.item():.3f}, M={model.mids.item():.3f}, H={model.highlights.item():.3f}")
                print(f"     Contrast: {model.contrast.item():.3f}, Saturation: {model.saturation.item():.3f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_loss,
                'mse_loss': avg_mse,
                'perceptual_loss': avg_perceptual,
                'hist_loss': avg_hist
            }, Path("data/perceptual_cinema_model.pth"))
            
            # Check if we've reached target quality
            if avg_perceptual < 0.1 and avg_mse < 0.02:
                print(f"🎯 TARGET QUALITY REACHED at epoch {epoch + 1}!")
                target_quality_reached = True
    
    print(f"\n✅ Training complete!")
    print(f"🎯 Best loss: {best_loss:.6f}")
    if target_quality_reached:
        print("🏆 HIGH QUALITY TARGET ACHIEVED!")
    else:
        print("📈 Continue training or adjust architecture for target quality")
    
    return model

def test_perceptual_model():
    """Test the perceptual model"""
    print("\n🧪 TESTING PERCEPTUAL CINEMA MODEL")
    print("=" * 50)
    
    model_path = Path("data/perceptual_cinema_model.pth")
    if not model_path.exists():
        print("❌ No perceptual model found. Train first!")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AdvancedCinemaTransform().to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded (epoch {checkpoint['epoch']})")
    print(f"📊 Perceptual loss: {checkpoint['perceptual_loss']:.6f}")
    
    # Test on training images
    training_pairs_dir = Path("data/training_pairs")
    iphone_files = list(training_pairs_dir.glob("iphone_*.dng"))[:5]
    
    results_dir = Path("data/results/perceptual_test")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for iphone_file in iphone_files:
        print(f"\n🎯 Processing: {iphone_file.name}")
        
        try:
            dataset = PerceptualDataset(Path("data/results/simple_depth_analysis"))
            processed_img = dataset.process_image_clean(str(iphone_file))
            
            if processed_img is None:
                continue
                
            rgb_tensor = torch.FloatTensor(processed_img).unsqueeze(0).to(device)
            
            # Transform
            with torch.no_grad():
                transformed = model(rgb_tensor)
                transformed_np = transformed.cpu().squeeze(0).numpy()
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
            cv2.putText(comparison, "Perceptual Cinema Transform", (266, 30), font, 0.6, (0, 255, 0), 2)
            
            # Save
            output_path = results_dir / f"perceptual_transform_{iphone_file.stem}.jpg"
            cv2.imwrite(str(output_path), comparison)
            
            print(f"  ✅ Saved: {output_path}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n✅ Perceptual test complete! Check: {results_dir}")

def main():
    """Main perceptual training pipeline"""
    print("🎬 PERCEPTUAL LOSS CINEMA PIPELINE")
    print("Building toward genuinely novel ML advancement")
    print("Goal: At least 1 high-quality transformation")
    print("\nChoose option:")
    print("1. Train perceptual model")
    print("2. Test perceptual model")
    print("3. Both")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice in ["1", "3"]:
        train_perceptual_model()
    
    if choice in ["2", "3"]:
        test_perceptual_model()

if __name__ == "__main__":
    main()
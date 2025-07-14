#!/usr/bin/env python3
"""
Cinema Model v2 - Enhanced for Better Color and Vibrancy
Addresses washed-out look with stronger transformations
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

class EnhancedColorNet(nn.Module):
    """Enhanced ML component with stronger color learning"""
    
    def __init__(self):
        super(EnhancedColorNet, self).__init__()
        
        # Deeper network for better color understanding
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Separate pathways for color and detail
        self.color_path = nn.Conv2d(32, 3, 3, padding=1)
        self.detail_path = nn.Conv2d(32, 3, 3, padding=1)
        
        # Stronger initial residual
        self.color_strength = nn.Parameter(torch.tensor(0.3))
        self.detail_strength = nn.Parameter(torch.tensor(0.15))
    
    def forward(self, x):
        features = self.features(x)
        
        # Color enhancement (can be stronger)
        color_enhance = torch.tanh(self.color_path(features))
        
        # Detail enhancement (more conservative)
        detail_enhance = torch.tanh(self.detail_path(features))
        
        # Combine with different strengths
        color_strength = torch.clamp(self.color_strength, 0.1, 0.5)
        detail_strength = torch.clamp(self.detail_strength, 0.05, 0.2)
        
        output = x + color_strength * color_enhance + detail_strength * detail_enhance
        
        return torch.clamp(output, 0, 1)

class CinemaTransformV2(nn.Module):
    """Cinema transformation with enhanced color capabilities"""
    
    def __init__(self):
        super(CinemaTransformV2, self).__init__()
        
        # Start with more aggressive defaults
        self.color_matrix = nn.Parameter(torch.tensor([
            [1.05, 0.05, 0.00],
            [0.00, 1.02, 0.00],
            [0.00, 0.02, 1.08]
        ]))
        self.color_bias = nn.Parameter(torch.zeros(3) * 0.02)
        
        # More aggressive tone curve
        self.shadows = nn.Parameter(torch.tensor(0.1))
        self.mids = nn.Parameter(torch.tensor(1.05))
        self.highlights = nn.Parameter(torch.tensor(0.95))
        
        # Stronger color grading defaults
        self.contrast = nn.Parameter(torch.tensor(1.1))
        self.saturation = nn.Parameter(torch.tensor(1.3))
        self.vibrance = nn.Parameter(torch.tensor(0.2))
        self.warmth = nn.Parameter(torch.tensor(0.05))
        
        # Cinema-specific parameters
        self.highlight_rolloff = nn.Parameter(torch.tensor(0.85))
        self.shadow_lift = nn.Parameter(torch.tensor(0.05))
        
        # Enhanced ML component
        self.ml_enhancement = EnhancedColorNet()
        
        # Higher ML contribution
        self.ml_blend = nn.Parameter(torch.tensor(0.35))
    
    def apply_cinematic_tone_curve(self, x):
        """Film-like S-curve with better highlight rolloff"""
        # Lift shadows
        x = x + self.shadow_lift * (1 - x) * torch.exp(-x * 4)
        
        # S-curve for midtones
        x_mid = torch.pow(x, 1.0 / self.mids)
        
        # Smooth highlight rolloff (film-like)
        highlight_mask = torch.clamp((x - 0.5) * 2, 0, 1)
        x_highlights = 1 - torch.pow(1 - x, 2.0 * self.highlight_rolloff)
        
        # Blend based on luminance
        x_toned = x_mid * (1 - highlight_mask) + x_highlights * highlight_mask
        
        # Shadow adjustment
        shadow_mask = torch.clamp((0.3 - x) * 3, 0, 1)
        x_final = x_toned + self.shadows * shadow_mask * x_toned
        
        return torch.clamp(x_final, 0, 1)
    
    def apply_film_color_matrix(self, x):
        """Film-like color matrix transformation"""
        # Allow more aggressive transforms
        matrix = torch.clamp(self.color_matrix, 0.5, 1.5)
        bias = torch.clamp(self.color_bias, -0.1, 0.1)
        
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1)
        
        # Apply matrix
        transformed = torch.bmm(matrix.unsqueeze(0).expand(b, -1, -1), x_flat)
        transformed = transformed + bias.view(1, 3, 1)
        
        return transformed.view(b, c, h, w)
    
    def apply_cinematic_color_grading(self, x):
        """Film-style color grading with enhanced saturation"""
        # More aggressive contrast
        contrast_c = torch.clamp(self.contrast, 0.8, 1.4)
        x_contrasted = torch.pow(torch.clamp(x, 1e-6, 1.0), 1.0 / contrast_c)
        
        # Calculate luminance
        luma = 0.299 * x_contrasted[:, 0:1, :, :] + \
               0.587 * x_contrasted[:, 1:2, :, :] + \
               0.114 * x_contrasted[:, 2:3, :, :]
        
        # Enhanced saturation (key for avoiding washed-out look)
        saturation_c = torch.clamp(self.saturation, 0.8, 1.6)
        x_saturated = luma + saturation_c * (x_contrasted - luma)
        
        # Vibrance - selective saturation boost for less saturated areas
        vibrance_c = torch.clamp(self.vibrance, -0.3, 0.4)
        sat_level = torch.mean(torch.abs(x_saturated - luma), dim=1, keepdim=True)
        vibrance_mask = torch.exp(-sat_level * 3)  # Boost less saturated areas more
        x_vibrant = x_saturated + vibrance_c * vibrance_mask * (x_saturated - luma)
        
        # Color temperature adjustment
        warmth_c = torch.clamp(self.warmth, -0.1, 0.15)
        x_final = x_vibrant.clone()
        x_final[:, 0, :, :] = torch.clamp(x_final[:, 0, :, :] * (1 + warmth_c), 0, 1)
        x_final[:, 1, :, :] = torch.clamp(x_final[:, 1, :, :] * (1 + warmth_c * 0.5), 0, 1)
        x_final[:, 2, :, :] = torch.clamp(x_final[:, 2, :, :] * (1 - warmth_c * 0.8), 0, 1)
        
        return torch.clamp(x_final, 0, 1)
    
    def forward(self, x):
        # Classical pipeline with film characteristics
        x_matrix = self.apply_film_color_matrix(x)
        x_toned = self.apply_cinematic_tone_curve(x_matrix)
        x_graded = self.apply_cinematic_color_grading(x_toned)
        
        # ML enhancement
        x_ml = self.ml_enhancement(x_graded)
        
        # Higher ML contribution for learning
        blend_ratio = torch.clamp(self.ml_blend, 0.2, 0.4)
        output = (1 - blend_ratio) * x_graded + blend_ratio * x_ml
        
        return torch.clamp(output, 0, 1)

class CinemaDataset(Dataset):
    """Dataset with augmentation for better learning"""
    
    def __init__(self, training_data_path: Path, size: int = 384, augment: bool = True):
        self.data_path = training_data_path
        self.size = size
        self.augment = augment
        self.pairs = self.load_pairs()
    
    def load_pairs(self):
        metadata_file = self.data_path / "depth_metadata.json"
        with open(metadata_file, 'r') as f:
            pairs = json.load(f)
        return pairs
    
    def augment_image(self, img):
        """Simple augmentations to increase dataset variety"""
        if not self.augment:
            return img
        
        # Random brightness adjustment
        if np.random.rand() > 0.5:
            brightness = np.random.uniform(0.9, 1.1)
            img = np.clip(img * brightness, 0, 1)
        
        # Random contrast adjustment
        if np.random.rand() > 0.5:
            contrast = np.random.uniform(0.9, 1.1)
            mean = np.mean(img)
            img = np.clip((img - mean) * contrast + mean, 0, 1)
        
        return img
    
    def process_image(self, file_path: str):
        """Process with optional augmentation"""
        try:
            with rawpy.imread(file_path) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    output_bps=16,
                    no_auto_bright=True,
                    user_flip=0
                )
            
            rgb_norm = np.clip(rgb.astype(np.float32) / 65535.0, 0, 1)
            
            # Augment before cropping
            if self.augment:
                rgb_norm = self.augment_image(rgb_norm)
            
            # Random crop for variety
            h, w = rgb_norm.shape[:2]
            if min(h, w) > self.size:
                y = np.random.randint(0, h - self.size)
                x = np.random.randint(0, w - self.size)
                rgb_crop = rgb_norm[y:y+self.size, x:x+self.size]
            else:
                rgb_crop = cv2.resize(rgb_norm, (self.size, self.size))
            
            return np.transpose(rgb_crop, (2, 0, 1))
            
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

def saturation_loss(pred, target):
    """Encourage proper saturation levels"""
    # Convert to HSV-like representation
    pred_max = torch.max(pred, dim=1)[0]
    pred_min = torch.min(pred, dim=1)[0]
    pred_sat = (pred_max - pred_min) / (pred_max + 1e-6)
    
    target_max = torch.max(target, dim=1)[0]
    target_min = torch.min(target, dim=1)[0]
    target_sat = (target_max - target_min) / (target_max + 1e-6)
    
    return F.l1_loss(pred_sat, target_sat)

def perceptual_loss(pred, target):
    """Enhanced perceptual loss"""
    # Edge-aware loss
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    
    # Compute gradients for each channel
    loss = 0
    for c in range(3):
        pred_c = pred[:, c:c+1, :, :]
        target_c = target[:, c:c+1, :, :]
        
        pred_gx = F.conv2d(pred_c, kernel_x.to(pred.device), padding=1)
        pred_gy = F.conv2d(pred_c, kernel_y.to(pred.device), padding=1)
        pred_grad = torch.sqrt(pred_gx**2 + pred_gy**2 + 1e-6)
        
        target_gx = F.conv2d(target_c, kernel_x.to(target.device), padding=1)
        target_gy = F.conv2d(target_c, kernel_y.to(target.device), padding=1)
        target_grad = torch.sqrt(target_gx**2 + target_gy**2 + 1e-6)
        
        loss += F.l1_loss(pred_grad, target_grad)
    
    return loss / 3

def train_v2_model():
    """Train enhanced model with better color"""
    print("🎬 TRAINING CINEMA MODEL V2")
    print("=" * 50)
    print("Enhanced for better saturation and vibrancy")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    data_path = Path("data/results/simple_depth_analysis")
    
    # Dataset with augmentation
    dataset = CinemaDataset(data_path, size=384, augment=True)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    model = CinemaTransformV2().to(device)
    
    # Higher learning rate for faster convergence
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=0.001,
        epochs=50,
        steps_per_epoch=len(dataloader)
    )
    
    print(f"📊 Dataset size: {len(dataset)} pairs")
    print(f"🔧 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"🎨 Focus: Enhanced color vibrancy")
    
    num_epochs = 50
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        losses = {'mse': 0, 'sat': 0, 'percep': 0}
        num_batches = 0
        
        for batch in dataloader:
            iphone_imgs = batch['iphone'].to(device)
            sony_imgs = batch['sony'].to(device)
            
            optimizer.zero_grad()
            predicted = model(iphone_imgs)
            
            # Multi-component loss with saturation emphasis
            mse_loss = F.mse_loss(predicted, sony_imgs)
            sat_loss = saturation_loss(predicted, sony_imgs)
            percep_loss = perceptual_loss(predicted, sony_imgs)
            
            # Emphasize saturation matching
            total_loss_batch = (
                0.4 * mse_loss +
                0.3 * sat_loss +    # Higher weight for saturation
                0.3 * percep_loss
            )
            
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += total_loss_batch.item()
            losses['mse'] += mse_loss.item()
            losses['sat'] += sat_loss.item()
            losses['percep'] += percep_loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        if epoch % 5 == 0:
            print(f"\n📊 Epoch {epoch + 1}/{num_epochs}:")
            print(f"     Total Loss: {avg_loss:.6f}")
            print(f"     MSE: {losses['mse']/num_batches:.4f}, Sat: {losses['sat']/num_batches:.4f}, Percep: {losses['percep']/num_batches:.4f}")
            
            with torch.no_grad():
                print(f"     Saturation: {model.saturation.item():.3f}, Vibrance: {model.vibrance.item():.3f}")
                print(f"     Contrast: {model.contrast.item():.3f}, ML Blend: {model.ml_blend.item():.3f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_loss
            }, Path("data/cinema_v2_model.pth"))
    
    print(f"\n✅ Training complete!")
    print(f"🎯 Best loss: {best_loss:.6f}")
    return model

def test_v2_model():
    """Test the enhanced model"""
    print("\n🧪 TESTING CINEMA MODEL V2")
    print("=" * 50)
    
    model_path = Path("data/cinema_v2_model.pth")
    if not model_path.exists():
        print("❌ No v2 model found. Train first!")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CinemaTransformV2().to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded (epoch {checkpoint['epoch']})")
    
    # Print final parameters
    print(f"📊 Learned parameters:")
    print(f"   Saturation: {model.saturation.item():.3f}")
    print(f"   Vibrance: {model.vibrance.item():.3f}")
    print(f"   Contrast: {model.contrast.item():.3f}")
    print(f"   Warmth: {model.warmth.item():.3f}")
    
    # Test on images
    training_pairs_dir = Path("data/training_pairs")
    iphone_files = list(training_pairs_dir.glob("iphone_*.dng"))[:5]
    
    results_dir = Path("data/results/cinema_v2_test")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = CinemaDataset(Path("data/results/simple_depth_analysis"), augment=False)
    
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
            
            rgb_norm = np.clip(rgb.astype(np.float32) / 65535.0, 0, 1)
            h, w = rgb_norm.shape[:2]
            max_dim = 1536
            
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
            
            # Save result
            original_display = (rgb_resized * 255).astype(np.uint8)
            transformed_display = (transformed_np * 255).astype(np.uint8)
            
            # Create side-by-side comparison
            comparison = np.hstack([
                cv2.cvtColor(original_display, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(transformed_display, cv2.COLOR_RGB2BGR)
            ])
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(comparison, "iPhone Original", (10, 30), font, 1, (0, 255, 0), 2)
            cv2.putText(comparison, "Cinema V2 Transform", (rgb_resized.shape[1] + 10, 30), font, 1, (0, 255, 0), 2)
            
            output_path = results_dir / f"v2_{iphone_file.stem}.jpg"
            cv2.imwrite(str(output_path), comparison, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            print(f"  ✅ Saved: {output_path}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n✅ Test complete! Check: {results_dir}")

def main():
    """Main entry point"""
    print("🎬 CINEMA MODEL V2 - ENHANCED COLOR")
    print("Fixing washed-out look with better saturation")
    print("\nChoose option:")
    print("1. Train enhanced model")
    print("2. Test enhanced model")
    print("3. Both")
    print("4. Compare v1 vs v2 results")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice in ["1", "3"]:
        train_v2_model()
    
    if choice in ["2", "3"]:
        test_v2_model()
    
    if choice == "4":
        compare_results()

def compare_results():
    """Visual comparison of v1 vs v2"""
    v1_dir = Path("data/results/hybrid_v1_1_stable")
    v2_dir = Path("data/results/cinema_v2_test")
    
    print("\n📊 Comparing V1 vs V2 results...")
    
    # Find matching files
    v1_files = list(v1_dir.glob("stable_*.jpg"))
    
    comparison_dir = Path("data/results/model_comparison")
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    for v1_file in v1_files[:3]:  # Compare first 3
        stem = v1_file.stem.replace("stable_", "")
        v2_file = v2_dir / f"v2_{stem}.jpg"
        
        if v2_file.exists():
            # Load both
            v1_img = cv2.imread(str(v1_file))
            v2_img = cv2.imread(str(v2_file))
            
            # Extract transformed parts
            h, w = v1_img.shape[:2]
            v1_trans = v1_img[:, w//2:]
            v2_trans = v2_img[:, w//2:]
            orig = v1_img[:, :w//2]
            
            # Create 3-way comparison
            comparison = np.hstack([orig, v1_trans, v2_trans])
            
            # Labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(comparison, "Original", (10, 30), font, 1, (0, 255, 0), 2)
            cv2.putText(comparison, "V1 (Washed)", (w//2 + 10, 30), font, 1, (0, 255, 255), 2)
            cv2.putText(comparison, "V2 (Enhanced)", (w + 10, 30), font, 1, (0, 255, 0), 2)
            
            output = comparison_dir / f"comparison_{stem}.jpg"
            cv2.imwrite(str(output), comparison, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"  ✅ Saved comparison: {output}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Cinema Model v1.2 - LUT-aware transformation with gradient fixes
Incorporates reference LUTs for better color science
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
import base64

def parse_cube_lut(file_path):
    """Parse a .cube LUT file into a 3D numpy array"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Find LUT size
        lut_size = None
        data_start = 0
        for i, line in enumerate(lines):
            if line.startswith('LUT_3D_SIZE'):
                lut_size = int(line.split()[1])
            elif not line.startswith('#') and not line.startswith('TITLE') and not line.startswith('LUT_3D_SIZE'):
                data_start = i
                break
        
        if lut_size is None:
            print(f"Warning: Could not find LUT size in {file_path}")
            return None
        
        # Parse LUT data
        lut_data = []
        for line in lines[data_start:]:
            if line.strip():
                values = [float(x) for x in line.split()]
                if len(values) == 3:
                    lut_data.append(values)
        
        # Reshape to 3D LUT
        lut_3d = np.array(lut_data).reshape((lut_size, lut_size, lut_size, 3))
        return lut_3d.astype(np.float32)
        
    except Exception as e:
        print(f"Error parsing LUT file {file_path}: {e}")
        return None

def decode_base64_lut(b64_file_path):
    """Decode base64 encoded LUT file"""
    try:
        with open(b64_file_path, 'r') as f:
            b64_content = f.read()
        
        # Decode base64
        decoded_bytes = base64.b64decode(b64_content)
        decoded_text = decoded_bytes.decode('utf-8')
        
        # Save to temp file and parse
        temp_path = Path(b64_file_path).with_suffix('.cube')
        with open(temp_path, 'w') as f:
            f.write(decoded_text)
        
        lut = parse_cube_lut(temp_path)
        temp_path.unlink()  # Clean up temp file
        
        return lut
        
    except Exception as e:
        print(f"Error decoding base64 LUT {b64_file_path}: {e}")
        return None

def apply_lut_torch(image, lut_3d):
    """Apply 3D LUT to image tensor"""
    if lut_3d is None:
        return image
    
    b, c, h, w = image.shape
    lut_size = lut_3d.shape[0]
    
    # Flatten spatial dimensions
    image_flat = image.permute(0, 2, 3, 1).reshape(-1, 3)
    
    # Scale to LUT coordinates
    coords = image_flat * (lut_size - 1)
    
    # Get integer and fractional parts
    coords_floor = coords.floor().long()
    coords_frac = coords - coords_floor.float()
    
    # Clamp coordinates
    coords_floor = torch.clamp(coords_floor, 0, lut_size - 2)
    
    # Trilinear interpolation
    lut_tensor = torch.from_numpy(lut_3d).to(image.device)
    
    # Get the 8 surrounding points
    x0, y0, z0 = coords_floor[:, 0], coords_floor[:, 1], coords_floor[:, 2]
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
    
    # Interpolation weights
    fx, fy, fz = coords_frac[:, 0], coords_frac[:, 1], coords_frac[:, 2]
    
    # Trilinear interpolation
    c000 = lut_tensor[x0, y0, z0]
    c001 = lut_tensor[x0, y0, z1]
    c010 = lut_tensor[x0, y1, z0]
    c011 = lut_tensor[x0, y1, z1]
    c100 = lut_tensor[x1, y0, z0]
    c101 = lut_tensor[x1, y0, z1]
    c110 = lut_tensor[x1, y1, z0]
    c111 = lut_tensor[x1, y1, z1]
    
    # Interpolate along x
    fx = fx.unsqueeze(1)
    c00 = c000 * (1 - fx) + c100 * fx
    c01 = c001 * (1 - fx) + c101 * fx
    c10 = c010 * (1 - fx) + c110 * fx
    c11 = c011 * (1 - fx) + c111 * fx
    
    # Interpolate along y
    fy = fy.unsqueeze(1)
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    
    # Interpolate along z
    fz = fz.unsqueeze(1)
    output = c0 * (1 - fz) + c1 * fz
    
    # Reshape back
    output = output.reshape(b, h, w, c).permute(0, 3, 1, 2)
    
    return output

class LUTAwareColorNet(nn.Module):
    """ML component that learns residuals on top of LUT"""
    
    def __init__(self):
        super(LUTAwareColorNet, self).__init__()
        
        # Feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),  # 6 channels: original + LUT processed
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),  # Avoid in-place operations
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False)
        )
        
        # Residual prediction
        self.residual = nn.Conv2d(32, 3, 3, padding=1)
        
        # Learnable residual strength
        self.residual_strength = nn.Parameter(torch.tensor(0.2))
    
    def forward(self, x_original, x_lut):
        # Concatenate original and LUT-processed
        x_combined = torch.cat([x_original, x_lut], dim=1)
        
        # Extract features
        features = self.features(x_combined)
        
        # Predict residual
        residual = torch.tanh(self.residual(features))
        
        # Apply residual with learnable strength
        strength = torch.clamp(self.residual_strength, 0.05, 0.4)
        output = x_lut + strength * residual
        
        return torch.clamp(output, 0, 1)

class CinemaTransformV1_2(nn.Module):
    """Cinema transformation v1.2 with LUT integration"""
    
    def __init__(self, reference_lut=None):
        super(CinemaTransformV1_2, self).__init__()
        
        self.reference_lut = reference_lut
        
        # Classical color adjustments (more conservative)
        self.color_matrix = nn.Parameter(torch.eye(3))
        self.color_bias = nn.Parameter(torch.zeros(3))
        
        # Tone curve parameters
        self.shadows = nn.Parameter(torch.tensor(0.05))
        self.mids = nn.Parameter(torch.tensor(1.0))
        self.highlights = nn.Parameter(torch.tensor(0.95))
        
        # Color grading
        self.contrast = nn.Parameter(torch.tensor(1.05))
        self.saturation = nn.Parameter(torch.tensor(1.2))
        self.warmth = nn.Parameter(torch.tensor(0.02))
        
        # LUT-aware ML component
        self.ml_enhancement = LUTAwareColorNet()
        
        # LUT blend ratio
        self.lut_blend = nn.Parameter(torch.tensor(0.7))
    
    def apply_color_matrix(self, x):
        """Apply learned color matrix"""
        matrix = torch.clamp(self.color_matrix, 0.8, 1.2)
        bias = torch.clamp(self.color_bias, -0.05, 0.05)
        
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1)
        
        transformed = torch.bmm(matrix.unsqueeze(0).expand(b, -1, -1), x_flat)
        transformed = transformed + bias.view(1, 3, 1)
        
        return transformed.view(b, c, h, w)
    
    def apply_tone_curve(self, x):
        """Apply tone curve"""
        shadows_c = torch.clamp(self.shadows, -0.1, 0.2)
        mids_c = torch.clamp(self.mids, 0.85, 1.15)
        highlights_c = torch.clamp(self.highlights, 0.85, 1.15)
        
        # Create new tensor to avoid in-place operations
        x_toned = torch.zeros_like(x)
        
        # Shadow lift
        shadow_mask = torch.clamp(1 - x, 0, 1) ** 2
        x_toned = x + shadows_c * shadow_mask * (1 - x)
        
        # Midtone adjustment
        x_toned = torch.pow(torch.clamp(x_toned, 1e-7, 1), 1.0 / mids_c)
        
        # Highlight compression
        highlight_mask = torch.clamp(x_toned, 0, 1) ** 2
        x_toned = x_toned * (1 - highlight_mask * (1 - highlights_c))
        
        return torch.clamp(x_toned, 0, 1)
    
    def apply_color_grading(self, x):
        """Apply color grading"""
        # Contrast
        contrast_c = torch.clamp(self.contrast, 0.9, 1.2)
        x_graded = torch.pow(torch.clamp(x, 1e-7, 1), 1.0 / contrast_c)
        
        # Calculate luminance
        luma = 0.299 * x_graded[:, 0:1, :, :] + \
               0.587 * x_graded[:, 1:2, :, :] + \
               0.114 * x_graded[:, 2:3, :, :]
        
        # Saturation
        saturation_c = torch.clamp(self.saturation, 0.8, 1.5)
        x_graded = luma + saturation_c * (x_graded - luma)
        
        # Warmth (create new tensor)
        warmth_c = torch.clamp(self.warmth, -0.05, 0.1)
        x_final = x_graded.clone()
        x_final[:, 0, :, :] = torch.clamp(x_final[:, 0, :, :] * (1 + warmth_c), 0, 1)
        x_final[:, 1, :, :] = torch.clamp(x_final[:, 1, :, :] * (1 + warmth_c * 0.5), 0, 1)
        x_final[:, 2, :, :] = torch.clamp(x_final[:, 2, :, :] * (1 - warmth_c * 0.7), 0, 1)
        
        return torch.clamp(x_final, 0, 1)
    
    def forward(self, x):
        # Apply classical adjustments first
        x_classical = self.apply_color_matrix(x)
        x_classical = self.apply_tone_curve(x_classical)
        x_classical = self.apply_color_grading(x_classical)
        
        # Apply reference LUT if available
        if self.reference_lut is not None:
            x_lut = apply_lut_torch(x_classical, self.reference_lut)
            lut_blend = torch.clamp(self.lut_blend, 0.5, 0.9)
            x_blended = lut_blend * x_lut + (1 - lut_blend) * x_classical
        else:
            x_blended = x_classical
            x_lut = x_classical
        
        # ML enhancement on top
        x_enhanced = self.ml_enhancement(x, x_blended)
        
        return x_enhanced

class CinemaDatasetV1_2(Dataset):
    """Dataset for v1.2 training"""
    
    def __init__(self, training_data_path: Path, size: int = 384):
        self.data_path = training_data_path
        self.size = size
        self.pairs = self.load_pairs()
    
    def load_pairs(self):
        metadata_file = self.data_path / "depth_metadata.json"
        with open(metadata_file, 'r') as f:
            pairs = json.load(f)
        return pairs
    
    def process_image(self, file_path: str):
        """Process image"""
        try:
            with rawpy.imread(file_path) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    output_bps=16,
                    no_auto_bright=True,
                    user_flip=0
                )
            
            rgb_norm = np.clip(rgb.astype(np.float32) / 65535.0, 0, 1)
            
            # Center crop
            h, w = rgb_norm.shape[:2]
            if min(h, w) > self.size:
                y = (h - self.size) // 2
                x = (w - self.size) // 2
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

def load_reference_lut():
    """Load reference LUT (Neutral A7s3)"""
    # Try different LUT sources
    lut_paths = [
        Path("data/Neutral A7s3_65x.cube"),
        Path("data/Neutral A7s3_65x.b64.txt"),
        Path("Neutral A7s3_65x.b64.txt")
    ]
    
    for lut_path in lut_paths:
        if lut_path.exists():
            print(f"Found LUT file: {lut_path}")
            if lut_path.suffix == '.cube':
                return parse_cube_lut(lut_path)
            elif lut_path.suffix == '.txt' and '.b64' in lut_path.name:
                return decode_base64_lut(lut_path)
    
    print("Warning: No reference LUT found, training without LUT")
    return None

def train_v1_2():
    """Train v1.2 model with LUT integration"""
    print("🎬 TRAINING CINEMA MODEL V1.2")
    print("=" * 50)
    print("LUT-aware transformation with gradient fixes")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Load reference LUT
    reference_lut = load_reference_lut()
    if reference_lut is not None:
        print(f"✅ Loaded reference LUT: {reference_lut.shape}")
    else:
        print("⚠️ Training without reference LUT")
    
    data_path = Path("data/results/simple_depth_analysis")
    
    # Dataset
    dataset = CinemaDatasetV1_2(data_path, size=384)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    model = CinemaTransformV1_2(reference_lut=reference_lut).to(device)
    
    # Optimizer with gradient clipping
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=0.00001)
    
    print(f"📊 Dataset size: {len(dataset)} pairs")
    print(f"🔧 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    num_epochs = 40
    best_loss = float('inf')
    
    # Enable anomaly detection for debugging
    torch.autograd.set_detect_anomaly(True)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            iphone_imgs = batch['iphone'].to(device)
            sony_imgs = batch['sony'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predicted = model(iphone_imgs)
            
            # Simple MSE loss for now
            loss = F.mse_loss(predicted, sony_imgs)
            
            # Add histogram matching loss
            hist_loss = 0
            for c in range(3):
                pred_hist = torch.histc(predicted[:, c, :, :], bins=64, min=0, max=1)
                target_hist = torch.histc(sony_imgs[:, c, :, :], bins=64, min=0, max=1)
                pred_hist = pred_hist / (pred_hist.sum() + 1e-8)
                target_hist = target_hist / (target_hist.sum() + 1e-8)
                hist_loss = hist_loss + F.mse_loss(pred_hist, target_hist)
            
            total_loss_batch = loss + 0.1 * hist_loss
            
            # Backward pass
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += total_loss_batch.item()
            num_batches += 1
        
        scheduler.step()
        avg_loss = total_loss / num_batches
        
        if epoch % 5 == 0:
            print(f"\n📊 Epoch {epoch + 1}/{num_epochs}:")
            print(f"     Loss: {avg_loss:.6f}")
            
            with torch.no_grad():
                print(f"     Saturation: {model.saturation.item():.3f}")
                print(f"     Contrast: {model.contrast.item():.3f}")
                if reference_lut is not None:
                    print(f"     LUT blend: {model.lut_blend.item():.3f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_loss,
                'has_lut': reference_lut is not None
            }, Path("data/cinema_v1_2_model.pth"))
    
    print(f"\n✅ Training complete!")
    print(f"🎯 Best loss: {best_loss:.6f}")
    
    # Disable anomaly detection
    torch.autograd.set_detect_anomaly(False)
    
    return model

def test_v1_2():
    """Test v1.2 model"""
    print("\n🧪 TESTING CINEMA MODEL V1.2")
    print("=" * 50)
    
    model_path = Path("data/cinema_v1_2_model.pth")
    if not model_path.exists():
        print("❌ No v1.2 model found. Train first!")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load reference LUT if model was trained with it
    reference_lut = None
    if checkpoint.get('has_lut', False):
        reference_lut = load_reference_lut()
    
    model = CinemaTransformV1_2(reference_lut=reference_lut).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded (epoch {checkpoint['epoch']})")
    
    # Test on images
    training_pairs_dir = Path("data/training_pairs")
    iphone_files = list(training_pairs_dir.glob("iphone_*.dng"))[:5]
    
    results_dir = Path("data/results/cinema_v1_2_test")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = CinemaDatasetV1_2(Path("data/results/simple_depth_analysis"))
    
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
            
            comparison = np.hstack([
                cv2.cvtColor(original_display, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(transformed_display, cv2.COLOR_RGB2BGR)
            ])
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(comparison, "iPhone Original", (10, 30), font, 1, (0, 255, 0), 2)
            cv2.putText(comparison, "Cinema v1.2 (LUT-aware)", (rgb_resized.shape[1] + 10, 30), font, 1, (0, 255, 0), 2)
            
            output_path = results_dir / f"v1_2_{iphone_file.stem}.jpg"
            cv2.imwrite(str(output_path), comparison, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            print(f"  ✅ Saved: {output_path}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n✅ Test complete! Check: {results_dir}")

def main():
    """Main entry point"""
    print("🎬 CINEMA MODEL V1.2 - LUT INTEGRATION")
    print("Reference LUT support + gradient fixes")
    print("\nChoose option:")
    print("1. Train v1.2 model")
    print("2. Test v1.2 model")
    print("3. Both")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice in ["1", "3"]:
        train_v1_2()
    
    if choice in ["2", "3"]:
        test_v1_2()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Cinema Model v1.3 - High Resolution Training & Enhanced LUT Support
- Train on full 4K resolution for better color depth preservation
- Support for expanded LUT library
- Improved model architecture for high-res processing
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
    """Apply 3D LUT to image tensor with trilinear interpolation"""
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

class HighResColorNet(nn.Module):
    """Enhanced ML component for high-resolution processing"""
    
    def __init__(self):
        super(HighResColorNet, self).__init__()
        
        # Multi-scale feature extraction for high-res images
        self.features = nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False)
        )
        
        # Residual prediction with attention
        self.residual = nn.Conv2d(32, 3, 3, padding=1)
        self.attention = nn.Conv2d(32, 1, 3, padding=1)
        
        # Learnable residual strength and spatial weighting
        self.residual_strength = nn.Parameter(torch.tensor(0.15))
        self.spatial_weight = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x_original, x_lut):
        # Concatenate original and LUT-processed
        x_combined = torch.cat([x_original, x_lut], dim=1)
        
        # Extract features
        features = self.features(x_combined)
        
        # Predict residual with spatial attention
        residual = torch.tanh(self.residual(features))
        attention_map = torch.sigmoid(self.attention(features))
        
        # Apply residual with learnable strength and attention
        strength = torch.clamp(self.residual_strength, 0.05, 0.3)
        weighted_residual = strength * residual * attention_map * self.spatial_weight
        
        output = x_lut + weighted_residual
        
        return torch.clamp(output, 0, 1)

class CinemaTransformV1_3(nn.Module):
    """Cinema transformation v1.3 with high-resolution support"""
    
    def __init__(self, reference_lut=None):
        super(CinemaTransformV1_3, self).__init__()
        
        self.reference_lut = reference_lut
        
        # Enhanced color adjustments
        self.color_matrix = nn.Parameter(torch.eye(3) * 0.98 + 0.02)  # Start closer to identity
        self.color_bias = nn.Parameter(torch.zeros(3))
        
        # Improved tone curve parameters
        self.shadows = nn.Parameter(torch.tensor(0.02))
        self.mids = nn.Parameter(torch.tensor(1.0))
        self.highlights = nn.Parameter(torch.tensor(0.98))
        
        # Color grading with better initialization
        self.contrast = nn.Parameter(torch.tensor(1.02))
        self.saturation = nn.Parameter(torch.tensor(1.15))
        self.warmth = nn.Parameter(torch.tensor(0.01))
        
        # High-res ML component
        self.ml_enhancement = HighResColorNet()
        
        # LUT blend ratio
        self.lut_blend = nn.Parameter(torch.tensor(0.75))
    
    def apply_color_matrix(self, x):
        """Apply learned color matrix"""
        matrix = torch.clamp(self.color_matrix, 0.85, 1.15)
        bias = torch.clamp(self.color_bias, -0.03, 0.03)
        
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1)
        
        # Matrix multiplication
        matrix_expanded = matrix.unsqueeze(0).expand(b, -1, -1)
        transformed = torch.bmm(matrix_expanded, x_flat)
        
        # Add bias
        bias_expanded = bias.view(1, 3, 1).expand(b, -1, x_flat.size(2))
        transformed = transformed + bias_expanded
        
        return transformed.view(b, c, h, w)
    
    def apply_tone_curve(self, x):
        """Apply tone curve without in-place operations"""
        shadows_c = torch.clamp(self.shadows, -0.05, 0.1)
        mids_c = torch.clamp(self.mids, 0.9, 1.1)
        highlights_c = torch.clamp(self.highlights, 0.9, 1.1)
        
        # Shadow lift (no in-place)
        shadow_mask = (1 - x).clamp(0, 1) ** 2
        shadow_lift = shadows_c * shadow_mask * (1 - x)
        x_with_shadows = x + shadow_lift
        
        # Midtone adjustment
        x_mids = torch.pow(x_with_shadows.clamp(1e-7, 1), 1.0 / mids_c)
        
        # Highlight compression
        highlight_mask = x_mids.clamp(0, 1) ** 2
        highlight_factor = 1 - highlight_mask * (1 - highlights_c)
        x_final = x_mids * highlight_factor
        
        return torch.clamp(x_final, 0, 1)
    
    def apply_color_grading(self, x):
        """Apply color grading without in-place operations"""
        # Contrast
        contrast_c = torch.clamp(self.contrast, 0.95, 1.15)
        x_contrasted = torch.pow(x.clamp(1e-7, 1), 1.0 / contrast_c)
        
        # Calculate luminance
        luma_r = x_contrasted[:, 0:1, :, :] * 0.299
        luma_g = x_contrasted[:, 1:2, :, :] * 0.587
        luma_b = x_contrasted[:, 2:3, :, :] * 0.114
        luma = luma_r + luma_g + luma_b
        
        # Saturation
        saturation_c = torch.clamp(self.saturation, 0.9, 1.4)
        color_diff = x_contrasted - luma
        x_saturated = luma + saturation_c * color_diff
        
        # Warmth - create new tensors for each channel
        warmth_c = torch.clamp(self.warmth, -0.03, 0.08)
        
        # Apply warmth to each channel separately
        r_channel = (x_saturated[:, 0:1, :, :] * (1 + warmth_c)).clamp(0, 1)
        g_channel = (x_saturated[:, 1:2, :, :] * (1 + warmth_c * 0.4)).clamp(0, 1)
        b_channel = (x_saturated[:, 2:3, :, :] * (1 - warmth_c * 0.6)).clamp(0, 1)
        
        # Concatenate channels
        x_final = torch.cat([r_channel, g_channel, b_channel], dim=1)
        
        return x_final
    
    def forward(self, x):
        # Apply classical adjustments
        x_classical = self.apply_color_matrix(x)
        x_classical = self.apply_tone_curve(x_classical)
        x_classical = self.apply_color_grading(x_classical)
        
        # Apply reference LUT if available
        if self.reference_lut is not None:
            x_lut = apply_lut_torch(x_classical, self.reference_lut)
            lut_blend = torch.clamp(self.lut_blend, 0.6, 0.85)
            x_blended = lut_blend * x_lut + (1 - lut_blend) * x_classical
        else:
            x_blended = x_classical
            x_lut = x_classical
        
        # ML enhancement on top
        x_enhanced = self.ml_enhancement(x, x_blended)
        
        return x_enhanced

class HighResDatasetV1_3(Dataset):
    """High-resolution dataset for v1.3 training"""
    
    def __init__(self, training_data_path: Path, target_size: int = 1024):
        self.data_path = training_data_path
        self.target_size = target_size  # Much higher resolution
        self.pairs = self.load_pairs()
        
    def load_pairs(self):
        metadata_file = self.data_path / "depth_metadata.json"
        with open(metadata_file, 'r') as f:
            pairs = json.load(f)
        return pairs
    
    def process_image_highres(self, file_path: str):
        """Process image at high resolution preserving color depth"""
        try:
            with rawpy.imread(file_path) as raw:
                # Use higher bit depth and preserve more color information
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    output_bps=16,
                    no_auto_bright=True,
                    user_flip=0,
                    gamma=(1, 1),  # Linear gamma for better color preservation
                    output_color=rawpy.ColorSpace.sRGB
                )
            
            # Preserve full dynamic range
            rgb_norm = rgb.astype(np.float32) / 65535.0
            
            # Smart resize to preserve detail
            h, w = rgb_norm.shape[:2]
            if min(h, w) > self.target_size:
                # Center crop to preserve aspect ratio and detail
                if h > w:
                    y_start = (h - self.target_size) // 2
                    rgb_crop = rgb_norm[y_start:y_start+self.target_size, :w]
                    rgb_resized = cv2.resize(rgb_crop, (self.target_size, self.target_size))
                else:
                    x_start = (w - self.target_size) // 2
                    rgb_crop = rgb_norm[:h, x_start:x_start+self.target_size]
                    rgb_resized = cv2.resize(rgb_crop, (self.target_size, self.target_size))
            else:
                rgb_resized = cv2.resize(rgb_norm, (self.target_size, self.target_size))
            
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

def load_all_luts():
    """Load all available LUTs from the luts folder"""
    lut_paths = [
        # New LUTs from your expanded collection
        Path("luts/Neutral A7s3_65x.cube"),
        Path("luts/Eterna A7s3_65x_Legacy.cube"),
        Path("luts/P6k Neutral Gen5.cube"),
        Path("luts/P6k Utopia Gen5-Legacy.cube"),
        Path("luts/VisionTeal A7s3_65x.cube"),
        Path("luts/Vision A7s3_65x.cube"),
        Path("luts/Eastman A7s3_65x.cube"),
        Path("luts/Bleach A7s3_65x.cube"),
        Path("luts/Utopia A7s3_65x.cube"),
        Path("luts/Tungsten A7s3_65x.cube"),
        Path("luts/Jamaica A7s3_65x.cube"),
        Path("luts/IceBlue A7s3_65x.cube"),
        
        # Base64 encoded versions as fallback
        Path("luts/Neutral A7s3_65x.b64.txt"),
        Path("luts/Eterna A7s3_65x_Legacy.b64.txt"),
        Path("luts/Neutral A7s3_Legacy_65x.b64.txt"),
    ]
    
    print("🎨 Searching for LUT files...")
    available_luts = {}
    
    for lut_path in lut_paths:
        if lut_path.exists():
            print(f"✅ Found LUT: {lut_path.name}")
            try:
                if lut_path.suffix == '.cube':
                    lut = parse_cube_lut(lut_path)
                elif lut_path.suffix == '.txt' and '.b64' in lut_path.name:
                    lut = decode_base64_lut(lut_path)
                else:
                    continue
                    
                if lut is not None:
                    # Use a clean name as key
                    clean_name = lut_path.stem.replace('.b64', '')
                    available_luts[clean_name] = lut
                    print(f"   Loaded: {lut.shape}")
            except Exception as e:
                print(f"   Error loading {lut_path}: {e}")
    
    if not available_luts:
        print("⚠️ No LUTs loaded successfully")
        return None
    
    # Return the best neutral LUT as primary
    if 'Neutral A7s3_65x' in available_luts:
        primary_lut = available_luts['Neutral A7s3_65x']
        print(f"🎯 Using primary LUT: Neutral A7s3_65x {primary_lut.shape}")
        return primary_lut, available_luts
    else:
        # Use first available LUT
        first_key = list(available_luts.keys())[0]
        primary_lut = available_luts[first_key]
        print(f"🎯 Using primary LUT: {first_key} {primary_lut.shape}")
        return primary_lut, available_luts

def train_v1_3():
    """Train v1.3 model with high resolution and enhanced LUTs"""
    print("🎬 TRAINING CINEMA MODEL V1.3")
    print("=" * 50)
    print("High-resolution training + expanded LUT support")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Load all available LUTs
    lut_result = load_all_luts()
    if lut_result is None:
        print("⚠️ Training without reference LUT")
        reference_lut = None
    else:
        reference_lut, all_luts = lut_result
        print(f"✅ Loaded {len(all_luts)} LUTs, using primary for training")
    
    data_path = Path("data/results/simple_depth_analysis")
    
    # High-resolution dataset
    dataset = HighResDatasetV1_3(data_path, target_size=1024)  # Much higher resolution
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)  # Smaller batch for high-res
    
    model = CinemaTransformV1_3(reference_lut=reference_lut).to(device)
    
    # Optimizer with learning rate scheduling
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower LR for high-res
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.00001)
    
    print(f"📊 Dataset size: {len(dataset)} pairs")
    print(f"🔧 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"📐 Training resolution: 1024x1024 (vs 384x384 in v1.2)")
    
    num_epochs = 50
    best_loss = float('inf')
    
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
            
            # Enhanced loss function
            mse_loss = F.mse_loss(predicted, sony_imgs)
            
            # Perceptual loss (simple version)
            perceptual_loss = F.l1_loss(predicted, sony_imgs)
            
            # Histogram matching loss
            hist_loss = 0
            for c in range(3):
                pred_hist = torch.histc(predicted[:, c, :, :], bins=64, min=0, max=1)
                target_hist = torch.histc(sony_imgs[:, c, :, :], bins=64, min=0, max=1)
                pred_hist = pred_hist / (pred_hist.sum() + 1e-8)
                target_hist = target_hist / (target_hist.sum() + 1e-8)
                hist_loss = hist_loss + F.mse_loss(pred_hist, target_hist)
            
            hist_loss = hist_loss / 3
            
            # Combined loss
            total_loss_batch = mse_loss + 0.1 * perceptual_loss + 0.05 * hist_loss
            
            # Backward pass
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += total_loss_batch.item()
            num_batches += 1
        
        scheduler.step()
        avg_loss = total_loss / num_batches
        
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f"\n📊 Epoch {epoch + 1}/{num_epochs}:")
            print(f"     Loss: {avg_loss:.6f}")
            print(f"     Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
            
            with torch.no_grad():
                print(f"     Saturation: {model.saturation.item():.3f}")
                print(f"     Contrast: {model.contrast.item():.3f}")
                print(f"     Warmth: {model.warmth.item():.3f}")
                if reference_lut is not None:
                    print(f"     LUT blend: {model.lut_blend.item():.3f}")
                print(f"     ML residual: {model.ml_enhancement.residual_strength.item():.3f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_loss,
                'has_lut': reference_lut is not None,
                'training_resolution': 1024
            }, Path("data/cinema_v1_3_model.pth"))
    
    print(f"\n✅ Training complete!")
    print(f"🎯 Best loss: {best_loss:.6f}")
    
    return model

def test_v1_3():
    """Test v1.3 model with high resolution output"""
    print("\n🧪 TESTING CINEMA MODEL V1.3")
    print("=" * 50)
    
    model_path = Path("data/cinema_v1_3_model.pth")
    if not model_path.exists():
        print("❌ No v1.3 model found. Train first!")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load reference LUT if model was trained with it
    lut_result = load_all_luts()
    reference_lut = None
    if checkpoint.get('has_lut', False) and lut_result is not None:
        reference_lut, all_luts = lut_result
    
    model = CinemaTransformV1_3(reference_lut=reference_lut).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    training_res = checkpoint.get('training_resolution', 1024)
    print(f"✅ Model loaded (epoch {checkpoint['epoch']}, trained at {training_res}x{training_res})")
    
    # Print learned parameters
    print(f"\n📊 Learned parameters:")
    print(f"   Saturation: {model.saturation.item():.3f}")
    print(f"   Contrast: {model.contrast.item():.3f}")
    print(f"   Warmth: {model.warmth.item():.3f}")
    if reference_lut is not None:
        print(f"   LUT blend: {model.lut_blend.item():.3f}")
    print(f"   ML residual strength: {model.ml_enhancement.residual_strength.item():.3f}")
    
    # Test on 5 images as requested
    training_pairs_dir = Path("data/training_pairs")
    iphone_files = list(training_pairs_dir.glob("iphone_*.dng"))[:5]
    
    results_dir = Path("data/results/cinema_v1_3_test")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for iphone_file in iphone_files:
        print(f"\n🎯 Processing: {iphone_file.name}")
        
        try:
            # Process at very high resolution
            with rawpy.imread(str(iphone_file)) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    output_bps=16,
                    no_auto_bright=True,
                    user_flip=0,
                    gamma=(1, 1),  # Linear gamma
                    output_color=rawpy.ColorSpace.sRGB
                )
            
            rgb_norm = np.clip(rgb.astype(np.float32) / 65535.0, 0, 1)
            h, w = rgb_norm.shape[:2]
            
            # Process at higher resolution (2K instead of previous lower res)
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
            
            # Save high-quality result
            original_display = (rgb_resized * 255).astype(np.uint8)
            transformed_display = (transformed_np * 255).astype(np.uint8)
            
            comparison = np.hstack([
                cv2.cvtColor(original_display, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(transformed_display, cv2.COLOR_RGB2BGR)
            ])
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = min(1.2, h / 800)
            thickness = max(2, int(h / 400))
            
            cv2.putText(comparison, "iPhone Original", (20, 50), font, font_scale, (0, 255, 0), thickness)
            label = f"Cinema v1.
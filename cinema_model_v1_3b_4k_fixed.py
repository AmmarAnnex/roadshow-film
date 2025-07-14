#!/usr/bin/env python3
"""
Cinema Model v1.3b Fixed - Compatible RAW Processing
Fixed rawpy parameter compatibility issues
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
import random
import gc

def parse_cube_lut(file_path):
    """Parse a .cube LUT file into a 3D numpy array"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
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
        
        lut_data = []
        for line in lines[data_start:]:
            if line.strip():
                values = [float(x) for x in line.split()]
                if len(values) == 3:
                    lut_data.append(values)
        
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
        
        decoded_bytes = base64.b64decode(b64_content)
        decoded_text = decoded_bytes.decode('utf-8')
        
        temp_path = Path(b64_file_path).with_suffix('.cube')
        with open(temp_path, 'w') as f:
            f.write(decoded_text)
        
        lut = parse_cube_lut(temp_path)
        temp_path.unlink()
        
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
    coords_floor = coords.floor().long()
    coords_frac = coords - coords_floor.float()
    coords_floor = torch.clamp(coords_floor, 0, lut_size - 2)
    
    # Trilinear interpolation
    lut_tensor = torch.from_numpy(lut_3d).to(image.device)
    
    x0, y0, z0 = coords_floor[:, 0], coords_floor[:, 1], coords_floor[:, 2]
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
    fx, fy, fz = coords_frac[:, 0], coords_frac[:, 1], coords_frac[:, 2]
    
    # 8-point interpolation
    c000 = lut_tensor[x0, y0, z0]
    c001 = lut_tensor[x0, y0, z1]
    c010 = lut_tensor[x0, y1, z0]
    c011 = lut_tensor[x0, y1, z1]
    c100 = lut_tensor[x1, y0, z0]
    c101 = lut_tensor[x1, y0, z1]
    c110 = lut_tensor[x1, y1, z0]
    c111 = lut_tensor[x1, y1, z1]
    
    fx = fx.unsqueeze(1)
    c00 = c000 * (1 - fx) + c100 * fx
    c01 = c001 * (1 - fx) + c101 * fx
    c10 = c010 * (1 - fx) + c110 * fx
    c11 = c011 * (1 - fx) + c111 * fx
    
    fy = fy.unsqueeze(1)
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    
    fz = fz.unsqueeze(1)
    output = c0 * (1 - fz) + c1 * fz
    
    output = output.reshape(b, h, w, c).permute(0, 3, 1, 2)
    return output

class UltraHighResColorNet(nn.Module):
    """Ultra high-resolution color network optimized for 4K"""
    
    def __init__(self):
        super(UltraHighResColorNet, self).__init__()
        
        # Efficient feature extraction for large images
        self.features = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=False)
        )
        
        # Residual prediction with spatial attention
        self.residual = nn.Conv2d(16, 3, 3, padding=1)
        self.attention = nn.Conv2d(16, 1, 3, padding=1)
        
        # Conservative residual strength for 4K
        self.residual_strength = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x_original, x_lut):
        # Concatenate original and LUT-processed
        x_combined = torch.cat([x_original, x_lut], dim=1)
        
        # Extract features efficiently
        features = self.features(x_combined)
        
        # Predict residual with spatial attention
        residual = torch.tanh(self.residual(features))
        attention_map = torch.sigmoid(self.attention(features))
        
        # Apply conservative residual
        strength = torch.clamp(self.residual_strength, 0.02, 0.2)
        weighted_residual = strength * residual * attention_map
        
        output = x_lut + weighted_residual
        return torch.clamp(output, 0, 1)

class CinemaTransformV1_3b(nn.Module):
    """4K-optimized cinema transformation"""
    
    def __init__(self, reference_lut=None):
        super(CinemaTransformV1_3b, self).__init__()
        
        self.reference_lut = reference_lut
        
        # Conservative color adjustments for 4K
        self.color_matrix = nn.Parameter(torch.eye(3) * 0.99 + 0.01)
        self.color_bias = nn.Parameter(torch.zeros(3) * 0.01)
        
        # Gentle tone curve for 4K detail preservation
        self.shadows = nn.Parameter(torch.tensor(0.01))
        self.mids = nn.Parameter(torch.tensor(1.0))
        self.highlights = nn.Parameter(torch.tensor(0.99))
        
        # Subtle color grading
        self.contrast = nn.Parameter(torch.tensor(1.01))
        self.saturation = nn.Parameter(torch.tensor(1.1))
        self.warmth = nn.Parameter(torch.tensor(0.005))
        
        # 4K-optimized ML component
        self.ml_enhancement = UltraHighResColorNet()
        
        # LUT blend ratio
        self.lut_blend = nn.Parameter(torch.tensor(0.8))
    
    def apply_color_matrix(self, x):
        """Apply learned color matrix"""
        matrix = torch.clamp(self.color_matrix, 0.9, 1.1)
        bias = torch.clamp(self.color_bias, -0.02, 0.02)
        
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1)
        
        matrix_expanded = matrix.unsqueeze(0).expand(b, -1, -1)
        transformed = torch.bmm(matrix_expanded, x_flat)
        
        bias_expanded = bias.view(1, 3, 1).expand(b, -1, x_flat.size(2))
        transformed = transformed + bias_expanded
        
        return transformed.view(b, c, h, w)
    
    def apply_tone_curve(self, x):
        """Apply gentle tone curve for 4K"""
        shadows_c = torch.clamp(self.shadows, -0.02, 0.05)
        mids_c = torch.clamp(self.mids, 0.95, 1.05)
        highlights_c = torch.clamp(self.highlights, 0.95, 1.05)
        
        # Very gentle shadow lift
        shadow_mask = (1 - x).clamp(0, 1) ** 3
        shadow_lift = shadows_c * shadow_mask * (1 - x) * 0.5
        x_with_shadows = x + shadow_lift
        
        # Gentle midtone adjustment
        x_mids = torch.pow(x_with_shadows.clamp(1e-7, 1), 1.0 / mids_c)
        
        # Gentle highlight compression
        highlight_mask = x_mids.clamp(0, 1) ** 3
        highlight_factor = 1 - highlight_mask * (1 - highlights_c) * 0.5
        x_final = x_mids * highlight_factor
        
        return torch.clamp(x_final, 0, 1)
    
    def apply_color_grading(self, x):
        """Apply subtle color grading for 4K"""
        contrast_c = torch.clamp(self.contrast, 0.98, 1.05)
        x_contrasted = torch.pow(x.clamp(1e-7, 1), 1.0 / contrast_c)
        
        # Calculate luminance
        luma_r = x_contrasted[:, 0:1, :, :] * 0.299
        luma_g = x_contrasted[:, 1:2, :, :] * 0.587
        luma_b = x_contrasted[:, 2:3, :, :] * 0.114
        luma = luma_r + luma_g + luma_b
        
        # Gentle saturation enhancement
        saturation_c = torch.clamp(self.saturation, 0.95, 1.3)
        color_diff = x_contrasted - luma
        x_saturated = luma + saturation_c * color_diff
        
        # Subtle warmth adjustment
        warmth_c = torch.clamp(self.warmth, -0.02, 0.05)
        
        r_channel = (x_saturated[:, 0:1, :, :] * (1 + warmth_c)).clamp(0, 1)
        g_channel = (x_saturated[:, 1:2, :, :] * (1 + warmth_c * 0.3)).clamp(0, 1)
        b_channel = (x_saturated[:, 2:3, :, :] * (1 - warmth_c * 0.5)).clamp(0, 1)
        
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
            lut_blend = torch.clamp(self.lut_blend, 0.7, 0.9)
            x_blended = lut_blend * x_lut + (1 - lut_blend) * x_classical
        else:
            x_blended = x_classical
            x_lut = x_classical
        
        # ML enhancement
        x_enhanced = self.ml_enhancement(x, x_blended)
        return x_enhanced

class Full4KDataset(Dataset):
    """Dataset that processes images at full 4K resolution"""
    
    def __init__(self, training_data_path: Path, patch_training=True, patch_size=512):
        self.data_path = training_data_path
        self.patch_training = patch_training
        self.patch_size = patch_size
        self.pairs = self.load_pairs()
        
    def load_pairs(self):
        metadata_file = self.data_path / "depth_metadata.json"
        with open(metadata_file, 'r') as f:
            pairs = json.load(f)
        return pairs
    
    def process_image_4k(self, file_path: str):
        """Process image at full 4K resolution with compatible rawpy settings"""
        try:
            with rawpy.imread(file_path) as raw:
                # Use only compatible rawpy parameters
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    output_bps=16,
                    no_auto_bright=True,
                    user_flip=0
                )
            
            # Convert to float with full precision
            rgb_norm = rgb.astype(np.float32) / 65535.0
            
            print(f"  📐 Loaded at {rgb_norm.shape[1]}x{rgb_norm.shape[0]} (full resolution)")
            
            return rgb_norm
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def extract_patch(self, img, patch_size):
        """Extract a random patch from the image"""
        h, w = img.shape[:2]
        
        if h <= patch_size and w <= patch_size:
            # Image is smaller than patch, pad or return as-is
            return cv2.resize(img, (patch_size, patch_size))
        
        # Random crop
        max_y = h - patch_size
        max_x = w - patch_size
        
        y = random.randint(0, max_y)
        x = random.randint(0, max_x)
        
        patch = img[y:y+patch_size, x:x+patch_size]
        return patch
    
    def __len__(self):
        # For patch training, we can have multiple patches per image pair
        if self.patch_training:
            return len(self.pairs) * 4  # 4 patches per pair per epoch
        else:
            return len(self.pairs)
    
    def __getitem__(self, idx):
        if self.patch_training:
            pair_idx = idx // 4  # Which image pair
            patch_idx = idx % 4   # Which patch from that pair
        else:
            pair_idx = idx
            patch_idx = 0
        
        pair = self.pairs[pair_idx]
        
        # Load full resolution images
        iphone_img = self.process_image_4k(pair['iphone_file'])
        sony_img = self.process_image_4k(pair['sony_file'])
        
        if iphone_img is None or sony_img is None:
            return self.__getitem__((idx + 1) % len(self))
        
        if self.patch_training:
            # Extract matching patches from both images
            # Use the same random seed for both to get corresponding patches
            random.seed(idx)  # Ensure same patch location for both images
            iphone_patch = self.extract_patch(iphone_img, self.patch_size)
            random.seed(idx)  # Reset seed
            sony_patch = self.extract_patch(sony_img, self.patch_size)
            
            iphone_tensor = torch.FloatTensor(np.transpose(iphone_patch, (2, 0, 1)))
            sony_tensor = torch.FloatTensor(np.transpose(sony_patch, (2, 0, 1)))
        else:
            # Use full images (memory intensive!)
            iphone_tensor = torch.FloatTensor(np.transpose(iphone_img, (2, 0, 1)))
            sony_tensor = torch.FloatTensor(np.transpose(sony_img, (2, 0, 1)))
        
        return {
            'iphone': iphone_tensor,
            'sony': sony_tensor,
            'pair_id': pair_idx,
            'patch_id': patch_idx if self.patch_training else 0
        }

def load_all_luts():
    """Load all available LUTs"""
    lut_paths = [
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
        Path("luts/Neutral A7s3_65x.b64.txt"),
        Path("luts/Eterna A7s3_65x_Legacy.b64.txt"),
        Path("luts/Neutral A7s3_Legacy_65x.b64.txt"),
    ]
    
    print("🎨 Loading LUTs for 4K training...")
    available_luts = {}
    
    for lut_path in lut_paths:
        if lut_path.exists():
            print(f"✅ Found: {lut_path.name}")
            try:
                if lut_path.suffix == '.cube':
                    lut = parse_cube_lut(lut_path)
                elif lut_path.suffix == '.txt' and '.b64' in lut_path.name:
                    lut = decode_base64_lut(lut_path)
                else:
                    continue
                    
                if lut is not None:
                    clean_name = lut_path.stem.replace('.b64', '')
                    available_luts[clean_name] = lut
                    print(f"   Loaded: {lut.shape}")
            except Exception as e:
                print(f"   Error: {e}")
    
    if not available_luts:
        print("⚠️ No LUTs loaded")
        return None
    
    # Return best neutral LUT
    if 'Neutral A7s3_65x' in available_luts:
        primary_lut = available_luts['Neutral A7s3_65x']
        print(f"🎯 Primary LUT: Neutral A7s3_65x {primary_lut.shape}")
        return primary_lut, available_luts
    else:
        first_key = list(available_luts.keys())[0]
        primary_lut = available_luts[first_key]
        print(f"🎯 Primary LUT: {first_key} {primary_lut.shape}")
        return primary_lut, available_luts

def train_4k_model():
    """Train model with 4K patches"""
    print("🎬 TRAINING CINEMA MODEL V1.3b - FULL 4K (FIXED)")
    print("=" * 60)
    print("Training on actual 4K resolution with patch-based learning")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    if device.type == 'cpu':
        print("⚠️  WARNING: Training on CPU will be very slow for 4K!")
        print("   Consider using a GPU or reducing patch size")
    
    # Load LUTs
    lut_result = load_all_luts()
    if lut_result is None:
        print("⚠️ Training without LUT")
        reference_lut = None
    else:
        reference_lut, all_luts = lut_result
        print(f"✅ Using {len(all_luts)} LUTs")
    
    data_path = Path("data/results/simple_depth_analysis")
    
    # 4K patch-based dataset
    print(f"\n📊 DATASET CONFIGURATION:")
    patch_size = 512  # Still manageable for training
    dataset = Full4KDataset(data_path, patch_training=True, patch_size=patch_size)
    
    # Smaller batch size for memory management
    batch_size = 1 if device.type == 'cpu' else 2
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    print(f"   Source resolution: 4K (patches extracted during training)")
    print(f"   Patch size: {patch_size}x{patch_size}")
    print(f"   Patches per image pair: 4")
    print(f"   Total training patches: {len(dataset)}")
    print(f"   Batch size: {batch_size}")
    
    model = CinemaTransformV1_3b(reference_lut=reference_lut).to(device)
    
    # Conservative optimizer for 4K training
    optimizer = optim.Adam(model.parameters(), lr=0.00005)  # Lower LR
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0.000005)
    
    print(f"🔧 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    num_epochs = 30  # Fewer epochs since we have more patches
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            iphone_imgs = batch['iphone'].to(device)
            sony_imgs = batch['sony'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predicted = model(iphone_imgs)
            
            # Multi-component loss
            mse_loss = F.mse_loss(predicted, sony_imgs)
            l1_loss = F.l1_loss(predicted, sony_imgs)
            
            # Combined loss (simpler for 4K training)
            total_loss_batch = mse_loss + 0.1 * l1_loss
            
            total_loss_batch.backward()
            
            # Gradient clipping is crucial for 4K training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            
            total_loss += total_loss_batch.item()
            num_batches += 1
            
            # Memory cleanup
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache() if device.type == 'cuda' else gc.collect()
            
            # Progress update for long training
            if batch_idx % 20 == 0:
                print(f"    Batch {batch_idx}/{len(dataloader)}, Loss: {total_loss_batch.item():.6f}")
        
        scheduler.step()
        avg_loss = total_loss / num_batches
        
        if epoch % 3 == 0 or epoch == num_epochs - 1:
            print(f"\n📊 Epoch {epoch + 1}/{num_epochs}:")
            print(f"     Loss: {avg_loss:.6f}")
            print(f"     LR: {scheduler.get_last_lr()[0]:.6f}")
            
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
                'training_resolution': '4K_patches',
                'patch_size': patch_size
            }, Path("data/cinema_v1_3b_4k_model.pth"))
    
    print(f"\n✅ 4K Training complete!")
    print(f"🎯 Best loss: {best_loss:.6f}")
    
    return model

def test_4k_model():
    """Test 4K model at full resolution"""
    print("\n🧪 TESTING 4K CINEMA MODEL V1.3b")
    print("=" * 60)
    
    model_path = Path("data/cinema_v1_3b_4k_model.pth")
    if not model_path.exists():
        print("❌ No 4K model found. Train first!")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load LUTs
    lut_result = load_all_luts()
    reference_lut = None
    if checkpoint.get('has_lut', False) and lut_result is not None:
        reference_lut, all_luts = lut_result
    
    model = CinemaTransformV1_3b(reference_lut=reference_lut).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    patch_size = checkpoint.get('patch_size', 512)
    print(f"✅ Model loaded (epoch {checkpoint['epoch']}, trained on {patch_size}x{patch_size} patches)")
    
    # Print learned parameters
    print(f"\n📊 Learned parameters:")
    print(f"   Saturation: {model.saturation.item():.3f}")
    print(f"   Warmth: {model.warmth.item():.3f}")
    if reference_lut is not None:
        print(f"   LUT blend: {model.lut_blend.item():.3f}")
    print(f"   ML residual: {model.ml_enhancement.residual_strength.item():.3f}")
    
    # Test on 5 images at FULL resolution
    training_pairs_dir = Path("data/training_pairs")
    iphone_files = list(training_pairs_dir.glob("iphone_*.dng"))[:5]
    
    results_dir = Path("data/results/cinema_v1_3b_4k_test")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🎯 PROCESSING 5 SAMPLES AT FULL 4K RESOLUTION:")
    
    for i, iphone_file in enumerate(iphone_files):
        print(f"\n📸 {i+1}/5: {iphone_file.name}")
        
        try:
            # Load at FULL 4K resolution with FIXED exposure settings
            with rawpy.imread(str(iphone_file)) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    output_bps=16,
                    no_auto_bright=False,  # ← FIXED: Enable auto brightness
                    user_flip=0
                )
            
            rgb_norm = np.clip(rgb.astype(np.float32) / 65535.0, 0, 1)
            
            # Add exposure compensation if still too dark
            if rgb_norm.mean() < 0.3:
                rgb_norm = np.clip(rgb_norm * 1.5, 0, 1)
                print(f"   📝 Applied 1.5x exposure compensation")
            
            h, w = rgb_norm.shape[:2]
            
            print(f"   📐 Source: {w}x{h} ({w*h/1000000:.1f}MP)")
            print(f"   📊 Exposure: mean={rgb_norm.mean():.3f}, max={rgb_norm.max():.3f}")
            
            # For very large images, we might need to process in tiles
            if w * h > 8000 * 8000:  # More than 64MP
                print(f"   ⚠️  Very large image, processing in tiles...")
                transformed_np = process_large_image_in_tiles(model, rgb_norm, device, tile_size=2048)
            else:
                # Process the full image
                rgb_tensor = torch.FloatTensor(np.transpose(rgb_norm, (2, 0, 1))).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    print(f"   🔄 Processing {w}x{h} image...")
                    transformed = model(rgb_tensor)
                    transformed_np = transformed.cpu().squeeze(0).numpy()
                    transformed_np = np.transpose(transformed_np, (1, 2, 0))
                    
                    # Clear GPU memory
                    del rgb_tensor, transformed
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
            
            # Save at maximum quality
            original_display = (rgb_norm * 255).astype(np.uint8)
            transformed_display = (transformed_np * 255).astype(np.uint8)
            
            # Create side-by-side comparison (resize for display if too large)
            display_width = min(2048, w)  # Max 2K width for comparison display
            if w > display_width:
                scale = display_width / w
                display_h = int(h * scale)
                original_resized = cv2.resize(original_display, (display_width, display_h))
                transformed_resized = cv2.resize(transformed_display, (display_width, display_h))
            else:
                original_resized = original_display
                transformed_resized = transformed_display
            
            comparison = np.hstack([
                cv2.cvtColor(original_resized, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(transformed_resized, cv2.COLOR_RGB2BGR)
            ])
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.8, min(2.0, display_width / 1000))
            thickness = max(2, int(display_width / 800))
            
            cv2.putText(comparison, "iPhone Original (4K Fixed)", (20, 60), font, font_scale, (0, 255, 0), thickness)
            cv2.putText(comparison, f"Cinema v1.3b (4K→4K)", (display_width + 20, 60), font, font_scale, (0, 255, 0), thickness)
            
            # Save comparison
            comparison_path = results_dir / f"v1_3b_4k_{iphone_file.stem}_comparison.jpg"
            cv2.imwrite(str(comparison_path), comparison, [cv2.IMWRITE_JPEG_QUALITY, 98])
            
            # Save full resolution transformed image separately
            full_res_path = results_dir / f"v1_3b_4k_{iphone_file.stem}_transformed.jpg"
            cv2.imwrite(str(full_res_path), cv2.cvtColor(transformed_display, cv2.COLOR_RGB2BGR), 
                       [cv2.IMWRITE_JPEG_QUALITY, 100])
            
            print(f"   ✅ Comparison: {comparison_path.name}")
            print(f"   ✅ Full 4K result: {full_res_path.name}")
            print(f"   📊 Output: {transformed_display.shape[1]}x{transformed_display.shape[0]}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n✅ 4K Test complete!")
    print(f"📁 Results: {results_dir}")
    print(f"🎯 All images processed at full 4K resolution with fixed exposure")

def process_large_image_in_tiles(model, image, device, tile_size=2048, overlap=64):
    """Process very large images in overlapping tiles"""
    h, w = image.shape[:2]
    result = np.zeros_like(image)
    
    # Calculate tile positions
    y_positions = list(range(0, h - tile_size + 1, tile_size - overlap))
    if y_positions[-1] + tile_size < h:
        y_positions.append(h - tile_size)
    
    x_positions = list(range(0, w - tile_size + 1, tile_size - overlap))
    if x_positions[-1] + tile_size < w:
        x_positions.append(w - tile_size)
    
    total_tiles = len(y_positions) * len(x_positions)
    tile_count = 0
    
    print(f"     Processing {total_tiles} tiles of {tile_size}x{tile_size}...")
    
    for y in y_positions:
        for x in x_positions:
            tile_count += 1
            
            # Extract tile
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            tile = image[y:y_end, x:x_end]
            
            # Pad if necessary
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded_tile = np.zeros((tile_size, tile_size, 3), dtype=tile.dtype)
                padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded_tile
            
            # Process tile
            tile_tensor = torch.FloatTensor(np.transpose(tile, (2, 0, 1))).unsqueeze(0).to(device)
            
            with torch.no_grad():
                transformed_tile = model(tile_tensor)
                transformed_tile_np = transformed_tile.cpu().squeeze(0).numpy()
                transformed_tile_np = np.transpose(transformed_tile_np, (1, 2, 0))
            
            # Place result (handle overlaps by averaging)
            actual_h = y_end - y
            actual_w = x_end - x
            result_tile = transformed_tile_np[:actual_h, :actual_w]
            
            if overlap > 0 and (y > 0 or x > 0):
                # Blend overlapping regions
                blend_y_start = overlap//2 if y > 0 else 0
                blend_x_start = overlap//2 if x > 0 else 0
                
                result[y + blend_y_start:y_end, x + blend_x_start:x_end] = \
                    (result[y + blend_y_start:y_end, x + blend_x_start:x_end] + 
                     result_tile[blend_y_start:actual_h, blend_x_start:actual_w]) / 2
            else:
                result[y:y_end, x:x_end] = result_tile
            
            if tile_count % 10 == 0:
                print(f"     Tile {tile_count}/{total_tiles}")
            
            # Clean up GPU memory
            del tile_tensor, transformed_tile
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    return result

def main():
    """Main entry point for 4K training"""
    print("🎬 CINEMA MODEL V1.3b - FULL 4K RESOLUTION (FIXED)")
    print("Fixed rawpy processing + proper exposure handling")
    print("\nFIXES APPLIED:")
    print("- ✅ Removed incompatible rawpy parameters")
    print("- ✅ Enabled auto brightness (no_auto_bright=False)")
    print("- ✅ Added exposure compensation for dark images")
    print("- ✅ Maintained full 4K processing pipeline")
    
    print("\nIMPORTANT NOTES:")
    print("- This will use much more memory than previous versions")
    print("- GPU recommended for reasonable training times")
    print("- Patch-based training extracts 512x512 patches from 4K images")
    print("- Testing processes full 4K resolution images")
    print("- Images should now have proper exposure (mean ~0.4-0.6)")
    
    print("\nChoose option:")
    print("1. Train 4K model (patch-based)")
    print("2. Test 4K model (full resolution)")
    print("3. Both")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice in ["1", "3"]:
        train_4k_model()
    
    if choice in ["2", "3"]:
        test_4k_model()

if __name__ == "__main__":
    main()
    print(f"   Warmth: {model.warmth.item():.3f}")
    if reference_lut is not None:
        print(f"   LUT blend: {model.lut_blend.item():.3f}")
    print(f"   ML residual: {model.ml_enhancement.residual_strength.item():.3f}")
    
    # Test on 5 images at FULL resolution
    training_pairs_dir = Path("data/training_pairs")
    iphone_files = list(training_pairs_dir.glob("iphone_*.dng"))[:5]
    
    results_dir = Path("data/results/cinema_v1_3b_4k_test")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🎯 PROCESSING 5 SAMPLES AT FULL 4K RESOLUTION:")
    
    for i, iphone_file in enumerate(iphone_files):
        print(f"\n📸 {i+1}/5: {iphone_file.name}")
        
        try:
            # Load at FULL 4K resolution with compatible settings
            with rawpy.imread(str(iphone_file)) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    output_bps=16,
                    no_auto_bright=True,
                    user_flip=0
                )
            
            rgb_norm = np.clip(rgb.astype(np.float32) / 65535.0, 0, 1)
            h, w = rgb_norm.shape[:2]
            
            print(f"   📐 Source: {w}x{h} ({w*h/1000000:.1f}MP)")
            
            # For very large images, we might need to process in tiles
            if w * h > 8000 * 8000:  # More than 64MP
                print(f"   ⚠️  Very large image, processing in tiles...")
                transformed_np = process_large_image_in_tiles(model, rgb_norm, device, tile_size=2048)
            else:
                # Process the full image
                rgb_tensor = torch.FloatTensor(np.transpose(rgb_norm, (2, 0, 1))).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    print(f"   🔄 Processing {w}x{h} image...")
                    transformed = model(rgb_tensor)
                    transformed_np = transformed.cpu().squeeze(0).numpy()
                    transformed_np = np.transpose(transformed_np, (1, 2, 0))
                    
                    # Clear GPU memory
                    del rgb_tensor, transformed
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
            
            # Save at maximum quality
            original_display = (rgb_norm * 255).astype(np.uint8)
            transformed_display = (transformed_np * 255).astype(np.uint8)
            
            # Create side-by-side comparison (resize for display if too large)
            display_width = min(2048, w)  # Max 2K width for comparison display
            if w > display_width:
                scale = display_width / w
                display_h = int(h * scale)
                original_resized = cv2.resize(original_display, (display_width, display_h))
                transformed_resized = cv2.resize(transformed_display, (display_width, display_h))
            else:
                original_resized = original_display
                transformed_resized = transformed_display
            
            comparison = np.hstack([
                cv2.cvtColor(original_resized, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(transformed_resized, cv2.COLOR_RGB2BGR)
            ])
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.8, min(2.0, display_width / 1000))
            thickness = max(2, int(display_width / 800))
            
            cv2.putText(comparison, "iPhone Original (4K)", (20, 60), font, font_scale, (0, 255, 0), thickness)
            cv2.putText(comparison, f"Cinema v1.3b (4K→4K)", (display_width + 20, 60), font, font_scale, (0, 255, 0), thickness)
            
            # Save comparison
            comparison_path = results_dir / f"v1_3b_4k_{iphone_file.stem}_comparison.jpg"
            cv2.imwrite(str(comparison_path), comparison, [cv2.IMWRITE_JPEG_QUALITY, 98])
            
            # Save full resolution transformed image separately
            full_res_path = results_dir / f"v1_3b_4k_{iphone_file.stem}_transformed.jpg"
            cv2.imwrite(str(full_res_path), cv2.cvtColor(transformed_display, cv2.COLOR_RGB2BGR), 
                       [cv2.IMWRITE_JPEG_QUALITY, 100])
            
            print(f"   ✅ Comparison: {comparison_path.name}")
            print(f"   ✅ Full 4K result: {full_res_path.name}")
            print(f"   📊 Output: {transformed_display.shape[1]}x{transformed_display.shape[0]}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n✅ 4K Test complete!")
    print(f"📁 Results: {results_dir}")
    print(f"🎯 All images processed at full 4K resolution")

def process_large_image_in_tiles(model, image, device, tile_size=2048, overlap=64):
    """Process very large images in overlapping tiles"""
    h, w = image.shape[:2]
    result = np.zeros_like(image)
    
    # Calculate tile positions
    y_positions = list(range(0, h - tile_size + 1, tile_size - overlap))
    if y_positions[-1] + tile_size < h:
        y_positions.append(h - tile_size)
    
    x_positions = list(range(0, w - tile_size + 1, tile_size - overlap))
    if x_positions[-1] + tile_size < w:
        x_positions.append(w - tile_size)
    
    total_tiles = len(y_positions) * len(x_positions)
    tile_count = 0
    
    print(f"     Processing {total_tiles} tiles of {tile_size}x{tile_size}...")
    
    for y in y_positions:
        for x in x_positions:
            tile_count += 1
            
            # Extract tile
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            tile = image[y:y_end, x:x_end]
            
            # Pad if necessary
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded_tile = np.zeros((tile_size, tile_size, 3), dtype=tile.dtype)
                padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded_tile
            
            # Process tile
            tile_tensor = torch.FloatTensor(np.transpose(tile, (2, 0, 1))).unsqueeze(0).to(device)
            
            with torch.no_grad():
                transformed_tile = model(tile_tensor)
                transformed_tile_np = transformed_tile.cpu().squeeze(0).numpy()
                transformed_tile_np = np.transpose(transformed_tile_np, (1, 2, 0))
            
            # Place result (handle overlaps by averaging)
            actual_h = y_end - y
            actual_w = x_end - x
            result_tile = transformed_tile_np[:actual_h, :actual_w]
            
            if overlap > 0 and (y > 0 or x > 0):
                # Blend overlapping regions
                blend_y_start = overlap//2 if y > 0 else 0
                blend_x_start = overlap//2 if x > 0 else 0
                
                result[y + blend_y_start:y_end, x + blend_x_start:x_end] = \
                    (result[y + blend_y_start:y_end, x + blend_x_start:x_end] + 
                     result_tile[blend_y_start:actual_h, blend_x_start:actual_w]) / 2
            else:
                result[y:y_end, x:x_end] = result_tile
            
            if tile_count % 10 == 0:
                print(f"     Tile {tile_count}/{total_tiles}")
            
            # Clean up GPU memory
            del tile_tensor, transformed_tile
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    return result

def main():
    """Main entry point for 4K training"""
    print("🎬 CINEMA MODEL V1.3b - FULL 4K RESOLUTION (FIXED)")
    print("Compatible rawpy processing for 4K training")
    print("\nIMPORTANT NOTES:")
    print("- Fixed rawpy parameter compatibility issues")
    print("- This will use much more memory than previous versions")
    print("- GPU recommended for reasonable training times")
    print("- Patch-based training extracts 512x512 patches from 4K images")
    print("- Testing processes full 4K resolution images")
    
    print("\nChoose option:")
    print("1. Train 4K model (patch-based)")
    print("2. Test 4K model (full resolution)")
    print("3. Both")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice in ["1", "3"]:
        train_4k_model()
    
    if choice in ["2", "3"]:
        test_4k_model()

if __name__ == "__main__":
    main()
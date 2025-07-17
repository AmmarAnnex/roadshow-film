#!/usr/bin/env python3
"""
Cinema Model v1.4m Improved - Conservative Exposure Fix
Fixes the massive overbrightening issue while keeping exposure correction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedExposureFixedColorTransform(nn.Module):
    """Improved v1.4m with much more conservative parameters"""
    
    def __init__(self):
        super(ImprovedExposureFixedColorTransform, self).__init__()
        
        # Same CNN architecture but conservative
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv_out = nn.Conv2d(16, 3, 3, padding=1)
        
        # Professional color matrix (much smaller initial values)
        self.color_matrix = nn.Parameter(torch.eye(3) + torch.randn(3, 3) * 0.002)  # 5x smaller
        self.color_bias = nn.Parameter(torch.randn(3) * 0.001)  # 5x smaller
        
        # FIXED: Much more conservative exposure parameters
        self.global_exposure = nn.Parameter(torch.tensor(1.01))   # Was 1.05 -> 1.01 (tiny boost)
        self.shadows = nn.Parameter(torch.tensor(0.01))           # Was 0.05 -> 0.01 (gentle lift)
        self.mids = nn.Parameter(torch.tensor(1.0))               # Neutral
        self.highlights = nn.Parameter(torch.tensor(0.98))        # Slight protection
        
        # Much more conservative color grading
        self.contrast = nn.Parameter(torch.tensor(1.01))          # Was 1.03 -> 1.01
        self.saturation = nn.Parameter(torch.tensor(1.02))        # Was 1.08 -> 1.02
        self.warmth = nn.Parameter(torch.tensor(0.002))           # Was 0.01 -> 0.002
        
        # Much smaller residual strength (like v1.4 4K)
        self.residual_strength = nn.Parameter(torch.tensor(0.02)) # Was 0.05 -> 0.02
        
    def apply_exposure_correction(self, x):
        """MUCH more conservative exposure handling"""
        exposure_factor = torch.clamp(self.global_exposure, 0.98, 1.03)  # Tight bounds
        return x * exposure_factor
    
    def apply_color_matrix(self, x):
        """Apply professional 3x3 color matrix with tighter bounds"""
        matrix = torch.clamp(self.color_matrix, 0.95, 1.05)    # Much tighter bounds
        bias = torch.clamp(self.color_bias, -0.005, 0.005)     # Much smaller bias
        
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1)
        
        # Matrix multiplication
        transformed = torch.bmm(matrix.unsqueeze(0).expand(b, -1, -1), x_flat)
        
        # Add bias
        bias_expanded = bias.view(1, 3, 1).expand(b, -1, x_flat.size(2))
        transformed = transformed + bias_expanded
        
        return transformed.view(b, c, h, w)
    
    def apply_professional_tone_curve(self, x):
        """Much more gentle tone curve"""
        shadows_c = torch.clamp(self.shadows, 0.0, 0.02)        # Much smaller range
        mids_c = torch.clamp(self.mids, 0.98, 1.02)             # Very tight
        highlights_c = torch.clamp(self.highlights, 0.95, 1.0)  # Protect highlights
        
        # Calculate luminance for masking
        luma = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        
        # Create smooth masks
        shadow_mask = (1 - luma).clamp(0, 1) ** 1.5
        highlight_mask = luma.clamp(0, 1) ** 1.5
        mid_mask = 1 - shadow_mask - highlight_mask
        
        # Much gentler adjustments
        shadow_adj = shadows_c * shadow_mask * 0.02           # Was 0.05 -> 0.02
        mid_adj = (mids_c - 1.0) * mid_mask * 0.2             # Was 0.5 -> 0.2
        highlight_adj = (highlights_c - 1.0) * highlight_mask * 0.1  # Was 0.3 -> 0.1
        
        return x + shadow_adj + mid_adj + highlight_adj
    
    def apply_color_grading(self, x):
        """Much more conservative color grading"""
        contrast_c = torch.clamp(self.contrast, 0.98, 1.05)      # Tighter bounds
        saturation_c = torch.clamp(self.saturation, 0.95, 1.1)   # Much smaller range
        warmth_c = torch.clamp(self.warmth, -0.005, 0.005)       # Tiny warmth
        
        # Contrast adjustment around 0.5 middle gray
        x_contrast = (x - 0.5) * contrast_c + 0.5
        
        # Saturation adjustment
        luma = 0.299 * x_contrast[:, 0:1] + 0.587 * x_contrast[:, 1:2] + 0.114 * x_contrast[:, 2:3]
        x_saturated = luma + (x_contrast - luma) * saturation_c
        
        # Warmth adjustment (minimal)
        warmth_matrix = torch.tensor([
            [1.0 + warmth_c, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0 - warmth_c]
        ], device=x.device, dtype=x.dtype)
        
        b, c, h, w = x_saturated.shape
        x_flat = x_saturated.view(b, c, -1)
        x_warm = torch.bmm(warmth_matrix.unsqueeze(0).expand(b, -1, -1), x_flat)
        
        return x_warm.view(b, c, h, w)
    
    def forward(self, x):
        # Store original for residual
        x_original = x
        
        # Apply GENTLE exposure correction
        x = self.apply_exposure_correction(x)
        
        # Apply color matrix
        x = self.apply_color_matrix(x)
        
        # Professional tone curve (much gentler)
        x = self.apply_professional_tone_curve(x)
        
        # Color grading (much more conservative)
        x = self.apply_color_grading(x)
        
        # CNN residual refinement (smaller strength)
        residual = F.relu(self.conv1(x_original))
        residual = F.relu(self.conv2(residual))
        residual = F.relu(self.conv3(residual))
        residual = F.relu(self.conv4(residual))
        residual = torch.tanh(self.conv_out(residual))
        
        # Apply residual with much smaller strength
        strength = torch.clamp(self.residual_strength, 0.01, 0.03)  # Much smaller
        x_final = x + strength * residual
        
        return torch.clamp(x_final, 0, 1)


def train_v14m_improved():
    """Train the improved v1.4m model with conservative parameters"""
    print("🎬 TRAINING IMPROVED CINEMA MODEL V1.4M")
    print("=" * 50)
    print("Conservative exposure fix - no more overbrightening!")
    
    # Use the improved model
    model = ImprovedExposureFixedColorTransform()
    
    # Print initial parameters to verify they're conservative
    print(f"📊 Initial Parameters:")
    print(f"   Global exposure: {model.global_exposure.item():.3f}")
    print(f"   Shadows: {model.shadows.item():.3f}")
    print(f"   Residual strength: {model.residual_strength.item():.3f}")
    
    # Rest of training code would be the same as before...
    # This is just showing the improved model architecture


if __name__ == "__main__":
    train_v14m_improved()
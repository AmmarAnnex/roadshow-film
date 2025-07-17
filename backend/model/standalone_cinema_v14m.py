#!/usr/bin/env python3
"""
Cinema Model v1.4m - Standalone Version for Deployment
No external dependencies - everything included in this file
"""

import cv2
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

class ExposureFixedColorTransform(nn.Module):
    """Professional color transform with fixed exposure handling"""
    
    def __init__(self):
        super(ExposureFixedColorTransform, self).__init__()
        
        # Multi-scale color processing
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv_out = nn.Conv2d(16, 3, 3, padding=1)
        
        # Professional color matrix (3x3)
        self.color_matrix = nn.Parameter(torch.eye(3) * 1.0 + torch.randn(3, 3) * 0.01)
        self.color_bias = nn.Parameter(torch.randn(3) * 0.005)
        
        # FIXED: Proper exposure and tone curve parameters
        self.global_exposure = nn.Parameter(torch.tensor(1.05))  # Slight brightness boost
        self.shadows = nn.Parameter(torch.tensor(0.05))          # Lift shadows
        self.mids = nn.Parameter(torch.tensor(1.0))              # Neutral mids
        self.highlights = nn.Parameter(torch.tensor(0.95))       # Protect highlights
        
        # Color grading controls
        self.contrast = nn.Parameter(torch.tensor(1.03))
        self.saturation = nn.Parameter(torch.tensor(1.08))
        self.warmth = nn.Parameter(torch.tensor(0.01))
        
        # FIXED: More conservative residual strength
        self.residual_strength = nn.Parameter(torch.tensor(0.05))
        
    def apply_exposure_correction(self, x):
        """FIXED: Proper exposure handling to prevent underexposure"""
        exposure_factor = torch.clamp(self.global_exposure, 0.9, 1.2)  # Prevent darkening
        return x * exposure_factor
    
    def apply_color_matrix(self, x):
        """Apply professional 3x3 color matrix"""
        matrix = torch.clamp(self.color_matrix, 0.8, 1.2)  # Conservative bounds
        bias = torch.clamp(self.color_bias, -0.02, 0.02)   # Small bias
        
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1)
        
        # Matrix multiplication
        transformed = torch.bmm(matrix.unsqueeze(0).expand(b, -1, -1), x_flat)
        
        # Add bias
        bias_expanded = bias.view(1, 3, 1).expand(b, -1, x_flat.size(2))
        transformed = transformed + bias_expanded
        
        return transformed.view(b, c, h, w)
    
    def apply_professional_tone_curve(self, x):
        """Professional tone curve with fixed exposure"""
        shadows_c = torch.clamp(self.shadows, 0.0, 0.1)      # Only lift, never crush
        mids_c = torch.clamp(self.mids, 0.95, 1.05)          # Conservative mids
        highlights_c = torch.clamp(self.highlights, 0.9, 1.0) # Protect highlights
        
        # Calculate luminance for masking
        luma = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        
        # Create smooth masks
        shadow_mask = (1 - luma).clamp(0, 1) ** 1.5
        highlight_mask = luma.clamp(0, 1) ** 1.5
        mid_mask = 1 - shadow_mask - highlight_mask
        
        # Apply adjustments (more conservative)
        shadow_adj = shadows_c * shadow_mask * 0.05   # Gentle shadow lift
        mid_adj = (mids_c - 1.0) * mid_mask * 0.5     # Conservative mid adjustment
        highlight_adj = (highlights_c - 1.0) * highlight_mask * 0.3  # Protect highlights
        
        return x + shadow_adj + mid_adj + highlight_adj
    
    def apply_color_grading(self, x):
        """Conservative color grading"""
        contrast_c = torch.clamp(self.contrast, 0.95, 1.1)    # Conservative contrast
        saturation_c = torch.clamp(self.saturation, 0.9, 1.2) # Conservative saturation
        warmth_c = torch.clamp(self.warmth, -0.01, 0.02)      # Subtle warmth
        
        # Contrast adjustment around 0.5 middle gray
        x_contrast = (x - 0.5) * contrast_c + 0.5
        
        # Saturation adjustment
        luma = 0.299 * x_contrast[:, 0:1] + 0.587 * x_contrast[:, 1:2] + 0.114 * x_contrast[:, 2:3]
        x_saturated = luma + (x_contrast - luma) * saturation_c
        
        # Warmth adjustment (very subtle)
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
        
        # FIXED: Apply exposure correction first to prevent underexposure
        x = self.apply_exposure_correction(x)
        
        # Apply color matrix
        x = self.apply_color_matrix(x)
        
        # Professional tone curve
        x = self.apply_professional_tone_curve(x)
        
        # Color grading
        x = self.apply_color_grading(x)
        
        # CNN residual refinement (conservative)
        residual = F.relu(self.conv1(x_original))
        residual = F.relu(self.conv2(residual))
        residual = F.relu(self.conv3(residual))
        residual = F.relu(self.conv4(residual))
        residual = torch.tanh(self.conv_out(residual))
        
        # Apply residual with conservative strength
        strength = torch.clamp(self.residual_strength, 0.02, 0.08)
        x_final = x + strength * residual
        
        return torch.clamp(x_final, 0, 1)
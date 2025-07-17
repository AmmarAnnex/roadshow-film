import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import tempfile
import os
from model.cinema_v1_4m import ExposureFixedColorTransform

class CinemaProcessor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()
        
    def load_model(self):
        """Load the trained v1.4m model"""
        try:
            model = ExposureFixedColorTransform().to(self.device)
            
            # Try multiple model paths
            model_paths = [
                Path("models/cinema_v1_4m_model.pth"),
                Path("model/cinema_v1_4m_model.pth"),
                Path("../models/cinema_v1_4m_model.pth")
            ]
            
            checkpoint = None
            for path in model_paths:
                if path.exists():
                    checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                    break
            
            if checkpoint is None:
                print("Warning: No model checkpoint found, using untrained model")
                return model
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            print(f"Model loaded on {self.device}")
            return model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def transform_video(self, input_path, intensity=0.8, exposure=0.5, color=0.7):
        """Transform video using cinema model"""
        if self.model is None:
            raise Exception("Model not loaded")
        
        # Create output path
        output_path = input_path.replace('.mp4', '_roadshow.mp4')
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception("Could not open video file")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame, intensity, exposure, color)
                
                # Write frame
                out.write(processed_frame)
                
                frame_count += 1
                if frame_count % 30 == 0:  # Progress every 30 frames
                    progress = (frame_count / total_frames) * 100
                    print(f"Processing: {progress:.1f}%")
        
        finally:
            cap.release()
            out.release()
        
        print(f"Video processed: {frame_count} frames")
        return output_path
    
    def process_frame(self, frame, intensity, exposure, color):
        """Process a single frame through the model"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0,1]
            frame_normalized = rgb_frame.astype(np.float32) / 255.0
            
            # Resize to model input size (768x768)
            original_size = frame_normalized.shape[:2]
            frame_resized = cv2.resize(frame_normalized, (768, 768))
            
            # Convert to tensor
            frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            # Apply model transformation
            with torch.no_grad():
                transformed_tensor = self.model(frame_tensor)
            
            # Convert back to numpy
            transformed_frame = transformed_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            # Resize back to original size
            transformed_frame = cv2.resize(transformed_frame, (original_size[1], original_size[0]))
            
            # Apply intensity blending
            blended_frame = self.blend_frames(frame_normalized, transformed_frame, intensity)
            
            # Apply additional effects
            final_frame = self.apply_effects(blended_frame, exposure, color)
            
            # Convert back to BGR and [0,255]
            final_frame = np.clip(final_frame * 255, 0, 255).astype(np.uint8)
            final_frame = cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR)
            
            return final_frame
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame  # Return original frame on error
    
    def blend_frames(self, original, transformed, intensity):
        """Blend original and transformed frames"""
        return original * (1 - intensity) + transformed * intensity
    
    def apply_effects(self, frame, exposure, color):
        """Apply additional cinematic effects"""
        # Exposure adjustment
        if exposure != 0.5:
            exposure_factor = 0.8 + (exposure * 0.4)  # Range: 0.8 to 1.2
            frame = np.clip(frame * exposure_factor, 0, 1)
        
        # Color grading
        if color != 0.5:
            # Simple color temperature adjustment
            color_factor = 0.9 + (color * 0.2)  # Range: 0.9 to 1.1
            frame[:, :, 0] *= color_factor  # Adjust red channel
            frame[:, :, 2] *= (2 - color_factor)  # Inverse adjust blue channel
        
        return np.clip(frame, 0, 1)
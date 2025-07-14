#!/usr/bin/env python3
"""
iPhone to Cinema Transformation Engine
Uses your training data to transform iPhone footage to match Sony+Zeiss characteristics
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
from typing import Dict, List, Tuple, Optional

class CinemaTransformDataset(Dataset):
    """Dataset for iPhone→Sony transformation training"""
    
    def __init__(self, training_data_path: Path, transform_size: Tuple[int, int] = (512, 512)):
        self.data_path = training_data_path
        self.transform_size = transform_size
        self.pairs = self.load_training_pairs()
        
    def load_training_pairs(self) -> List[Dict]:
        """Load training pairs from your collected data"""
        metadata_file = self.data_path / "depth_metadata.json"
        
        if not metadata_file.exists():
            raise FileNotFoundError(f"Training metadata not found: {metadata_file}")
            
        with open(metadata_file, 'r') as f:
            pairs = json.load(f)
            
        print(f"📊 Loaded {len(pairs)} training pairs")
        return pairs
    
    def process_raw_image(self, file_path: str) -> np.ndarray:
        """Process RAW file to normalized RGB"""
        try:
            with rawpy.imread(file_path) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    output_bps=16,
                    no_auto_bright=True,
                    user_flip=0
                )
            
            # Normalize to [0, 1] and resize
            rgb_norm = rgb.astype(np.float32) / 65535.0
            rgb_resized = cv2.resize(rgb_norm, self.transform_size)
            
            # Convert to CHW format for PyTorch
            return np.transpose(rgb_resized, (2, 0, 1))
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Load iPhone and Sony images
        iphone_img = self.process_raw_image(pair['iphone_file'])
        sony_img = self.process_raw_image(pair['sony_file'])
        
        if iphone_img is None or sony_img is None:
            # Return next valid pair if this one fails
            return self.__getitem__((idx + 1) % len(self.pairs))
        
        # Extract metadata features
        iphone_meta = pair['iphone_metadata']
        sony_meta = pair['sony_metadata']
        analysis = pair['analysis']
        
        # Create metadata feature vector
        metadata_features = [
            float(iphone_meta['iso']) / 1000.0 if iphone_meta['iso'] != 'Unknown' else 0.0,
            float(analysis['color_difference']) / 100.0,
            float(analysis['dynamic_range_difference']) / 20.0,
            float(analysis['depth_difference']) / 100.0,
            float(analysis['focus_variance_difference']) / 1000.0,
            float(analysis['sharpness_difference']) / 100.0
        ]
        
        return {
            'iphone': torch.FloatTensor(iphone_img),
            'sony': torch.FloatTensor(sony_img),
            'metadata': torch.FloatTensor(metadata_features),
            'pair_id': idx
        }

class CinemaTransformNet(nn.Module):
    """Neural network to transform iPhone images to Sony+Zeiss characteristics"""
    
    def __init__(self, metadata_dim: int = 6):
        super(CinemaTransformNet, self).__init__()
        
        # Encoder: Extract iPhone image features
        self.encoder = nn.Sequential(
            # Input: 3 x 512 x 512
            nn.Conv2d(3, 64, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128 x 256 x 256
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 256 x 128 x 128
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Metadata processing
        self.metadata_net = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 512),
            nn.ReLU(inplace=True)
        )
        
        # Transformer: Apply cinema characteristics
        self.transformer = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Decoder: Reconstruct to Sony characteristics
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 128 x 256 x 256
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 64 x 512 x 512
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 3, 7, padding=3),  # 3 x 512 x 512
            nn.Sigmoid()  # Output in [0, 1] range
        )
    
    def forward(self, iphone_img, metadata):
        # Encode iPhone image
        features = self.encoder(iphone_img)  # [B, 512, 128, 128]
        
        # Process metadata
        meta_features = self.metadata_net(metadata)  # [B, 512]
        meta_features = meta_features.unsqueeze(-1).unsqueeze(-1)  # [B, 512, 1, 1]
        meta_features = meta_features.expand(-1, -1, features.size(2), features.size(3))  # [B, 512, 128, 128]
        
        # Combine image features with metadata
        combined_features = features + meta_features
        
        # Transform
        transformed = self.transformer(combined_features)
        
        # Decode to Sony characteristics
        output = self.decoder(transformed)
        
        return output

class CinemaTransformTrainer:
    """Trainer for the cinema transformation model"""
    
    def __init__(self, data_path: Path, device: str = 'cpu'):
        self.device = torch.device(device)
        self.data_path = data_path
        
        # Initialize dataset and dataloader
        self.dataset = CinemaTransformDataset(data_path / "results" / "simple_depth_analysis")
        self.dataloader = DataLoader(self.dataset, batch_size=2, shuffle=True, num_workers=0)
        
        # Initialize model
        self.model = CinemaTransformNet().to(self.device)
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        print(f"🎬 Cinema Transform Trainer initialized")
        print(f"📊 Dataset size: {len(self.dataset)} pairs")
        print(f"🔧 Device: {self.device}")
    
    def perceptual_loss(self, pred, target):
        """Simple perceptual loss using gradients"""
        # Compute gradients
        pred_grad_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        pred_grad_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        
        target_grad_x = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        target_grad_y = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
        
        grad_loss = self.l1_loss(pred_grad_x, target_grad_x) + self.l1_loss(pred_grad_y, target_grad_y)
        return grad_loss
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.dataloader):
            iphone_imgs = batch['iphone'].to(self.device)
            sony_imgs = batch['sony'].to(self.device)
            metadata = batch['metadata'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred_sony = self.model(iphone_imgs, metadata)
            
            # Compute losses
            mse_loss = self.mse_loss(pred_sony, sony_imgs)
            l1_loss = self.l1_loss(pred_sony, sony_imgs)
            perceptual_loss = self.perceptual_loss(pred_sony, sony_imgs)
            
            # Combined loss
            total_loss_batch = mse_loss + 0.1 * l1_loss + 0.01 * perceptual_loss
            
            # Backward pass
            total_loss_batch.backward()
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            num_batches += 1
            
            if batch_idx % 5 == 0:
                print(f"    Batch {batch_idx}: MSE={mse_loss.item():.6f}, L1={l1_loss.item():.6f}, Perceptual={perceptual_loss.item():.6f}")
        
        avg_loss = total_loss / max(num_batches, 1)
        print(f"  📊 Epoch {epoch} Average Loss: {avg_loss:.6f}")
        return avg_loss
    
    def save_model(self, epoch: int, loss: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'timestamp': datetime.now().isoformat()
        }
        
        save_path = self.data_path / f"cinema_transform_model_epoch_{epoch}.pth"
        torch.save(checkpoint, save_path)
        print(f"  💾 Model saved: {save_path}")
    
    def train(self, num_epochs: int = 50):
        """Train the model"""
        print(f"\n🎬 TRAINING CINEMA TRANSFORMATION MODEL")
        print("=" * 50)
        print(f"Training iPhone→Sony+Zeiss transformation for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            print(f"\n🎬 Epoch {epoch + 1}/{num_epochs}")
            
            loss = self.train_epoch(epoch + 1)
            
            # Save model every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_model(epoch + 1, loss)
        
        # Save final model
        self.save_model(num_epochs, loss)
        
        print(f"\n✅ Training complete!")
        print(f"🎯 Final model saved for iPhone→Cinema transformation")

class CinemaTransformInference:
    """Inference engine for transforming iPhone images to cinema characteristics"""
    
    def __init__(self, model_path: Path, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model = CinemaTransformNet().to(self.device)
        
        # Load trained model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"🎬 Cinema Transform Inference loaded")
        print(f"📊 Model trained for {checkpoint['epoch']} epochs")
        print(f"📊 Final loss: {checkpoint['loss']:.6f}")
    
    def transform_image(self, iphone_image_path: str, target_metadata: Dict = None) -> np.ndarray:
        """Transform iPhone image to Sony+Zeiss characteristics"""
        # Process iPhone image
        with rawpy.imread(iphone_image_path) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                output_bps=16,
                no_auto_bright=True,
                user_flip=0
            )
        
        # Normalize and prepare for model
        rgb_norm = rgb.astype(np.float32) / 65535.0
        rgb_resized = cv2.resize(rgb_norm, (512, 512))
        rgb_tensor = torch.FloatTensor(np.transpose(rgb_resized, (2, 0, 1))).unsqueeze(0).to(self.device)
        
        # Default metadata (you can customize this)
        if target_metadata is None:
            target_metadata = [0.4, 0.16, 0.2, 0.4, 0.3, 0.04]  # Average values from training
        
        metadata_tensor = torch.FloatTensor(target_metadata).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            transformed = self.model(rgb_tensor, metadata_tensor)
        
        # Convert back to numpy
        output_np = transformed.cpu().squeeze(0).numpy()
        output_np = np.transpose(output_np, (1, 2, 0))
        
        # Convert to 8-bit for display
        output_8bit = (output_np * 255).astype(np.uint8)
        
        return output_8bit

def main():
    """Main training pipeline"""
    data_path = Path("data")
    
    # Check if training data exists
    training_metadata = data_path / "results" / "simple_depth_analysis" / "depth_metadata.json"
    if not training_metadata.exists():
        print("❌ Training data not found!")
        print("Run simple_depth_pipeline.py first to generate training data")
        return
    
    # Initialize trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = CinemaTransformTrainer(data_path, device)
    
    # Train model
    trainer.train(num_epochs=20)  # Start with 20 epochs
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. Test the trained model on new iPhone images")
    print("2. Fine-tune with more training data")
    print("3. Implement video processing pipeline")
    print("4. Build the full Roadshow Film application")

if __name__ == "__main__":
    main()
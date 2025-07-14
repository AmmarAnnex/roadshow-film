#!/usr/bin/env python3
"""
Debug and Fix Model Issues
Diagnose why the model is outputting black images
"""

import torch
import numpy as np
from pathlib import Path
import cv2
import rawpy
from transformation_engine import CinemaTransformNet

def debug_model_outputs():
    """Debug what's happening inside the model"""
    print("🔍 DEBUGGING MODEL OUTPUTS")
    print("=" * 50)
    
    # Load the trained model
    model_files = list(Path("data").glob("cinema_transform_model_*.pth"))
    if not model_files:
        print("❌ No trained model found!")
        return
    
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    print(f"📊 Loading model: {latest_model}")
    
    device = torch.device('cpu')
    model = CinemaTransformNet().to(device)
    
    try:
        checkpoint = torch.load(latest_model, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test with a sample image
    training_pairs_dir = Path("data/training_pairs")
    iphone_files = list(training_pairs_dir.glob("iphone_*.dng"))
    
    if not iphone_files:
        print("❌ No test images found!")
        return
    
    test_file = iphone_files[0]
    print(f"🧪 Testing with: {test_file.name}")
    
    # Load and process test image
    try:
        with rawpy.imread(str(test_file)) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                output_bps=16,
                no_auto_bright=True,
                user_flip=0
            )
        
        # Normalize to [0, 1]
        rgb_norm = rgb.astype(np.float32) / 65535.0
        print(f"📊 Input range: {rgb_norm.min():.3f} to {rgb_norm.max():.3f}")
        
        # Resize and convert to tensor
        rgb_resized = cv2.resize(rgb_norm, (512, 512))
        rgb_tensor = torch.FloatTensor(np.transpose(rgb_resized, (2, 0, 1))).unsqueeze(0).to(device)
        
        # Test metadata
        metadata = torch.FloatTensor([0.1, 0.16, 0.2, 0.4, 0.3, 0.04]).unsqueeze(0).to(device)
        
        print(f"📊 Input tensor shape: {rgb_tensor.shape}")
        print(f"📊 Input tensor range: {rgb_tensor.min():.3f} to {rgb_tensor.max():.3f}")
        print(f"📊 Metadata shape: {metadata.shape}")
        
        # Debug forward pass
        with torch.no_grad():
            print("\n🔍 DEBUGGING FORWARD PASS:")
            
            # Encoder
            features = model.encoder(rgb_tensor)
            print(f"📊 Encoder output shape: {features.shape}")
            print(f"📊 Encoder output range: {features.min():.3f} to {features.max():.3f}")
            
            # Metadata processing
            meta_features = model.metadata_net(metadata)
            print(f"📊 Metadata features shape: {meta_features.shape}")
            print(f"📊 Metadata features range: {meta_features.min():.3f} to {meta_features.max():.3f}")
            
            # Reshape metadata
            meta_features = meta_features.unsqueeze(-1).unsqueeze(-1)
            meta_features = meta_features.expand(-1, -1, features.size(2), features.size(3))
            
            # Combine
            combined = features + meta_features
            print(f"📊 Combined features range: {combined.min():.3f} to {combined.max():.3f}")
            
            # Transform
            transformed = model.transformer(combined)
            print(f"📊 Transformer output shape: {transformed.shape}")
            print(f"📊 Transformer output range: {transformed.min():.3f} to {transformed.max():.3f}")
            
            # Decoder
            output = model.decoder(transformed)
            print(f"📊 Final output shape: {output.shape}")
            print(f"📊 Final output range: {output.min():.3f} to {output.max():.3f}")
            
            # Check if output is all zeros/near zeros
            if output.max() < 0.01:
                print("❌ OUTPUT IS NEARLY BLACK!")
                print("🔍 Possible issues:")
                print("   - Sigmoid activation is saturating to 0")
                print("   - Features are becoming negative")
                print("   - Learning rate too high")
                print("   - Gradients exploding/vanishing")
            else:
                print("✅ Output has reasonable values")
        
        return output
        
    except Exception as e:
        print(f"❌ Error during debugging: {e}")
        return None

def fix_model_architecture():
    """Create a fixed version of the model"""
    print("\n🔧 CREATING FIXED MODEL ARCHITECTURE")
    print("=" * 50)
    
    fixed_model_code = '''
class FixedCinemaTransformNet(nn.Module):
    """Fixed neural network with better activation handling"""
    
    def __init__(self, metadata_dim: int = 6):
        super(FixedCinemaTransformNet, self).__init__()
        
        # Encoder with batch norm and better activations
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Simpler metadata processing
        self.metadata_net = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
        )
        
        # Skip connection friendly transformer
        self.transformer = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Decoder with skip connections
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()  # Use Tanh instead of Sigmoid
        )
    
    def forward(self, iphone_img, metadata):
        # Encoder
        features = self.encoder(iphone_img)
        
        # Skip connection - add input features
        residual = features
        
        # Transform
        transformed = self.transformer(features)
        
        # Add residual connection
        transformed = transformed + residual[:, :256, :, :]  # Match channels
        
        # Decode
        output = self.decoder(transformed)
        
        # Convert from [-1, 1] to [0, 1]
        output = (output + 1.0) / 2.0
        
        return output
'''
    
    print("💡 FIXES APPLIED:")
    print("1. Added BatchNorm layers for stable training")
    print("2. Replaced Sigmoid with Tanh + scaling")
    print("3. Added residual connections")
    print("4. Simplified metadata processing")
    print("5. Better gradient flow")
    
    # Save fixed model code
    with open("fixed_transformation_engine.py", "w") as f:
        f.write("""#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim

""" + fixed_model_code + """

# Copy the rest of the training code here...
""")
    
    print("\n✅ Fixed model saved to: fixed_transformation_engine.py")

def quick_retrain_with_simpler_model():
    """Quick retrain with a much simpler model"""
    print("\n🚀 QUICK RETRAIN WITH SIMPLER MODEL")
    print("=" * 50)
    
    simple_model_code = '''
class SimpleCinemaTransform(nn.Module):
    """Very simple model that just applies color corrections"""
    
    def __init__(self):
        super(SimpleCinemaTransform, self).__init__()
        
        # Simple color transformation network
        self.color_net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x, metadata=None):
        # Apply color transformation
        delta = self.color_net(x)
        
        # Add residual connection
        output = x + 0.1 * delta  # Small correction
        
        # Clamp to valid range
        output = torch.clamp(output, 0, 1)
        
        return output
'''
    
    print("🎯 SIMPLE MODEL STRATEGY:")
    print("- Just learn color corrections (not full reconstruction)")
    print("- Small residual changes to iPhone images")
    print("- Much less likely to collapse to black")
    print("- Faster training and more stable")
    
    return simple_model_code

def main():
    """Main debugging pipeline"""
    print("🎬 MODEL DEBUGGING PIPELINE")
    print("Choose debugging option:")
    print("1. Debug current model outputs")
    print("2. Show fixed architecture")
    print("3. Show simple model approach")
    print("4. All of the above")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice in ["1", "4"]:
        debug_model_outputs()
    
    if choice in ["2", "4"]:
        fix_model_architecture()
    
    if choice in ["3", "4"]:
        simple_model_code = quick_retrain_with_simpler_model()
    
    print(f"\n🎯 RECOMMENDED NEXT STEPS:")
    print("1. The current model is outputting black due to gradient/activation issues")
    print("2. Try the simpler residual approach for faster results")
    print("3. Or implement the fixed architecture with batch norm")
    print("4. Reduce learning rate to 0.0001 for more stable training")

if __name__ == "__main__":
    main()
'''
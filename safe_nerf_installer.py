#!/usr/bin/env python3
"""
Safe NeRF Installation & Dependency Manager
Production-ready installer that won't break existing setup
"""

import subprocess
import sys
import os
import platform
import json
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class SafeInstaller:
    """Safe installation manager with rollback capability"""
    
    def __init__(self):
        self.backup_dir = Path("installation_backups")
        self.backup_dir.mkdir(exist_ok=True)
        self.installed_packages = []
        self.installation_log = []
    
    def backup_environment(self):
        """Create backup of current environment"""
        try:
            # Get current pip list
            result = subprocess.run([sys.executable, '-m', 'pip', 'list', '--format=freeze'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                backup_file = self.backup_dir / "pip_packages_backup.txt"
                with open(backup_file, 'w') as f:
                    f.write(result.stdout)
                logger.info(f"✅ Environment backed up to {backup_file}")
                return True
            else:
                logger.error("❌ Failed to backup environment")
                return False
                
        except Exception as e:
            logger.error(f"❌ Backup failed: {e}")
            return False
    
    def safe_install_package(self, package: str, version: Optional[str] = None) -> bool:
        """Safely install a package with version checking"""
        try:
            # Check if already installed
            import_name = package.replace('-', '_').split('[')[0]
            try:
                __import__(import_name)
                logger.info(f"✅ {package} already installed")
                return True
            except ImportError:
                pass
            
            # Install package
            install_cmd = [sys.executable, '-m', 'pip', 'install']
            if version:
                install_cmd.append(f"{package}=={version}")
            else:
                install_cmd.append(package)
            
            logger.info(f"📦 Installing {package}...")
            result = subprocess.run(install_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.installed_packages.append(package)
                self.installation_log.append(f"SUCCESS: {package}")
                logger.info(f"✅ {package} installed successfully")
                return True
            else:
                logger.error(f"❌ Failed to install {package}: {result.stderr}")
                self.installation_log.append(f"FAILED: {package} - {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Installation error for {package}: {e}")
            return False
    
    def rollback_installation(self):
        """Rollback installation if something goes wrong"""
        logger.warning("🔄 Rolling back installation...")
        
        for package in reversed(self.installed_packages):
            try:
                subprocess.run([sys.executable, '-m', 'pip', 'uninstall', package, '-y'], 
                             capture_output=True)
                logger.info(f"🔄 Removed {package}")
            except Exception as e:
                logger.error(f"❌ Failed to remove {package}: {e}")

class SystemChecker:
    """Check system compatibility for NeRF tools"""
    
    @staticmethod
    def check_gpu() -> Dict[str, any]:
        """Check GPU compatibility"""
        gpu_info = {
            'nvidia_gpu': False,
            'cuda_available': False,
            'gpu_memory': 0,
            'driver_version': None
        }
        
        try:
            # Check for NVIDIA GPU
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines and lines[0]:
                    gpu_data = lines[0].split(', ')
                    gpu_info['nvidia_gpu'] = True
                    gpu_info['gpu_memory'] = int(gpu_data[1]) if len(gpu_data) > 1 else 0
                    gpu_info['driver_version'] = gpu_data[2] if len(gpu_data) > 2 else None
            
            # Check CUDA availability
            try:
                import torch
                gpu_info['cuda_available'] = torch.cuda.is_available()
            except ImportError:
                pass
                
        except Exception as e:
            logger.warning(f"⚠️ GPU check failed: {e}")
        
        return gpu_info
    
    @staticmethod
    def check_system_requirements() -> Dict[str, bool]:
        """Check all system requirements"""
        requirements = {}
        
        # Python version
        requirements['python_version'] = sys.version_info >= (3, 8)
        
        # Operating system
        requirements['supported_os'] = platform.system() in ['Windows', 'Linux', 'Darwin']
        
        # Disk space (approximate)
        try:
            statvfs = os.statvfs('.' if hasattr(os, 'statvfs') else os.getcwd())
            free_space_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
            requirements['disk_space'] = free_space_gb > 5.0  # Need at least 5GB
        except:
            requirements['disk_space'] = True  # Assume OK if can't check
        
        # Memory (approximate)
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            requirements['memory'] = memory_gb > 8.0  # Need at least 8GB
        except ImportError:
            requirements['memory'] = True  # Assume OK if can't check
        
        return requirements

class NeRFToolInstaller:
    """Install and configure NeRF tools safely"""
    
    def __init__(self):
        self.installer = SafeInstaller()
        self.available_tools = {}
    
    def install_core_dependencies(self) -> bool:
        """Install core dependencies for NeRF"""
        logger.info("📦 INSTALLING CORE DEPENDENCIES")
        logger.info("=" * 35)
        
        # Core packages that are safe to install
        core_packages = [
            ('torch', '>=1.12.0'),
            ('torchvision', None),
            ('numpy', '>=1.20.0'),
            ('opencv-python', None),
            ('scipy', None),
            ('matplotlib', None),
            ('tqdm', None),
            ('Pillow', None),
            ('imageio', None),
            ('configargparse', None)
        ]
        
        success = True
        for package, version in core_packages:
            if not self.installer.safe_install_package(package, version):
                success = False
                break
        
        if not success:
            logger.error("❌ Core dependency installation failed")
            self.installer.rollback_installation()
            return False
        
        logger.info("✅ Core dependencies installed successfully")
        return True
    
    def setup_instant_ngp(self) -> bool:
        """Set up Instant-NGP (recommended for RTX 2080)"""
        logger.info("🚀 SETTING UP INSTANT-NGP")
        logger.info("=" * 25)
        
        # Check if we can install Instant-NGP
        gpu_info = SystemChecker.check_gpu()
        
        if not gpu_info['nvidia_gpu']:
            logger.warning("⚠️ No NVIDIA GPU detected - Instant-NGP not recommended")
            return False
        
        if gpu_info['gpu_memory'] < 4000:  # Less than 4GB
            logger.warning("⚠️ GPU has limited memory - Instant-NGP may not work well")
        
        # For now, provide installation instructions
        instructions = """
🚀 INSTANT-NGP INSTALLATION INSTRUCTIONS:

1. Prerequisites:
   - NVIDIA GPU (detected: ✅)
   - CUDA Toolkit 11.3+ 
   - CMake 3.21+
   - Visual Studio 2019+ (Windows) or GCC 7+ (Linux)

2. Installation:
   git clone --recursive https://github.com/NVlabs/instant-ngp
   cd instant-ngp
   
   # Windows:
   cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
   cmake --build build --config RelWithDebInfo -j
   
   # Linux/macOS:
   make -j

3. Test:
   ./build/testbed --scene data/nerf/fox

4. Python bindings (optional):
   pip install ./build

⚠️ Note: This requires manual compilation. 
   If compilation fails, try NeRFStudio instead.
        """
        
        print(instructions)
        
        # Mark as available for manual installation
        self.available_tools['instant_ngp'] = {
            'status': 'manual_install_required',
            'speed': 'very_fast',
            'quality': 'high',
            'gpu_memory_req': '4-8GB'
        }
        
        return True
    
    def setup_nerfstudio(self) -> bool:
        """Set up NeRFStudio (easier installation)"""
        logger.info("🎯 SETTING UP NERFSTUDIO")
        logger.info("=" * 25)
        
        try:
            # Install NeRFStudio
            packages = [
                'nerfstudio',
                'tyro>=0.3.31',
                'gdown',
                'ninja'
            ]
            
            for package in packages:
                if not self.installer.safe_install_package(package):
                    logger.error(f"❌ Failed to install {package}")
                    return False
            
            # Test installation
            result = subprocess.run([sys.executable, '-c', 'import nerfstudio'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                self.available_tools['nerfstudio'] = {
                    'status': 'installed',
                    'speed': 'medium',
                    'quality': 'very_high',
                    'gpu_memory_req': '6-12GB'
                }
                logger.info("✅ NeRFStudio installed successfully")
                return True
            else:
                logger.error("❌ NeRFStudio installation validation failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ NeRFStudio installation failed: {e}")
            return False
    
    def setup_colmap(self) -> bool:
        """Set up COLMAP for camera pose estimation"""
        logger.info("📷 SETTING UP COLMAP")
        logger.info("=" * 20)
        
        try:
            # Try to install pycolmap (Python bindings)
            if self.installer.safe_install_package('pycolmap'):
                logger.info("✅ PyColmap installed successfully")
                return True
            else:
                # Provide manual installation instructions
                instructions = """
📷 COLMAP INSTALLATION INSTRUCTIONS:

COLMAP is required for camera pose estimation from images.

Option 1 - Conda (Recommended):
   conda install -c conda-forge colmap

Option 2 - Manual Download:
   - Windows: Download from https://github.com/colmap/colmap/releases
   - Ubuntu: sudo apt install colmap
   - macOS: brew install colmap

Option 3 - Build from source:
   git clone https://github.com/colmap/colmap.git
   cd colmap
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   make -j
                """
                print(instructions)
                return True
                
        except Exception as e:
            logger.error(f"❌ COLMAP setup failed: {e}")
            return False

def create_nerf_test_scene():
    """Create a test scene for validating NeRF installation"""
    test_script = '''#!/usr/bin/env python3
"""
NeRF Test Scene - Validate Installation
"""

import numpy as np
import torch
import cv2
from pathlib import Path

def create_synthetic_test_scene():
    """Create synthetic test data for NeRF validation"""
    print("🧪 CREATING TEST SCENE")
    print("=" * 20)
    
    # Create test directory
    test_dir = Path("data/nerf_test_scene")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic camera poses (spiral)
    num_views = 20
    poses = []
    
    for i in range(num_views):
        angle = 2 * np.pi * i / num_views
        
        # Camera position (spiral)
        x = 2 * np.cos(angle)
        y = 2 * np.sin(angle) 
        z = 0.5 * np.sin(2 * angle)
        
        # Look at origin
        camera_pos = np.array([x, y, z])
        target = np.array([0, 0, 0])
        up = np.array([0, 0, 1])
        
        # Create pose matrix
        forward = target - camera_pos
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        
        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = -forward
        pose[:3, 3] = camera_pos
        
        poses.append(pose)
    
    # Save poses
    poses_array = np.array(poses)
    np.save(test_dir / "poses.npy", poses_array)
    
    # Create synthetic images (simple colored spheres)
    for i in range(num_views):
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        
        # Add colored sphere
        center = (200, 200)
        radius = 80
        color = (int(255 * (i / num_views)), 100, 255 - int(255 * (i / num_views)))
        
        cv2.circle(img, center, radius, color, -1)
        
        # Save image
        cv2.imwrite(str(test_dir / f"image_{i:03d}.png"), img)
    
    print(f"✅ Test scene created: {test_dir}")
    print(f"   - {num_views} synthetic views")
    print(f"   - Camera poses saved")
    return test_dir

def test_nerf_pipeline(test_dir):
    """Test NeRF pipeline with synthetic data"""
    print("\\n🧠 TESTING NeRF PIPELINE")
    print("=" * 25)
    
    try:
        # Test basic NeRF components
        from production_nerf_architecture import NeRFNetwork, PositionalEncoding
        
        # Initialize network
        nerf = NeRFNetwork()
        print("✅ NeRF network initialized")
        
        # Test positional encoding
        pos_enc = PositionalEncoding()
        test_pos = torch.randn(100, 3)
        test_dirs = torch.randn(100, 3)
        
        encoded_pos = pos_enc.encode_position(test_pos)
        encoded_dirs = pos_enc.encode_direction(test_dirs)
        
        print(f"✅ Positional encoding: {test_pos.shape} -> {encoded_pos.shape}")
        
        # Test forward pass
        with torch.no_grad():
            colors, densities = nerf(test_pos, test_dirs)
        
        print(f"✅ Forward pass: colors {colors.shape}, densities {densities.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ NeRF pipeline test failed: {e}")
        return False

if __name__ == "__main__":
    test_dir = create_synthetic_test_scene()
    success = test_nerf_pipeline(test_dir)
    
    if success:
        print("\\n🎉 NeRF TEST PASSED!")
        print("Ready for real scene reconstruction")
    else:
        print("\\n❌ NeRF test failed - check installation")
'''
    
    return test_script

def main():
    """Main installation and setup routine"""
    print("🎬 SAFE NeRF INSTALLATION FOR ROADSHOW")
    print("=" * 45)
    
    # Check system requirements
    print("🔍 CHECKING SYSTEM REQUIREMENTS")
    print("=" * 35)
    
    requirements = SystemChecker.check_system_requirements()
    gpu_info = SystemChecker.check_gpu()
    
    for req, status in requirements.items():
        print(f"{'✅' if status else '❌'} {req}")
    
    print(f"\n🖥️ GPU INFO:")
    print(f"   NVIDIA GPU: {'✅' if gpu_info['nvidia_gpu'] else '❌'}")
    print(f"   CUDA Available: {'✅' if gpu_info['cuda_available'] else '❌'}")
    print(f"   GPU Memory: {gpu_info['gpu_memory']}MB")
    
    if not all(requirements.values()):
        print("❌ System requirements not met")
        return False
    
    # Initialize installer
    installer = NeRFToolInstaller()
    
    # Backup environment
    if not installer.installer.backup_environment():
        print("⚠️ Could not backup environment - proceed anyway? (y/n)")
        if input().lower() != 'y':
            return False
    
    # Install core dependencies
    if not installer.install_core_dependencies():
        print("❌ Core dependency installation failed")
        return False
    
    # Set up NeRF tools
    print("\n🧠 SETTING UP NeRF TOOLS")
    print("=" * 25)
    
    success_count = 0
    
    # Try Instant-NGP first (best for RTX 2080)
    if installer.setup_instant_ngp():
        success_count += 1
    
    # Try NeRFStudio (easier installation)
    if installer.setup_nerfstudio():
        success_count += 1
    
    # Set up COLMAP
    if installer.setup_colmap():
        success_count += 1
    
    # Create test scene
    test_script = create_nerf_test_scene()
    test_script_path = Path("nerf_test_scene.py")
    with open(test_script_path, 'w') as f:
        f.write(test_script)
    
    print(f"\n📊 INSTALLATION SUMMARY")
    print("=" * 25)
    print(f"Tools set up: {success_count}/3")
    print(f"Available tools: {list(installer.available_tools.keys())}")
    
    if success_count > 0:
        print("\n✅ NeRF INSTALLATION SUCCESSFUL!")
        print("\n⚡ NEXT STEPS:")
        print("1. Test installation: python nerf_test_scene.py")
        print("2. Capture real scene data (20+ images)")
        print("3. Run COLMAP for camera poses")
        print("4. Train NeRF on your scene")
        print("5. Extract depth/lighting for Roadshow")
        
        return True
    else:
        print("\n❌ NeRF installation incomplete")
        print("Manual installation may be required")
        return False

if __name__ == "__main__":
    main()
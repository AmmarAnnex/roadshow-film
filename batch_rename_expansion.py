#!/usr/bin/env python3
"""
Batch Rename & Data Expansion Pipeline
Rename new captures and prepare for advanced training
"""

import os
import shutil
from pathlib import Path
import json
from datetime import datetime

def batch_rename_new_captures():
    """Rename new iPhone and Sony captures starting from 031"""
    print("📁 BATCH RENAMING NEW CAPTURES")
    print("=" * 35)
    
    # Define source and destination paths
    source_iphone_dir = Path("data/training_pairs")  # Where new IMG_xxxx.DNG files are
    source_sony_dir = Path("data/training_pairs")    # Where new DSCxxxxx.ARW files are
    
    # Find existing numbered files to determine starting number
    existing_iphone = list(source_iphone_dir.glob("iphone_*.dng"))
    existing_sony = list(source_sony_dir.glob("sony_*.arw"))
    
    if existing_iphone:
        last_iphone_num = max([int(f.stem.split('_')[1]) for f in existing_iphone])
    else:
        last_iphone_num = 0
    
    if existing_sony:
        last_sony_num = max([int(f.stem.split('_')[1]) for f in existing_sony])
    else:
        last_sony_num = 0
    
    start_num = max(last_iphone_num, last_sony_num) + 1
    
    print(f"📊 Starting rename from: {start_num:03d}")
    print(f"📱 Existing iPhone files: {len(existing_iphone)}")
    print(f"📷 Existing Sony files: {len(existing_sony)}")
    
    # Find new iPhone files (IMG_xxxx.DNG format)
    new_iphone_files = list(source_iphone_dir.glob("IMG_*.DNG"))
    new_iphone_files.sort()  # Sort by filename for consistent pairing
    
    # Find new Sony files (DSCxxxxx.ARW format)
    new_sony_files = list(source_sony_dir.glob("DSC*.ARW"))
    new_sony_files.sort()  # Sort by filename for consistent pairing
    
    print(f"\n🔍 Found {len(new_iphone_files)} new iPhone files")
    print(f"🔍 Found {len(new_sony_files)} new Sony files")
    
    if len(new_iphone_files) != len(new_sony_files):
        print("⚠️ WARNING: Mismatch in file counts!")
        print("Please ensure equal number of iPhone and Sony captures")
        return False
    
    renamed_pairs = []
    
    # Rename files
    for i, (iphone_file, sony_file) in enumerate(zip(new_iphone_files, new_sony_files)):
        new_num = start_num + i
        
        # New filenames
        new_iphone_name = f"iphone_{new_num:03d}.dng"
        new_sony_name = f"sony_{new_num:03d}.arw"
        
        new_iphone_path = source_iphone_dir / new_iphone_name
        new_sony_path = source_sony_dir / new_sony_name
        
        # Rename files
        try:
            iphone_file.rename(new_iphone_path)
            sony_file.rename(new_sony_path)
            
            print(f"✅ Renamed pair {new_num:03d}:")
            print(f"   {iphone_file.name} → {new_iphone_name}")
            print(f"   {sony_file.name} → {new_sony_name}")
            
            renamed_pairs.append({
                'number': new_num,
                'iphone_file': str(new_iphone_path),
                'sony_file': str(new_sony_path),
                'original_iphone': iphone_file.name,
                'original_sony': sony_file.name
            })
            
        except Exception as e:
            print(f"❌ Error renaming {iphone_file.name}: {e}")
    
    # Save rename log
    rename_log = {
        'timestamp': datetime.now().isoformat(),
        'start_number': start_num,
        'pairs_renamed': len(renamed_pairs),
        'pairs': renamed_pairs
    }
    
    log_path = Path("data/rename_log.json")
    with open(log_path, 'w') as f:
        json.dump(rename_log, f, indent=2)
    
    print(f"\n✅ Successfully renamed {len(renamed_pairs)} pairs!")
    print(f"📝 Rename log saved to: {log_path}")
    
    return True

def update_metadata_with_new_pairs():
    """Update depth_metadata.json with new pairs"""
    print("\n📊 UPDATING METADATA")
    print("=" * 20)
    
    metadata_path = Path("data/results/simple_depth_analysis/depth_metadata.json")
    
    if not metadata_path.exists():
        print("❌ Metadata file not found. Run depth analysis first.")
        return
    
    # Load existing metadata
    with open(metadata_path, 'r') as f:
        existing_pairs = json.load(f)
    
    # Find all current training pairs
    training_dir = Path("data/training_pairs")
    iphone_files = sorted(list(training_dir.glob("iphone_*.dng")))
    sony_files = sorted(list(training_dir.glob("sony_*.arw")))
    
    print(f"📱 Found {len(iphone_files)} iPhone files")
    print(f"📷 Found {len(sony_files)} Sony files")
    
    if len(iphone_files) != len(sony_files):
        print("⚠️ Mismatch in training pairs!")
        return
    
    # Create new metadata entries
    new_pairs = []
    existing_count = len(existing_pairs)
    
    for iphone_file, sony_file in zip(iphone_files, sony_files):
        # Extract numbers from filenames
        iphone_num = int(iphone_file.stem.split('_')[1])
        sony_num = int(sony_file.stem.split('_')[1])
        
        if iphone_num == sony_num:
            pair_entry = {
                'pair_id': iphone_num,
                'iphone_file': str(iphone_file),
                'sony_file': str(sony_file),
                'captured_date': datetime.now().isoformat(),
                'setup': 'iPhone_2.5x_Sony_50mm_f1.4'
            }
            new_pairs.append(pair_entry)
    
    # Save updated metadata
    with open(metadata_path, 'w') as f:
        json.dump(new_pairs, f, indent=2)
    
    print(f"✅ Updated metadata with {len(new_pairs)} pairs")
    print(f"📈 Total pairs: {len(new_pairs)} (was {existing_count})")

def prepare_advanced_training_structure():
    """Prepare directory structure for advanced features"""
    print("\n🏗️ PREPARING ADVANCED TRAINING STRUCTURE")
    print("=" * 40)
    
    advanced_dirs = [
        "data/blackmagic_12k",
        "data/blackmagic_12k/raw_footage",
        "data/blackmagic_12k/extracted_frames", 
        "data/nerf_training",
        "data/nerf_training/captures",
        "data/nerf_training/reconstructions",
        "data/nerf_training/synthetic_renders",
        "data/bit_depth_expansion",
        "data/bit_depth_expansion/8bit_source",
        "data/bit_depth_expansion/10bit_target",
        "data/colorchecker_validation",
        "data/light_sensor_data",
        "data/pi_ai_camera"
    ]
    
    for dir_path in advanced_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {dir_path}")
    
    # Create configuration files
    create_advanced_config_files()

def create_advanced_config_files():
    """Create configuration files for advanced features"""
    
    # NeRF training config
    nerf_config = {
        "scene_types": ["indoor", "outdoor", "portrait", "landscape"],
        "capture_requirements": {
            "min_images": 50,
            "max_images": 200,
            "camera_poses": "auto_colmap",
            "lighting_conditions": ["natural", "mixed", "artificial"]
        },
        "training_parameters": {
            "instant_ngp": {
                "resolution": 512,
                "iterations": 10000,
                "learning_rate": 0.01
            }
        }
    }
    
    with open("data/nerf_training/config.json", 'w') as f:
        json.dump(nerf_config, f, indent=2)
    
    # Bit-depth expansion config  
    bit_depth_config = {
        "source_format": "8bit_sRGB",
        "target_format": "10bit_Log",
        "expansion_method": "neural_inverse_tone_mapping",
        "training_approach": "synthetic_hdr_pairs",
        "validation_metrics": ["psnr", "ssim", "delta_e", "dynamic_range"]
    }
    
    with open("data/bit_depth_expansion/config.json", 'w') as f:
        json.dump(bit_depth_config, f, indent=2)
    
    # Pi sensor integration config
    pi_config = {
        "hardware": {
            "pi_version": "Pi 5",
            "ai_camera": "Sony IMX500",
            "light_sensors": 3,
            "sensor_types": ["AS7341_spectral", "TSL2591_dynamic", "VEML7700_ambient"]
        },
        "integration_goals": {
            "real_time_depth": True,
            "ambient_light_analysis": True,
            "spectral_color_analysis": True,
            "scene_understanding": True
        }
    }
    
    with open("data/pi_ai_camera/config.json", 'w') as f:
        json.dump(pi_config, f, indent=2)

def create_data_expansion_roadmap():
    """Create roadmap for data expansion"""
    print("\n🗺️ DATA EXPANSION ROADMAP")
    print("=" * 25)
    
    roadmap = {
        "Phase 1: Current Session (26 new pairs)": {
            "status": "ready_to_process",
            "pairs": "031-056",
            "focus": "Indoor subjects, varied lighting",
            "next_steps": ["rename_files", "update_metadata", "train_v1.5"]
        },
        "Phase 2: Park Session (25 more pairs)": {
            "status": "planned",
            "pairs": "057-081", 
            "focus": "Outdoor, natural lighting, portraits",
            "equipment": "iPhone 2.5x + Sony 50mm f/1.4"
        },
        "Phase 3: Blackmagic Integration": {
            "status": "hardware_ready",
            "goal": "Video frame extraction training",
            "benefit": "100+ frames per scene vs 1 still",
            "workflow": "parallel_capture → frame_extraction → training"
        },
        "Phase 4: NeRF Enhancement": {
            "status": "research_validated",
            "goal": "3D scene understanding for color science",
            "method": "Instant-NGP on RTX 2080",
            "application": "synthetic_training_data + lighting_analysis"
        },
        "Phase 5: Bit-Depth Expansion": {
            "status": "conceptual",
            "goal": "8-bit → 10-bit log-like expansion", 
            "revolutionary": "Mobile filmmaking game-changer",
            "technical": "Neural inverse tone mapping"
        }
    }
    
    for phase, details in roadmap.items():
        print(f"\n🚀 {phase}:")
        for key, value in details.items():
            if isinstance(value, list):
                print(f"   {key}: {', '.join(value)}")
            else:
                print(f"   {key}: {value}")
    
    # Save roadmap
    with open("data/expansion_roadmap.json", 'w') as f:
        json.dump(roadmap, f, indent=2)

def main():
    """Main execution pipeline"""
    print("🎬 ROADSHOW DATA EXPANSION PIPELINE")
    print("=" * 40)
    
    print("Phase 1: Rename new captures")
    if batch_rename_new_captures():
        print("\nPhase 2: Update metadata")
        update_metadata_with_new_pairs()
        
        print("\nPhase 3: Prepare advanced structure")
        prepare_advanced_training_structure()
        
        print("\nPhase 4: Create roadmap")
        create_data_expansion_roadmap()
        
        print(f"\n🎯 NEXT IMMEDIATE STEPS:")
        print("1. Train v1.5 with expanded dataset (56 pairs)")
        print("2. Implement ColorChecker validation")
        print("3. Plan Blackmagic parallel capture workflow")
        print("4. Set up Instant-NGP for NeRF experiments")
        print("5. Design bit-depth expansion architecture")
        
        print(f"\n✅ Ready for advanced cinematic AI development!")
    else:
        print("❌ Rename failed. Check file locations and try again.")

if __name__ == "__main__":
    main()
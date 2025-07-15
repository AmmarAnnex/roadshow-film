#!/usr/bin/env python3
"""
Training Data Expansion Script
Add 23 new iPhone + Sony matched pairs to your training dataset
"""

import os
import shutil
from pathlib import Path
import json
from datetime import datetime
import re

def find_and_organize_new_pairs():
    """Find and organize the new iPhone and Sony pairs"""
    
    print("📁 TRAINING DATA EXPANSION")
    print("=" * 40)
    
    # Setup directories
    project_root = Path.cwd()
    training_dir = project_root / "data" / "training_pairs"
    training_dir.mkdir(parents=True, exist_ok=True)
    
    # Look for new files in common locations
    possible_locations = [
        project_root,  # Current directory
        project_root / "new_photos",
        project_root / "data",
        Path.home() / "Desktop",
        Path.home() / "Downloads",
        Path.home() / "Pictures"
    ]
    
    print("🔍 Searching for new iPhone and Sony files...")
    
    # Find new iPhone DNG files
    new_iphone_files = []
    new_sony_files = []
    
    for location in possible_locations:
        if location.exists():
            # Look for iPhone files (IMG_*.DNG pattern)
            iphone_files = list(location.glob("IMG_*.DNG")) + list(location.glob("IMG_*.dng"))
            # Look for Sony files (DSC*.ARW pattern)  
            sony_files = list(location.glob("DSC*.ARW")) + list(location.glob("DSC*.arw"))
            
            if iphone_files:
                print(f"📱 Found {len(iphone_files)} iPhone files in {location}")
                new_iphone_files.extend(iphone_files)
            
            if sony_files:
                print(f"📷 Found {len(sony_files)} Sony files in {location}")
                new_sony_files.extend(sony_files)
    
    # Remove duplicates and sort
    new_iphone_files = sorted(list(set(new_iphone_files)))
    new_sony_files = sorted(list(set(new_sony_files)))
    
    print(f"\n📊 DISCOVERY SUMMARY:")
    print(f"📱 iPhone DNG files: {len(new_iphone_files)}")
    print(f"📷 Sony ARW files: {len(new_sony_files)}")
    
    return new_iphone_files, new_sony_files

def get_existing_pair_count():
    """Count existing training pairs"""
    
    training_dir = Path("data/training_pairs")
    
    if not training_dir.exists():
        return 0
    
    existing_iphone = list(training_dir.glob("iphone_*.dng"))
    existing_sony = list(training_dir.glob("sony_*.arw"))
    
    iphone_count = len(existing_iphone)
    sony_count = len(existing_sony)
    
    print(f"📊 EXISTING PAIRS:")
    print(f"   iPhone files: {iphone_count}")
    print(f"   Sony files: {sony_count}")
    
    if iphone_count != sony_count:
        print(f"⚠️ Warning: Mismatch in existing pairs!")
        return min(iphone_count, sony_count)
    
    return iphone_count

def smart_pair_matching(iphone_files, sony_files):
    """Smart matching of iPhone and Sony files based on timestamps/sequence"""
    
    print("\n🧠 SMART PAIR MATCHING")
    print("=" * 25)
    
    # Extract numbers from filenames for sorting
    def extract_number(filename):
        numbers = re.findall(r'\d+', filename.stem)
        return int(numbers[-1]) if numbers else 0
    
    # Sort by extracted numbers
    iphone_sorted = sorted(iphone_files, key=extract_number)
    sony_sorted = sorted(sony_files, key=extract_number)
    
    print(f"📱 iPhone range: {extract_number(iphone_sorted[0])} to {extract_number(iphone_sorted[-1])}")
    print(f"📷 Sony range: {extract_number(sony_sorted[0])} to {extract_number(sony_sorted[-1])}")
    
    # Take the minimum count to ensure pairs
    min_count = min(len(iphone_sorted), len(sony_sorted))
    
    if len(iphone_sorted) != len(sony_sorted):
        print(f"⚠️ File count mismatch - using first {min_count} pairs")
    
    matched_pairs = list(zip(iphone_sorted[:min_count], sony_sorted[:min_count]))
    
    print(f"✅ Created {len(matched_pairs)} matched pairs")
    
    return matched_pairs

def copy_and_rename_pairs(matched_pairs, start_number):
    """Copy and rename the matched pairs to training directory"""
    
    print(f"\n📁 COPYING AND RENAMING PAIRS")
    print("=" * 35)
    
    training_dir = Path("data/training_pairs")
    training_dir.mkdir(parents=True, exist_ok=True)
    
    copied_pairs = []
    successful_copies = 0
    
    for i, (iphone_file, sony_file) in enumerate(matched_pairs):
        pair_number = start_number + i
        
        # New filenames
        new_iphone_name = f"iphone_{pair_number:03d}.dng"
        new_sony_name = f"sony_{pair_number:03d}.arw"
        
        iphone_dest = training_dir / new_iphone_name
        sony_dest = training_dir / new_sony_name
        
        try:
            # Copy files
            shutil.copy2(iphone_file, iphone_dest)
            shutil.copy2(sony_file, sony_dest)
            
            # Record successful copy
            pair_info = {
                'pair_id': pair_number,
                'iphone_file': str(iphone_dest),
                'sony_file': str(sony_dest),
                'original_iphone': str(iphone_file),
                'original_sony': str(sony_file),
                'timestamp': datetime.now().isoformat()
            }
            
            copied_pairs.append(pair_info)
            successful_copies += 1
            
            print(f"✅ Pair {pair_number:03d}: {iphone_file.name} + {sony_file.name}")
            
        except Exception as e:
            print(f"❌ Error copying pair {pair_number:03d}: {e}")
            continue
    
    print(f"\n📊 COPY RESULTS:")
    print(f"✅ Successfully copied: {successful_copies} pairs")
    print(f"❌ Failed copies: {len(matched_pairs) - successful_copies}")
    
    return copied_pairs

def update_metadata(new_pairs):
    """Update the training metadata with new pairs"""
    
    print(f"\n📋 UPDATING METADATA")
    print("=" * 20)
    
    # Metadata file location
    metadata_file = Path("data/results/simple_depth_analysis/depth_metadata.json")
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing metadata
    existing_metadata = []
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                existing_metadata = json.load(f)
            print(f"📖 Loaded {len(existing_metadata)} existing metadata entries")
        except Exception as e:
            print(f"⚠️ Error loading existing metadata: {e}")
            existing_metadata = []
    
    # Combine old and new metadata
    all_metadata = existing_metadata + new_pairs
    
    # Save updated metadata
    try:
        with open(metadata_file, 'w') as f:
            json.dump(all_metadata, f, indent=2)
        
        print(f"✅ Updated metadata saved")
        print(f"📊 Total pairs in metadata: {len(all_metadata)}")
        
    except Exception as e:
        print(f"❌ Error saving metadata: {e}")
        return False
    
    return True

def validate_training_setup():
    """Validate the complete training setup"""
    
    print(f"\n🔍 VALIDATION")
    print("=" * 15)
    
    training_dir = Path("data/training_pairs")
    
    # Count files
    iphone_files = list(training_dir.glob("iphone_*.dng"))
    sony_files = list(training_dir.glob("sony_*.arw"))
    
    print(f"📁 Training directory: {training_dir}")
    print(f"📱 iPhone DNG files: {len(iphone_files)}")
    print(f"📷 Sony ARW files: {len(sony_files)}")
    
    # Check for gaps in numbering
    iphone_numbers = sorted([int(f.stem.split('_')[1]) for f in iphone_files])
    sony_numbers = sorted([int(f.stem.split('_')[1]) for f in sony_files])
    
    print(f"📊 iPhone range: {min(iphone_numbers)} to {max(iphone_numbers)}")
    print(f"📊 Sony range: {min(sony_numbers)} to {max(sony_numbers)}")
    
    # Check for missing pairs
    missing_pairs = []
    for i in range(min(iphone_numbers), max(iphone_numbers) + 1):
        if i not in iphone_numbers or i not in sony_numbers:
            missing_pairs.append(i)
    
    if missing_pairs:
        print(f"⚠️ Missing pairs: {missing_pairs}")
    else:
        print(f"✅ All pairs complete - no gaps found")
    
    # File size check
    avg_iphone_size = sum(f.stat().st_size for f in iphone_files) / len(iphone_files) / (1024*1024)
    avg_sony_size = sum(f.stat().st_size for f in sony_files) / len(sony_files) / (1024*1024)
    
    print(f"💾 Average iPhone file size: {avg_iphone_size:.1f} MB")
    print(f"💾 Average Sony file size: {avg_sony_size:.1f} MB")
    
    return len(iphone_files) == len(sony_files) and len(missing_pairs) == 0

def main():
    """Main expansion workflow"""
    
    print("🎬 ROADSHOW TRAINING DATA EXPANSION")
    print("=" * 45)
    print("Adding 23 new iPhone + Sony matched pairs")
    print("Current dataset: 56 pairs → Target: 79 pairs")
    print()
    
    # Step 1: Count existing pairs
    existing_count = get_existing_pair_count()
    start_number = existing_count + 1
    
    print(f"🔢 Starting new pairs from number: {start_number:03d} (continuing from 056)")
    print(f"🎯 Target: 79 total pairs (056 + 23 = 079)")
    print()
    
    # Step 2: Find new files
    iphone_files, sony_files = find_and_organize_new_pairs()
    
    if not iphone_files or not sony_files:
        print("❌ No new files found!")
        print("\n💡 Make sure your new iPhone DNG and Sony ARW files are in:")
        print("   - Current directory")
        print("   - data/ folder") 
        print("   - Desktop")
        print("   - Downloads")
        return False
    
    # Step 3: Smart pair matching
    matched_pairs = smart_pair_matching(iphone_files, sony_files)
    
    if len(matched_pairs) < 20:
        print(f"⚠️ Warning: Only found {len(matched_pairs)} pairs (expected ~23)")
        proceed = input("Continue anyway? (y/n): ").lower().strip()
        if proceed != 'y':
            return False
    
    # Step 4: Copy and rename files
    new_pairs = copy_and_rename_pairs(matched_pairs, start_number)
    
    if not new_pairs:
        print("❌ No pairs were successfully copied!")
        return False
    
    # Step 5: Update metadata
    if not update_metadata(new_pairs):
        print("❌ Failed to update metadata!")
        return False
    
    # Step 6: Validate setup
    if not validate_training_setup():
        print("⚠️ Validation found issues - please check manually")
    
    # Step 7: Success summary
    total_pairs = existing_count + len(new_pairs)
    
    print(f"\n🎉 SUCCESS!")
    print("=" * 15)
    print(f"✅ Added {len(new_pairs)} new training pairs")
    print(f"📊 Total training pairs: {total_pairs} (MASSIVE DATASET!)")
    print(f"🆔 Pair range: 001 to {total_pairs:03d}")
    print(f"📈 Dataset growth: 30 → 56 → {total_pairs} pairs")
    print(f"📁 All files in: data/training_pairs/")
    print(f"📋 Metadata updated: data/results/simple_depth_analysis/depth_metadata.json")
    
    print(f"\n⚡ NEXT STEPS:")
    print("1. Train v1.5 model with expanded dataset:")
    print("   python clean_training_pipeline.py")
    print("2. Run scientific analysis:")
    print("   python fixed_scientific_analysis.py")
    print("3. Compare with v1.4 results!")
    
    return True

if __name__ == "__main__":
    main()

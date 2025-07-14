#!/usr/bin/env python3
"""
LUT Verification Tool
Checks all LUT files in the luts directory and verifies they can be loaded
"""

import base64
import numpy as np
from pathlib import Path

def parse_cube_lut(file_path):
    """Parse a .cube LUT file"""
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
            return None, "No LUT_3D_SIZE found"
        
        lut_data = []
        for line in lines[data_start:]:
            if line.strip():
                values = [float(x) for x in line.split()]
                if len(values) == 3:
                    lut_data.append(values)
        
        expected_entries = lut_size ** 3
        if len(lut_data) != expected_entries:
            return None, f"Expected {expected_entries} entries, found {len(lut_data)}"
        
        lut_3d = np.array(lut_data).reshape((lut_size, lut_size, lut_size, 3))
        return lut_3d.astype(np.float32), "OK"
        
    except Exception as e:
        return None, str(e)

def decode_base64_lut(file_path):
    """Decode base64 LUT file"""
    try:
        with open(file_path, 'r') as f:
            b64_content = f.read().strip()
        
        if not b64_content:
            return None, "Empty file"
        
        decoded_bytes = base64.b64decode(b64_content)
        decoded_text = decoded_bytes.decode('utf-8')
        
        # Save to temp file and parse
        temp_path = Path(file_path).with_suffix('.cube')
        with open(temp_path, 'w') as f:
            f.write(decoded_text)
        
        lut, status = parse_cube_lut(temp_path)
        temp_path.unlink()  # Clean up
        
        return lut, status
        
    except Exception as e:
        return None, str(e)

def verify_all_luts():
    """Verify all LUT files in the luts directory"""
    luts_dir = Path("luts")
    
    if not luts_dir.exists():
        print("❌ luts/ directory not found!")
        print("Create the directory and place your LUT files there.")
        return
    
    print("🎨 LUT VERIFICATION REPORT")
    print("=" * 50)
    
    # Find all LUT files
    cube_files = list(luts_dir.glob("*.cube"))
    b64_files = list(luts_dir.glob("*.b64.txt"))
    
    all_files = cube_files + b64_files
    
    if not all_files:
        print("❌ No LUT files found in luts/ directory!")
        print("Expected file types: .cube, .b64.txt")
        return
    
    print(f"📁 Found {len(all_files)} LUT files:")
    print(f"   - {len(cube_files)} .cube files")
    print(f"   - {len(b64_files)} .b64.txt files")
    
    print(f"\n🔍 VERIFICATION RESULTS:")
    print("-" * 50)
    
    successful = 0
    failed = 0
    
    for lut_file in sorted(all_files):
        print(f"\n📄 {lut_file.name}")
        print(f"   Size: {lut_file.stat().st_size:,} bytes")
        
        if lut_file.suffix == '.cube':
            lut, status = parse_cube_lut(lut_file)
        elif lut_file.suffix == '.txt' and '.b64' in lut_file.name:
            lut, status = decode_base64_lut(lut_file)
        else:
            lut, status = None, "Unknown file type"
        
        if lut is not None:
            print(f"   ✅ {status}")
            print(f"   📐 Shape: {lut.shape}")
            print(f"   📊 Range: {lut.min():.3f} to {lut.max():.3f}")
            
            # Basic validation
            if lut.min() < 0 or lut.max() > 1:
                print(f"   ⚠️  Warning: Values outside [0,1] range")
            
            successful += 1
        else:
            print(f"   ❌ {status}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"📊 SUMMARY:")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📈 Success rate: {100*successful/(successful+failed):.1f}%")
    
    if successful > 0:
        print(f"\n🎯 RECOMMENDATIONS:")
        if failed == 0:
            print("   Perfect! All LUTs loaded successfully.")
            print("   The cinema model will use these LUTs for training.")
        else:
            print("   Some LUTs failed to load. Check the error messages above.")
            print("   The model will use only the successfully loaded LUTs.")
        
        print(f"\n💡 USAGE:")
        print("   Primary LUT selection priority:")
        print("   1. Neutral A7s3_65x.cube")
        print("   2. First successfully loaded LUT")
        
        # Show what the model will actually use
        preference_order = [
            "Neutral A7s3_65x.cube",
            "Neutral A7s3_65x.b64.txt",
            "P6k Neutral Gen5.cube",
            "Eterna A7s3_65x_Legacy.cube"
        ]
        
        available_preferred = []
        for pref in preference_order:
            if (luts_dir / pref).exists():
                lut_test, status_test = None, ""
                if pref.endswith('.cube'):
                    lut_test, status_test = parse_cube_lut(luts_dir / pref)
                elif pref.endswith('.b64.txt'):
                    lut_test, status_test = decode_base64_lut(luts_dir / pref)
                
                if lut_test is not None:
                    available_preferred.append(pref)
        
        if available_preferred:
            print(f"   🎯 Model will likely use: {available_preferred[0]}")
    else:
        print(f"\n❌ No LUTs loaded successfully!")
        print("   The model will train without LUT support.")
        print("   Check your LUT files and try again.")

def main():
    verify_all_luts()

if __name__ == "__main__":
    main()
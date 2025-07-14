#!/usr/bin/env python3
"""
Decode base64 LUT files to standard .cube format
"""

import base64
from pathlib import Path

def decode_lut_file(input_path, output_path=None):
    """Decode a base64 LUT file to .cube format"""
    input_path = Path(input_path)
    
    if output_path is None:
        # Remove .b64.txt and add .cube
        output_path = input_path.parent / input_path.name.replace('.b64.txt', '.cube')
    
    try:
        # Read base64 content
        with open(input_path, 'r') as f:
            b64_content = f.read().strip()
        
        # Decode base64
        decoded_bytes = base64.b64decode(b64_content)
        decoded_text = decoded_bytes.decode('utf-8')
        
        # Write decoded content
        with open(output_path, 'w') as f:
            f.write(decoded_text)
        
        print(f"✅ Decoded: {input_path.name} → {output_path.name}")
        
        # Print first few lines to verify
        lines = decoded_text.split('\n')[:10]
        print("   First few lines:")
        for line in lines:
            if line.strip():
                print(f"   {line}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error decoding {input_path.name}: {e}")
        return False

def main():
    """Decode all LUT files in the luts folder"""
    print("🔄 DECODING BASE64 LUT FILES")
    print("=" * 50)
    
    # Create luts folder if it doesn't exist
    luts_folder = Path("luts")
    luts_folder.mkdir(exist_ok=True)
    
    # Find all .b64.txt files
    b64_files = list(Path(".").glob("*.b64.txt"))
    b64_files.extend(list(luts_folder.glob("*.b64.txt")))
    
    if not b64_files:
        print("❌ No .
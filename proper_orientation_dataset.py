
class ProperOrientationDataset(Dataset):
    """Dataset with proper EXIF orientation handling"""
    
    def process_image_with_exif(self, file_path: str, size: int = 256):
        """Process image with proper EXIF orientation"""
        # Get EXIF orientation
        exif_orientation = get_exif_orientation(file_path)
        
        # Process RAW with no auto-rotation
        with rawpy.imread(file_path) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                output_bps=16,
                no_auto_bright=True,
                user_flip=0,  # No rawpy rotation
                use_auto_wb=False,
                no_auto_scale=True
            )
        
        # Apply EXIF rotation manually
        if exif_orientation != 1:
            rgb = apply_exif_rotation(rgb, exif_orientation)
        
        # Normalize and resize
        rgb_norm = rgb.astype(np.float32) / 65535.0
        rgb_resized = cv2.resize(rgb_norm, (size, size))
        
        return np.transpose(rgb_resized, (2, 0, 1))

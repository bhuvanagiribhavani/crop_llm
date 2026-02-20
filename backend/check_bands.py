"""
================================================================================
CHECK SENTINEL-2 DATASET BANDS
================================================================================
This script checks whether the Sentinel-2 LULC dataset contains:
- True multispectral bands (including NIR for NDVI calculation)
- Or only RGB images (3 channels)

Usage:
    python check_bands.py
================================================================================
"""

import os
import sys
from pathlib import Path


def check_image_bands(image_path: str) -> dict:
    """
    Check the number of bands/channels in an image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        dict with image info (bands, height, width, format)
    """
    result = {
        'path': image_path,
        'filename': os.path.basename(image_path),
        'extension': os.path.splitext(image_path)[1].lower(),
        'bands': None,
        'height': None,
        'width': None,
        'format': None,
        'error': None
    }
    
    ext = result['extension']
    
    # ===== TIFF files - use rasterio =====
    if ext in ['.tif', '.tiff']:
        try:
            import rasterio
            result['format'] = 'GeoTIFF'
            
            with rasterio.open(image_path) as src:
                result['bands'] = src.count
                result['height'] = src.height
                result['width'] = src.width
                
                # Get band descriptions if available
                result['band_descriptions'] = src.descriptions
                result['dtype'] = str(src.dtypes[0])
                result['crs'] = str(src.crs) if src.crs else None
                
        except ImportError:
            result['error'] = "rasterio not installed. Install with: pip install rasterio"
        except Exception as e:
            result['error'] = f"Error reading TIFF: {str(e)}"
    
    # ===== PNG/JPG files - use OpenCV or PIL =====
    elif ext in ['.png', '.jpg', '.jpeg']:
        result['format'] = 'Standard Image (PNG/JPG)'
        
        try:
            import cv2
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            
            if img is None:
                result['error'] = "Failed to load image with OpenCV"
            else:
                if len(img.shape) == 2:
                    # Grayscale
                    result['bands'] = 1
                    result['height'], result['width'] = img.shape
                else:
                    # Color image
                    result['height'], result['width'], result['bands'] = img.shape
                result['dtype'] = str(img.dtype)
                
        except ImportError:
            # Fallback to PIL
            try:
                from PIL import Image
                img = Image.open(image_path)
                result['width'], result['height'] = img.size
                result['bands'] = len(img.getbands())
                result['dtype'] = img.mode
            except ImportError:
                result['error'] = "Neither OpenCV nor PIL installed"
            except Exception as e:
                result['error'] = f"Error reading image: {str(e)}"
    
    else:
        result['error'] = f"Unsupported file format: {ext}"
    
    return result


def main():
    """Main function to check dataset bands."""
    
    print("=" * 70)
    print("SENTINEL-2 LULC DATASET BAND CHECKER")
    print("=" * 70)
    
    # ===== Define dataset paths =====
    dataset_root = Path("SEN-2 LULC")
    
    # Directories to check
    image_dirs = [
        dataset_root / "train_images",
        dataset_root / "val_images", 
        dataset_root / "test_images"
    ]
    
    # ===== Find a sample image =====
    print("\n" + "-" * 50)
    print("SEARCHING FOR SAMPLE IMAGE...")
    print("-" * 50)
    
    sample_image = None
    
    for img_dir in image_dirs:
        if not img_dir.exists():
            print(f"  ✗ Directory not found: {img_dir}")
            continue
            
        print(f"  ✓ Found directory: {img_dir}")
        
        # Look for image files (check subdirectories too)
        for ext in ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg']:
            # Check root of directory
            files = list(img_dir.glob(ext))
            if files:
                sample_image = files[0]
                break
            
            # Check subdirectories (e.g., train_images/train/)
            files = list(img_dir.glob(f"**/{ext}"))
            if files:
                sample_image = files[0]
                break
        
        if sample_image:
            break
    
    if sample_image is None:
        print("\n❌ ERROR: No image files found in the dataset!")
        print("   Please check that the dataset exists at: SEN-2 LULC/")
        sys.exit(1)
    
    # ===== Analyze the sample image =====
    print("\n" + "-" * 50)
    print("ANALYZING SAMPLE IMAGE...")
    print("-" * 50)
    
    print(f"\n📁 File Path: {sample_image}")
    print(f"📄 Filename:  {sample_image.name}")
    print(f"📋 Extension: {sample_image.suffix}")
    
    # Check the image
    result = check_image_bands(str(sample_image))
    
    if result['error']:
        print(f"\n❌ ERROR: {result['error']}")
        sys.exit(1)
    
    # ===== Print results =====
    print("\n" + "-" * 50)
    print("IMAGE PROPERTIES")
    print("-" * 50)
    
    print(f"\n🖼️  Format:     {result['format']}")
    print(f"📐 Dimensions: {result['width']} x {result['height']} pixels")
    print(f"🎨 Bands:      {result['bands']}")
    print(f"📊 Data Type:  {result['dtype']}")
    
    if result.get('crs'):
        print(f"🌍 CRS:        {result['crs']}")
    
    if result.get('band_descriptions') and any(result['band_descriptions']):
        print(f"📝 Band Names: {result['band_descriptions']}")
    
    # ===== Count files in dataset =====
    print("\n" + "-" * 50)
    print("DATASET FILE COUNT")
    print("-" * 50)
    
    for img_dir in image_dirs:
        if img_dir.exists():
            # Count all image files including subdirectories
            count = len(list(img_dir.glob("**/*.png"))) + \
                    len(list(img_dir.glob("**/*.jpg"))) + \
                    len(list(img_dir.glob("**/*.tif"))) + \
                    len(list(img_dir.glob("**/*.tiff")))
            print(f"  {img_dir.name}: {count:,} images")
    
    # ===== Final Conclusion =====
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    if result['bands'] and result['bands'] > 3:
        print(f"""
✅ This dataset contains MULTISPECTRAL Sentinel-2 bands!

   Number of bands: {result['bands']}
   
   Typical Sentinel-2 band configuration:
   - Band 1:  Coastal aerosol (443 nm)
   - Band 2:  Blue (490 nm)
   - Band 3:  Green (560 nm)
   - Band 4:  Red (665 nm)
   - Band 5:  Red Edge 1 (705 nm)
   - Band 6:  Red Edge 2 (740 nm)
   - Band 7:  Red Edge 3 (783 nm)
   - Band 8:  NIR (842 nm) ← Required for NDVI
   - Band 8A: Narrow NIR (865 nm)
   - Band 9:  Water Vapor (945 nm)
   - Band 10: Cirrus (1375 nm)
   - Band 11: SWIR 1 (1610 nm)
   - Band 12: SWIR 2 (2190 nm)
   
   🌿 TRUE NDVI CALCULATION IS POSSIBLE!
   
   NDVI = (NIR - RED) / (NIR + RED)
   
   You can use the NIR band (typically Band 8) for accurate 
   vegetation health analysis.
""")
    
    elif result['bands'] == 3:
        print(f"""
⚠️  This dataset contains only RGB images (3 channels)

   Number of channels: {result['bands']} (Red, Green, Blue)
   
   The Near-Infrared (NIR) band required for NDVI is NOT available.
   
   🚫 TRUE NDVI CALCULATION IS NOT POSSIBLE
   
   Alternative options for vegetation analysis:
   
   1. VGI (Vegetation Greenness Index):
      VGI = G / (R + G + B)
      Uses green channel proportion as vegetation proxy
   
   2. ExG (Excess Green Index):
      ExG = 2*G - R - B
      Highlights green vegetation
   
   3. VARI (Visible Atmospherically Resistant Index):
      VARI = (G - R) / (G + R - B)
      Better atmospheric correction
   
   These RGB-based indices provide approximate vegetation health 
   estimates but are less accurate than true NDVI.
""")
    
    elif result['bands'] == 1:
        print(f"""
ℹ️  This dataset contains GRAYSCALE images (1 channel)

   Number of channels: {result['bands']}
   
   These may be:
   - Pre-processed masks
   - Single-band satellite products
   - Label/annotation images
   
   🚫 NDVI CALCULATION IS NOT POSSIBLE with single-band images.
""")
    
    else:
        print(f"""
❓ Unexpected number of bands: {result['bands']}

   Please verify the dataset format manually.
""")
    
    print("=" * 70)
    
    return result


if __name__ == "__main__":
    result = main()

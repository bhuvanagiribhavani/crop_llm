"""
================================================================================
DETAILED SENTINEL-2 DATASET BAND CHECK
================================================================================
This script thoroughly checks multiple sample files to verify if NIR bands
are present in the dataset.

Usage:
    python check_bands_detailed.py
================================================================================
"""

import os
import sys
from pathlib import Path
from collections import defaultdict


def check_image_bands(image_path: str) -> dict:
    """Check image bands/channels."""
    result = {
        'path': str(image_path),
        'filename': os.path.basename(image_path),
        'extension': os.path.splitext(image_path)[1].lower(),
        'bands': None,
        'height': None,
        'width': None,
        'format': None,
        'error': None
    }
    
    ext = result['extension']
    
    # ===== TIFF files =====
    if ext in ['.tif', '.tiff']:
        try:
            import rasterio
            result['format'] = 'GeoTIFF'
            
            with rasterio.open(image_path) as src:
                result['bands'] = src.count
                result['height'] = src.height
                result['width'] = src.width
                result['band_descriptions'] = list(src.descriptions)
                result['dtype'] = str(src.dtypes[0])
                
        except ImportError:
            result['error'] = "rasterio not installed"
        except Exception as e:
            result['error'] = str(e)
    
    # ===== PNG/JPG files =====
    elif ext in ['.png', '.jpg', '.jpeg']:
        result['format'] = 'PNG/JPG'
        
        try:
            import cv2
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            
            if img is None:
                result['error'] = "Failed to load"
            else:
                if len(img.shape) == 2:
                    result['bands'] = 1
                    result['height'], result['width'] = img.shape
                else:
                    result['height'], result['width'], result['bands'] = img.shape
                result['dtype'] = str(img.dtype)
                
        except ImportError:
            try:
                from PIL import Image
                img = Image.open(image_path)
                result['width'], result['height'] = img.size
                result['bands'] = len(img.getbands())
                result['mode'] = img.mode
            except Exception as e:
                result['error'] = str(e)
        except Exception as e:
            result['error'] = str(e)
    
    else:
        result['error'] = f"Unsupported format: {ext}"
    
    return result


def main():
    """Main function."""
    
    print("=" * 75)
    print("DETAILED SENTINEL-2 LULC DATASET BAND CHECK")
    print("=" * 75)
    
    dataset_root = Path("SEN-2 LULC")
    
    if not dataset_root.exists():
        print(f"\n❌ ERROR: Dataset not found at {dataset_root}")
        sys.exit(1)
    
    # ===== Check directory structure =====
    print("\n" + "-" * 75)
    print("DATASET DIRECTORY STRUCTURE")
    print("-" * 75)
    
    for item in sorted(dataset_root.iterdir()):
        if item.is_dir():
            file_count = len(list(item.glob("**/*")))
            print(f"\n📁 {item.name}/ ({file_count} items)")
            
            # Check first level subdirectories
            for subitem in sorted(item.iterdir())[:5]:
                if subitem.is_dir():
                    img_count = len([f for f in subitem.glob("*") if f.is_file()])
                    print(f"   ├── {subitem.name}/ ({img_count} files)")
    
    # ===== Find and check sample images =====
    print("\n" + "-" * 75)
    print("CHECKING SAMPLE FILES FROM EACH DIRECTORY")
    print("-" * 75)
    
    image_dirs = [
        ("Train Images", dataset_root / "train_images"),
        ("Val Images", dataset_root / "val_images"),
        ("Test Images", dataset_root / "test_images"),
    ]
    
    results_by_dir = {}
    band_stats = defaultdict(int)
    format_stats = defaultdict(int)
    
    for dir_label, img_dir in image_dirs:
        if not img_dir.exists():
            print(f"\n❌ {dir_label}: Directory not found")
            continue
        
        print(f"\n{dir_label}:")
        
        # Find samples
        samples = []
        
        # Check subdirectories first (e.g., train_images/train/)
        for subdir in img_dir.iterdir():
            if subdir.is_dir():
                for img_file in list(subdir.glob("*"))[:3]:
                    if img_file.is_file() and img_file.suffix.lower() in ['.png', '.jpg', '.tiff', '.tif']:
                        samples.append(img_file)
        
        # If no subdirectories, check root
        if not samples:
            for img_file in list(img_dir.glob("*"))[:3]:
                if img_file.is_file() and img_file.suffix.lower() in ['.png', '.jpg', '.tiff', '.tif']:
                    samples.append(img_file)
        
        if not samples:
            print("   ⚠️  No image files found")
            continue
        
        for sample in samples[:3]:  # Check first 3 files
            result = check_image_bands(str(sample))
            
            status = "✓" if not result['error'] else "✗"
            print(f"\n   {status} {result['filename']}")
            print(f"      Extension: {result['extension']}")
            print(f"      Format: {result['format']}")
            
            if result['bands']:
                print(f"      Bands/Channels: {result['bands']}")
                print(f"      Dimensions: {result['width']} x {result['height']}")
                print(f"      Data Type: {result['dtype']}")
                
                if result.get('band_descriptions') and any(result['band_descriptions']):
                    print(f"      Band Descriptions: {result['band_descriptions']}")
                
                band_stats[result['bands']] += 1
                format_stats[result['format']] += 1
            
            if result['error']:
                print(f"      ❌ Error: {result['error']}")
        
        results_by_dir[dir_label] = samples
    
    # ===== Summary Statistics =====
    print("\n" + "=" * 75)
    print("SUMMARY STATISTICS")
    print("=" * 75)
    
    print("\nFile Formats Found:")
    for fmt, count in format_stats.items():
        print(f"  - {fmt}: {count} files checked")
    
    print("\nBand Configuration Found:")
    for bands, count in sorted(band_stats.items()):
        if bands == 3:
            print(f"  - {bands} channels (RGB ONLY): {count} files")
        elif bands > 3:
            print(f"  - {bands} channels (MULTISPECTRAL - including possible NIR!): {count} files")
        elif bands == 1:
            print(f"  - {bands} channel (GRAYSCALE): {count} files")
    
    # ===== Final Conclusion =====
    print("\n" + "=" * 75)
    print("CONCLUSION")
    print("=" * 75)
    
    if any(b > 3 for b in band_stats.keys()):
        print("""
✅ MULTISPECTRAL BANDS DETECTED!

Your dataset appears to contain multispectral imagery with more than 3 channels.
This likely includes NIR (Near-Infrared) band for NDVI calculation.

🌿 TRUE NDVI CALCULATION IS POSSIBLE!

Next steps:
1. Identify which band contains NIR data
2. Update your preprocessing to use NIR band
3. Calculate: NDVI = (NIR - RED) / (NIR + RED)
""")
    elif all(b == 3 for b in band_stats.keys() if b is not None):
        print("""
⚠️  ONLY RGB IMAGES DETECTED

Your dataset contains 3-channel RGB images only.
The NIR band required for true NDVI is NOT available.

🚫 TRUE NDVI NOT POSSIBLE, but alternatives exist:

✓ Using VGI (Vegetation Greenness Index) - ALREADY IMPLEMENTED
  VGI = G / (R + G + B)
  
✓ Other RGB-based indices:
  - ExG = 2*G - R - B
  - VARI = (G - R) / (G + R - B)

Your current implementation with adaptive VGI thresholds is appropriate
for this dataset.
""")
    else:
        print("""
❓ MIXED OR UNCLEAR BAND CONFIGURATION

The dataset may contain files with different band configurations.
Please verify manually or contact the dataset provider.
""")
    
    print("=" * 75)


if __name__ == "__main__":
    main()

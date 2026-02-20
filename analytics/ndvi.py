"""
================================================================================
CROP HEALTH ANALYSIS USING VEGETATION INDICES
================================================================================
This module provides functions for computing vegetation indices from 
Sentinel-2 satellite imagery for crop health assessment.

Supported Indices:
- NDVI (Normalized Difference Vegetation Index)
- NDWI (Normalized Difference Water Index)
- EVI (Enhanced Vegetation Index)

Sentinel-2 Band Reference:
- Band 2: Blue (490 nm)
- Band 3: Green (560 nm)
- Band 4: Red (665 nm)
- Band 8: NIR (842 nm)

Author: Deep Learning Project
Date: 2026
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import warnings

# Suppress division warnings (we handle them explicitly)
warnings.filterwarnings('ignore', category=RuntimeWarning)


# ============================================================================
# VEGETATION INDEX COMPUTATIONS
# ============================================================================

def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """
    Compute Normalized Difference Vegetation Index (NDVI).
    
    NDVI is the most widely used vegetation index for assessing plant health,
    vegetation density, and biomass. It exploits the difference in reflectance
    between the NIR and Red bands.
    
    Formula:
        NDVI = (NIR - RED) / (NIR + RED)
    
    Value Range: [-1, 1]
        - High values (0.6-1.0): Dense, healthy vegetation
        - Medium values (0.2-0.6): Sparse vegetation, crops
        - Low values (-0.1-0.2): Bare soil, rocks
        - Negative values: Water, clouds, snow
    
    Args:
        nir (np.ndarray): Near-Infrared band (Sentinel-2 Band 8)
        red (np.ndarray): Red band (Sentinel-2 Band 4)
    
    Returns:
        np.ndarray: NDVI values in range [-1, 1]
    
    References:
        Rouse, J.W., et al. (1974). "Monitoring vegetation systems in the 
        Great Plains with ERTS." NASA Special Publication, 351, 309-317.
    """
    # Convert to float for precision
    nir = nir.astype(np.float64)
    red = red.astype(np.float64)
    
    # Compute numerator and denominator
    numerator = nir - red
    denominator = nir + red
    
    # Handle division by zero safely
    # Where denominator is zero, set NDVI to 0 (undefined)
    ndvi = np.divide(
        numerator, 
        denominator, 
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0
    )
    
    # Clip to valid range [-1, 1]
    ndvi = np.clip(ndvi, -1.0, 1.0)
    
    return ndvi


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Compute Normalized Difference Water Index (NDWI).
    
    NDWI is used to monitor changes in water content of leaves and
    to delineate water bodies. It's sensitive to liquid water in vegetation.
    
    Formula:
        NDWI = (GREEN - NIR) / (GREEN + NIR)
    
    Value Range: [-1, 1]
        - Positive values: Water bodies, wet surfaces
        - Near zero: Bare soil, urban areas
        - Negative values: Vegetation (dry)
    
    Args:
        green (np.ndarray): Green band (Sentinel-2 Band 3)
        nir (np.ndarray): Near-Infrared band (Sentinel-2 Band 8)
    
    Returns:
        np.ndarray: NDWI values in range [-1, 1]
    
    References:
        McFeeters, S.K. (1996). "The use of the Normalized Difference Water 
        Index (NDWI) in the delineation of open water features."
        International Journal of Remote Sensing, 17(7), 1425-1432.
    """
    # Convert to float for precision
    green = green.astype(np.float64)
    nir = nir.astype(np.float64)
    
    # Compute numerator and denominator
    numerator = green - nir
    denominator = green + nir
    
    # Handle division by zero safely
    ndwi = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0
    )
    
    # Clip to valid range [-1, 1]
    ndwi = np.clip(ndwi, -1.0, 1.0)
    
    return ndwi


def compute_evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray,
                G: float = 2.5, C1: float = 6.0, C2: float = 7.5, 
                L: float = 1.0) -> np.ndarray:
    """
    Compute Enhanced Vegetation Index (EVI).
    
    EVI is an optimized vegetation index designed to enhance the vegetation
    signal with improved sensitivity in high biomass regions. It reduces
    atmospheric influences and soil background noise.
    
    Formula:
        EVI = G * (NIR - RED) / (NIR + C1*RED - C2*BLUE + L)
    
    Default coefficients (for MODIS/Sentinel-2):
        G = 2.5 (gain factor)
        C1 = 6.0 (aerosol resistance coefficient for red)
        C2 = 7.5 (aerosol resistance coefficient for blue)
        L = 1.0 (canopy background adjustment)
    
    Value Range: [-1, 1] (typically 0 to 1 for vegetation)
        - High values (>0.5): Dense, healthy vegetation
        - Medium values (0.2-0.5): Moderate vegetation
        - Low values (<0.2): Sparse or stressed vegetation
    
    Args:
        nir (np.ndarray): Near-Infrared band (Sentinel-2 Band 8)
        red (np.ndarray): Red band (Sentinel-2 Band 4)
        blue (np.ndarray): Blue band (Sentinel-2 Band 2)
        G (float): Gain factor (default: 2.5)
        C1 (float): Coefficient for red band (default: 6.0)
        C2 (float): Coefficient for blue band (default: 7.5)
        L (float): Canopy background adjustment (default: 1.0)
    
    Returns:
        np.ndarray: EVI values (typically in range [-1, 1])
    
    References:
        Huete, A., et al. (2002). "Overview of the radiometric and biophysical 
        performance of the MODIS vegetation indices." Remote Sensing of 
        Environment, 83(1-2), 195-213.
    """
    # Convert to float for precision
    nir = nir.astype(np.float64)
    red = red.astype(np.float64)
    blue = blue.astype(np.float64)
    
    # Compute numerator and denominator
    numerator = G * (nir - red)
    denominator = nir + C1 * red - C2 * blue + L
    
    # Handle division by zero safely
    evi = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0
    )
    
    # Clip to reasonable range [-1, 1]
    evi = np.clip(evi, -1.0, 1.0)
    
    return evi


# ============================================================================
# CROP HEALTH CLASSIFICATION
# ============================================================================

def classify_crop_health(ndvi_map: np.ndarray) -> np.ndarray:
    """
    Classify crop health based on NDVI values.
    
    Classification Scheme:
        - Healthy (2):   NDVI > 0.6  - Dense, vigorous vegetation
        - Moderate (1):  0.3 <= NDVI <= 0.6 - Normal crop condition
        - Poor (0):      NDVI < 0.3 - Stressed, sparse, or bare
    
    Args:
        ndvi_map (np.ndarray): NDVI values array
    
    Returns:
        np.ndarray: Classification map with values:
            0 = Poor health / Bare soil
            1 = Moderate health
            2 = Healthy / Dense vegetation
    """
    # Initialize classification map
    health_map = np.zeros_like(ndvi_map, dtype=np.uint8)
    
    # Classify based on thresholds
    health_map[ndvi_map < 0.3] = 0       # Poor health
    health_map[(ndvi_map >= 0.3) & (ndvi_map <= 0.6)] = 1  # Moderate
    health_map[ndvi_map > 0.6] = 2       # Healthy
    
    return health_map


def get_health_statistics(health_map: np.ndarray) -> dict:
    """
    Compute statistics from crop health classification map.
    
    Args:
        health_map (np.ndarray): Classification map from classify_crop_health()
    
    Returns:
        dict: Statistics including pixel counts and percentages for each class
    """
    total_pixels = health_map.size
    
    poor_count = np.sum(health_map == 0)
    moderate_count = np.sum(health_map == 1)
    healthy_count = np.sum(health_map == 2)
    
    stats = {
        'total_pixels': total_pixels,
        'poor': {
            'count': int(poor_count),
            'percentage': (poor_count / total_pixels) * 100
        },
        'moderate': {
            'count': int(moderate_count),
            'percentage': (moderate_count / total_pixels) * 100
        },
        'healthy': {
            'count': int(healthy_count),
            'percentage': (healthy_count / total_pixels) * 100
        }
    }
    
    return stats


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_ndvi_colormap():
    """
    Create a custom colormap for NDVI visualization.
    
    Color scheme:
        - Blue: Water (negative NDVI)
        - Brown/Tan: Bare soil (low NDVI)
        - Light green: Sparse vegetation
        - Dark green: Dense vegetation
    """
    colors = [
        '#0000FF',  # -1.0: Blue (water)
        '#87CEEB',  # -0.5: Light blue
        '#D2B48C',  # 0.0: Tan (bare soil)
        '#C4A569',  # 0.2: Light brown
        '#90EE90',  # 0.4: Light green
        '#32CD32',  # 0.6: Lime green
        '#228B22',  # 0.8: Forest green
        '#006400',  # 1.0: Dark green
    ]
    
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'ndvi_custom', colors, N=256
    )
    return cmap


def visualize_ndvi(ndvi_map: np.ndarray, save_path: str = None,
                   title: str = "NDVI Map", show_colorbar: bool = True):
    """
    Visualize NDVI map with custom colormap.
    
    Args:
        ndvi_map (np.ndarray): NDVI values array
        save_path (str): Path to save the figure (optional)
        title (str): Plot title
        show_colorbar (bool): Whether to show colorbar
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use custom NDVI colormap
    cmap = create_ndvi_colormap()
    
    # Plot NDVI
    im = ax.imshow(ndvi_map, cmap=cmap, vmin=-1, vmax=1)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('NDVI Value', fontsize=12)
        cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ NDVI visualization saved to: {save_path}")
    
    plt.close()
    return fig


def visualize_health_map(health_map: np.ndarray, save_path: str = None,
                         title: str = "Crop Health Classification"):
    """
    Visualize crop health classification map.
    
    Args:
        health_map (np.ndarray): Health classification map
        save_path (str): Path to save the figure (optional)
        title (str): Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define colors for each health class
    colors = ['#FF4444', '#FFAA00', '#00AA00']  # Red, Orange, Green
    cmap = mcolors.ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    # Plot health map
    im = ax.imshow(health_map, cmap=cmap, norm=norm)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add colorbar with labels
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['Poor', 'Moderate', 'Healthy'])
    cbar.set_label('Crop Health', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Health map saved to: {save_path}")
    
    plt.close()
    return fig


def visualize_all_indices(nir: np.ndarray, red: np.ndarray, 
                          green: np.ndarray, blue: np.ndarray,
                          save_path: str = None):
    """
    Compute and visualize all vegetation indices in a single figure.
    
    Args:
        nir, red, green, blue: Sentinel-2 band arrays
        save_path (str): Path to save the figure (optional)
    """
    # Compute indices
    ndvi = compute_ndvi(nir, red)
    ndwi = compute_ndwi(green, nir)
    evi = compute_evi(nir, red, blue)
    health = classify_crop_health(ndvi)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # NDVI
    cmap_ndvi = create_ndvi_colormap()
    im1 = axes[0, 0].imshow(ndvi, cmap=cmap_ndvi, vmin=-1, vmax=1)
    axes[0, 0].set_title('NDVI (Vegetation Index)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
    
    # NDWI
    im2 = axes[0, 1].imshow(ndwi, cmap='RdYlBu', vmin=-1, vmax=1)
    axes[0, 1].set_title('NDWI (Water Index)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
    
    # EVI
    im3 = axes[1, 0].imshow(evi, cmap='YlGn', vmin=-0.5, vmax=1)
    axes[1, 0].set_title('EVI (Enhanced Vegetation Index)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
    
    # Health Classification
    colors = ['#FF4444', '#FFAA00', '#00AA00']
    cmap_health = mcolors.ListedColormap(colors)
    im4 = axes[1, 1].imshow(health, cmap=cmap_health, vmin=0, vmax=2)
    axes[1, 1].set_title('Crop Health Classification', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    cbar4 = plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, ticks=[0, 1, 2])
    cbar4.ax.set_yticklabels(['Poor', 'Moderate', 'Healthy'])
    
    plt.suptitle('Sentinel-2 Vegetation Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ All indices visualization saved to: {save_path}")
    
    plt.close()
    return fig


# ============================================================================
# DEMO / MAIN FUNCTION
# ============================================================================

def generate_synthetic_bands(size: tuple = (256, 256), seed: int = 42):
    """
    Generate synthetic Sentinel-2 bands for demonstration.
    
    Creates realistic-looking bands with vegetation, water, and bare soil patterns.
    
    Args:
        size (tuple): Image dimensions (height, width)
        seed (int): Random seed for reproducibility
    
    Returns:
        tuple: (nir, red, green, blue) numpy arrays
    """
    np.random.seed(seed)
    h, w = size
    
    # Create base patterns
    x = np.linspace(0, 4*np.pi, w)
    y = np.linspace(0, 4*np.pi, h)
    X, Y = np.meshgrid(x, y)
    
    # Vegetation pattern (high NIR, low red)
    vegetation = 0.5 + 0.3 * np.sin(X) * np.cos(Y) + 0.2 * np.random.rand(h, w)
    vegetation = np.clip(vegetation, 0, 1)
    
    # Water body (circular region)
    center_x, center_y = w // 3, h // 3
    distance = np.sqrt((np.arange(w) - center_x)**2 + (np.arange(h)[:, None] - center_y)**2)
    water_mask = distance < min(h, w) // 6
    
    # Generate bands
    # NIR: High for vegetation, low for water
    nir = 0.7 * vegetation + 0.2 * np.random.rand(h, w)
    nir[water_mask] = 0.1 + 0.05 * np.random.rand(np.sum(water_mask))
    
    # Red: Low for vegetation, moderate for bare soil
    red = 0.3 * (1 - vegetation) + 0.1 * np.random.rand(h, w)
    red[water_mask] = 0.15 + 0.05 * np.random.rand(np.sum(water_mask))
    
    # Green: Moderate for vegetation, high for water
    green = 0.4 * vegetation + 0.2 * np.random.rand(h, w)
    green[water_mask] = 0.3 + 0.1 * np.random.rand(np.sum(water_mask))
    
    # Blue: Low for vegetation, high for water
    blue = 0.2 * (1 - vegetation) + 0.1 * np.random.rand(h, w)
    blue[water_mask] = 0.4 + 0.1 * np.random.rand(np.sum(water_mask))
    
    # Scale to typical Sentinel-2 reflectance range [0, 10000]
    nir = (nir * 10000).astype(np.float32)
    red = (red * 10000).astype(np.float32)
    green = (green * 10000).astype(np.float32)
    blue = (blue * 10000).astype(np.float32)
    
    return nir, red, green, blue


def load_image_as_bands(image_path: str):
    """
    Load an RGB image and treat channels as pseudo-bands for demonstration.
    
    Note: This is for demo purposes. Real Sentinel-2 analysis requires
    actual multispectral bands (B2, B3, B4, B8).
    
    Args:
        image_path (str): Path to image file
    
    Returns:
        tuple: (nir, red, green, blue) - simulated bands from RGB
    """
    from PIL import Image
    
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img, dtype=np.float32)
    
    # Extract RGB channels
    red = img_array[:, :, 0]
    green = img_array[:, :, 1]
    blue = img_array[:, :, 2]
    
    # Simulate NIR from vegetation response (higher for green vegetation)
    # This is an approximation for demo purposes
    nir = 0.5 * green + 0.3 * red + 0.2 * blue + np.random.rand(*red.shape) * 50
    
    return nir, red, green, blue


def main():
    """
    Main demo function for crop health analysis.
    
    Demonstrates:
    1. Generating/loading band data
    2. Computing vegetation indices
    3. Classifying crop health
    4. Saving visualizations
    """
    print("=" * 70)
    print("CROP HEALTH ANALYSIS - VEGETATION INDICES DEMO")
    print("=" * 70)
    
    # Output directory
    output_dir = Path("outputs/health")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to load a sample image from test set, else use synthetic data
    test_images_dir = Path("SEN-2 LULC/test_images")
    
    if test_images_dir.exists():
        image_files = list(test_images_dir.glob("*.png")) + list(test_images_dir.glob("*.jpg"))
        if image_files:
            print(f"\n✓ Loading sample image: {image_files[0].name}")
            nir, red, green, blue = load_image_as_bands(str(image_files[0]))
            print(f"  Image shape: {red.shape}")
        else:
            print("\n⚠ No images found, generating synthetic bands...")
            nir, red, green, blue = generate_synthetic_bands(size=(256, 256))
    else:
        print("\n⚠ Test images directory not found, generating synthetic bands...")
        nir, red, green, blue = generate_synthetic_bands(size=(256, 256))
    
    print(f"  Band shapes: {nir.shape}")
    
    # ==================== COMPUTE INDICES ====================
    print("\n" + "-" * 50)
    print("COMPUTING VEGETATION INDICES...")
    print("-" * 50)
    
    # Compute NDVI
    ndvi = compute_ndvi(nir, red)
    print(f"\n✓ NDVI computed:")
    print(f"  - Min: {ndvi.min():.4f}")
    print(f"  - Max: {ndvi.max():.4f}")
    print(f"  - Mean: {ndvi.mean():.4f}")
    
    # Compute NDWI
    ndwi = compute_ndwi(green, nir)
    print(f"\n✓ NDWI computed:")
    print(f"  - Min: {ndwi.min():.4f}")
    print(f"  - Max: {ndwi.max():.4f}")
    print(f"  - Mean: {ndwi.mean():.4f}")
    
    # Compute EVI
    evi = compute_evi(nir, red, blue)
    print(f"\n✓ EVI computed:")
    print(f"  - Min: {evi.min():.4f}")
    print(f"  - Max: {evi.max():.4f}")
    print(f"  - Mean: {evi.mean():.4f}")
    
    # ==================== CLASSIFY HEALTH ====================
    print("\n" + "-" * 50)
    print("CLASSIFYING CROP HEALTH...")
    print("-" * 50)
    
    health_map = classify_crop_health(ndvi)
    stats = get_health_statistics(health_map)
    
    print(f"\n✓ Health Classification Results:")
    print(f"  - Poor (NDVI < 0.3):       {stats['poor']['percentage']:6.2f}% ({stats['poor']['count']:,} pixels)")
    print(f"  - Moderate (0.3-0.6):      {stats['moderate']['percentage']:6.2f}% ({stats['moderate']['count']:,} pixels)")
    print(f"  - Healthy (NDVI > 0.6):    {stats['healthy']['percentage']:6.2f}% ({stats['healthy']['count']:,} pixels)")
    
    # ==================== SAVE VISUALIZATIONS ====================
    print("\n" + "-" * 50)
    print("SAVING VISUALIZATIONS...")
    print("-" * 50)
    
    # Save NDVI map
    visualize_ndvi(
        ndvi, 
        save_path=str(output_dir / "ndvi.png"),
        title="NDVI - Normalized Difference Vegetation Index"
    )
    
    # Save health map
    visualize_health_map(
        health_map,
        save_path=str(output_dir / "health_classification.png"),
        title="Crop Health Classification (based on NDVI)"
    )
    
    # Save all indices visualization
    visualize_all_indices(
        nir, red, green, blue,
        save_path=str(output_dir / "all_indices.png")
    )
    
    # ==================== SUMMARY ====================
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\nOutput files saved to: {output_dir.absolute()}")
    print("  - ndvi.png: NDVI visualization")
    print("  - health_classification.png: Crop health map")
    print("  - all_indices.png: All vegetation indices comparison")
    
    return ndvi, ndwi, evi, health_map, stats


if __name__ == "__main__":
    ndvi, ndwi, evi, health_map, stats = main()

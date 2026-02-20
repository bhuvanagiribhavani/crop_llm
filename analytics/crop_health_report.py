"""
================================================================================
CROP HEALTH ANALYTICS REPORT GENERATOR
================================================================================
This script performs crop health analysis on predicted segmentation masks
from the U-Net Sentinel-2 LULC classification model.

IMPORTANT NOTE ON VEGETATION INDEX:
-----------------------------------
True NDVI (Normalized Difference Vegetation Index) requires the NIR (Near-Infrared)
band, which is not available in standard RGB images:

    NDVI = (NIR - RED) / (NIR + RED)

Since our Sentinel-2 LULC dataset provides RGB patches only (3 channels),
we use a Vegetation Greenness Index (VGI) as a proxy for vegetation health:

    VGI = G / (R + G + B + ε)

where:
    - G = Green channel value
    - R = Red channel value  
    - B = Blue channel value
    - ε = 1e-6 (small constant to avoid division by zero)

VGI leverages the fact that healthy vegetation reflects more green light
relative to red and blue, making it a reasonable proxy for vegetation vigor
when multispectral bands are unavailable.

Analysis includes:
- VGI computation from RGB channels
- Per-class vegetation health assessment
- Health classification (Healthy/Moderate/Poor)
- Statistical summary and visualizations

Author: Deep Learning Project
Date: 2026
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from PIL import Image
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

# Class definitions for Sentinel-2 LULC
CLASS_NAMES = {
    0: "Water",
    1: "Trees",
    2: "Grass",
    3: "Flooded Vegetation",
    4: "Crops",
    5: "Scrub/Shrub",
    6: "Built Area",
    7: "Bare Ground"
}

# Crop-related classes for analysis
CROP_CLASSES = [4, 5, 6]  # Crops, Scrub/Shrub, Built Area

# Vegetation classes (for NDVI analysis)
VEGETATION_CLASSES = [1, 2, 3, 4, 5]  # Trees, Grass, Flooded Veg, Crops, Scrub

# Sentinel-2 pixel resolution (10m for RGB bands)
PIXEL_RESOLUTION_M = 10  # meters per pixel

# Adaptive threshold configuration
# Thresholds are computed dynamically based on vegetation pixel statistics:
#   Healthy:   VGI > mean + 0.5 * std
#   Moderate:  mean - 0.5 * std <= VGI <= mean + 0.5 * std  
#   Poor:      VGI < mean - 0.5 * std
ADAPTIVE_THRESHOLD_FACTOR = 0.5  # multiplier for standard deviation

# Global variables to store computed adaptive thresholds (set in main())
VGI_HEALTHY_THRESHOLD = None    # Will be set to: mean + 0.5 * std
VGI_MODERATE_THRESHOLD = None   # Will be set to: mean - 0.5 * std


# ============================================================================
# VGI COMPUTATION (RGB-BASED VEGETATION PROXY)
# ============================================================================

def compute_vgi_from_rgb(image: np.ndarray) -> np.ndarray:
    """
    Compute Vegetation Greenness Index (VGI) from RGB image.
    
    IMPORTANT: True NDVI requires NIR band which is not available in RGB images.
    Since our Sentinel-2 LULC dataset provides RGB patches only (3 channels),
    we use VGI as a proxy for vegetation health estimation.
    
    Formula:
        VGI = G / (R + G + B + ε)
    
    where ε = 1e-6 prevents division by zero.
    
    Rationale:
    - Healthy vegetation reflects more green light (chlorophyll)
    - Higher green proportion relative to total brightness indicates vegetation
    - VGI ranges from 0 to ~0.5+ for very green vegetation
    
    Args:
        image (np.ndarray): RGB image array (H, W, 3) in range [0, 255]
    
    Returns:
        np.ndarray: VGI values normalized to range [0, 1]
    """
    # Convert to float for precision
    image = image.astype(np.float64)
    
    # Extract RGB channels
    if len(image.shape) == 3 and image.shape[2] >= 3:
        red = image[:, :, 0]
        green = image[:, :, 1]
        blue = image[:, :, 2]
    else:
        # Grayscale image - cannot compute meaningful VGI
        print("⚠ Warning: Image is not RGB, returning zeros")
        return np.zeros(image.shape[:2], dtype=np.float64)
    
    # Compute VGI: proportion of green in total RGB
    # VGI = G / (R + G + B + epsilon)
    epsilon = 1e-6  # Small constant to avoid division by zero
    total = red + green + blue + epsilon
    
    vgi = green / total
    
    # Normalize to [0, 1] range
    # Theoretical max is ~0.33 for pure green, but we scale for better visualization
    vgi_normalized = np.clip(vgi * 2.0, 0.0, 1.0)  # Scale and clip
    
    return vgi_normalized


def compute_adaptive_thresholds(vgi: np.ndarray, mask: np.ndarray, 
                                  vegetation_classes: list = None) -> tuple:
    """
    Compute adaptive health thresholds based on VGI statistics over vegetation pixels.
    
    This approach provides balanced health classifications by using the actual
    distribution of VGI values in the image rather than fixed thresholds.
    
    Thresholds:
    - Healthy:   VGI > mean + 0.5 * std
    - Moderate:  mean - 0.5 * std <= VGI <= mean + 0.5 * std
    - Poor:      VGI < mean - 0.5 * std
    
    Args:
        vgi (np.ndarray): VGI values array
        mask (np.ndarray): Segmentation mask with class labels
        vegetation_classes (list): List of vegetation class IDs to consider
    
    Returns:
        tuple: (healthy_threshold, moderate_threshold, vgi_mean, vgi_std)
    """
    if vegetation_classes is None:
        vegetation_classes = VEGETATION_CLASSES
    
    # Create mask for vegetation pixels
    veg_mask = np.isin(mask, vegetation_classes)
    
    if np.sum(veg_mask) == 0:
        # No vegetation pixels - use all pixels
        print("⚠ Warning: No vegetation pixels found, using all pixels for thresholds")
        veg_vgi = vgi.flatten()
    else:
        veg_vgi = vgi[veg_mask]
    
    # Compute statistics
    vgi_mean = np.mean(veg_vgi)
    vgi_std = np.std(veg_vgi)
    
    # Compute adaptive thresholds
    healthy_threshold = vgi_mean + ADAPTIVE_THRESHOLD_FACTOR * vgi_std
    moderate_threshold = vgi_mean - ADAPTIVE_THRESHOLD_FACTOR * vgi_std
    
    # Ensure thresholds are within valid range [0, 1]
    healthy_threshold = np.clip(healthy_threshold, 0.0, 1.0)
    moderate_threshold = np.clip(moderate_threshold, 0.0, 1.0)
    
    return healthy_threshold, moderate_threshold, vgi_mean, vgi_std


def classify_health(vgi_value: float, healthy_thresh: float = None, 
                    moderate_thresh: float = None) -> str:
    """
    Classify vegetation health based on VGI value using adaptive thresholds.
    
    Classification scheme (adaptive, based on image statistics):
    - Healthy:  VGI > mean + 0.5*std (above average vegetation vigor)
    - Moderate: mean - 0.5*std <= VGI <= mean + 0.5*std (average condition)
    - Poor:     VGI < mean - 0.5*std (below average, stressed vegetation)
    
    Args:
        vgi_value (float): Mean VGI value
        healthy_thresh (float): Threshold for healthy classification
        moderate_thresh (float): Threshold for poor/moderate boundary
    
    Returns:
        str: Health category
    """
    # Use global thresholds if not provided
    if healthy_thresh is None:
        healthy_thresh = VGI_HEALTHY_THRESHOLD
    if moderate_thresh is None:
        moderate_thresh = VGI_MODERATE_THRESHOLD
    
    # Handle case where thresholds not yet computed
    if healthy_thresh is None or moderate_thresh is None:
        return "Unknown"
    
    if vgi_value > healthy_thresh:
        return "Healthy"
    elif vgi_value >= moderate_thresh:
        return "Moderate"
    else:
        return "Poor"


def get_health_color(category: str) -> str:
    """Get color for health category."""
    colors = {
        "Healthy": "#2ECC71",    # Green
        "Moderate": "#F39C12",   # Orange
        "Poor": "#E74C3C"        # Red
    }
    return colors.get(category, "#95A5A6")


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_class_health(mask: np.ndarray, vgi: np.ndarray, 
                         class_id: int, healthy_thresh: float = None,
                         moderate_thresh: float = None) -> dict:
    """
    Analyze vegetation health for a specific class using VGI with adaptive thresholds.
    
    Args:
        mask (np.ndarray): Segmentation mask with class labels
        vgi (np.ndarray): VGI (Vegetation Greenness Index) values array
        class_id (int): Class ID to analyze
        healthy_thresh (float): Adaptive threshold for healthy classification
        moderate_thresh (float): Adaptive threshold for poor/moderate boundary
    
    Returns:
        dict: Statistics for the class
    """
    # Get pixels belonging to this class
    class_mask = mask == class_id
    pixel_count = np.sum(class_mask)
    
    if pixel_count == 0:
        return {
            'class_id': class_id,
            'class_name': CLASS_NAMES.get(class_id, f"Class {class_id}"),
            'pixel_count': 0,
            'area_hectares': 0.0,
            'mean_vgi': np.nan,
            'std_vgi': np.nan,
            'min_vgi': np.nan,
            'max_vgi': np.nan,
            'health_category': "N/A"
        }
    
    # Extract VGI values for this class
    class_vgi = vgi[class_mask]
    
    # Compute statistics
    mean_vgi = np.mean(class_vgi)
    std_vgi = np.std(class_vgi)
    min_vgi = np.min(class_vgi)
    max_vgi = np.max(class_vgi)
    
    # Calculate area in hectares
    # Area = pixel_count * (resolution_m)^2 / 10000 (m² to hectares)
    area_m2 = pixel_count * (PIXEL_RESOLUTION_M ** 2)
    area_hectares = area_m2 / 10000
    
    # Classify health based on VGI using adaptive thresholds
    health_category = classify_health(mean_vgi, healthy_thresh, moderate_thresh)
    
    return {
        'class_id': class_id,
        'class_name': CLASS_NAMES.get(class_id, f"Class {class_id}"),
        'pixel_count': int(pixel_count),
        'area_hectares': round(area_hectares, 4),
        'mean_vgi': round(mean_vgi, 4),
        'std_vgi': round(std_vgi, 4),
        'min_vgi': round(min_vgi, 4),
        'max_vgi': round(max_vgi, 4),
        'health_category': health_category
    }


def generate_health_report(mask: np.ndarray, vgi: np.ndarray,
                           classes_to_analyze: list = None,
                           healthy_thresh: float = None,
                           moderate_thresh: float = None) -> pd.DataFrame:
    """
    Generate comprehensive health report for all specified classes using VGI
    with adaptive thresholds.
    
    Args:
        mask (np.ndarray): Segmentation mask
        vgi (np.ndarray): VGI (Vegetation Greenness Index) values
        classes_to_analyze (list): List of class IDs to analyze
        healthy_thresh (float): Adaptive threshold for healthy classification
        moderate_thresh (float): Adaptive threshold for poor/moderate boundary
    
    Returns:
        pd.DataFrame: Health report dataframe
    """
    if classes_to_analyze is None:
        classes_to_analyze = list(CLASS_NAMES.keys())
    
    results = []
    for class_id in classes_to_analyze:
        stats = analyze_class_health(mask, vgi, class_id, healthy_thresh, moderate_thresh)
        results.append(stats)
    
    df = pd.DataFrame(results)
    
    # Reorder columns
    columns = ['class_id', 'class_name', 'pixel_count', 'area_hectares',
               'mean_vgi', 'std_vgi', 'min_vgi', 'max_vgi', 'health_category']
    df = df[columns]
    
    return df


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_vgi_colormap():
    """Create custom colormap for VGI visualization (low to high vegetation)."""
    colors = [
        '#8B0000',  # Dark red (0.0 - no vegetation)
        '#FF4500',  # Orange red
        '#FFA500',  # Orange
        '#FFD700',  # Gold
        '#ADFF2F',  # Green yellow
        '#32CD32',  # Lime green
        '#228B22',  # Forest green
        '#006400',  # Dark green (1.0 - dense vegetation)
    ]
    return mcolors.LinearSegmentedColormap.from_list('vgi', colors, N=256)


def plot_vgi_bar_chart(df: pd.DataFrame, save_path: str,
                        healthy_thresh: float = None, moderate_thresh: float = None):
    """
    Create bar chart of mean VGI per class with adaptive threshold lines.
    
    Args:
        df (pd.DataFrame): Health report dataframe
        save_path (str): Path to save the figure
        healthy_thresh (float): Adaptive healthy threshold
        moderate_thresh (float): Adaptive moderate/poor threshold
    """
    # Use global thresholds if not provided
    if healthy_thresh is None:
        healthy_thresh = VGI_HEALTHY_THRESHOLD
    if moderate_thresh is None:
        moderate_thresh = VGI_MODERATE_THRESHOLD
    
    # Filter out classes with no pixels
    df_valid = df[df['pixel_count'] > 0].copy()
    
    if len(df_valid) == 0:
        print("⚠ No valid classes to plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get colors based on health category
    colors = [get_health_color(cat) for cat in df_valid['health_category']]
    
    # Create bar chart
    bars = ax.bar(df_valid['class_name'], df_valid['mean_vgi'], 
                  color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar, val, std in zip(bars, df_valid['mean_vgi'], df_valid['std_vgi']):
        height = bar.get_height()
        ax.annotate(f'{val:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add adaptive threshold lines
    if healthy_thresh is not None:
        ax.axhline(y=healthy_thresh, color='green', linestyle='--', 
                   linewidth=2, label=f'Healthy (>{healthy_thresh:.3f})')
    if moderate_thresh is not None:
        ax.axhline(y=moderate_thresh, color='orange', linestyle='--', 
                   linewidth=2, label=f'Moderate/Poor ({moderate_thresh:.3f})')
    
    # Customize plot
    ax.set_xlabel('Land Cover Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean VGI (Vegetation Greenness Index)', fontsize=12, fontweight='bold')
    ax.set_title('Vegetation Health Index by Land Cover Class\n(Adaptive Thresholds: mean ± 0.5*std)', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0.0, 1.0)  # VGI ranges from 0 to 1
    ax.grid(axis='y', alpha=0.3)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Add health category legend with adaptive thresholds
    legend_elements = [
        Patch(facecolor='#2ECC71', edgecolor='black', 
              label=f'Healthy (>{healthy_thresh:.3f})' if healthy_thresh else 'Healthy'),
        Patch(facecolor='#F39C12', edgecolor='black', 
              label=f'Moderate ({moderate_thresh:.3f}-{healthy_thresh:.3f})' if moderate_thresh and healthy_thresh else 'Moderate'),
        Patch(facecolor='#E74C3C', edgecolor='black', 
              label=f'Poor (<{moderate_thresh:.3f})' if moderate_thresh else 'Poor')
    ]
    ax.legend(handles=legend_elements, loc='upper left', title='Health Category (Adaptive)')
    
    plt.tight_layout()
    
    # Save figure
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ VGI bar chart saved to: {save_path}")


def create_vgi_overlay(image: np.ndarray, vgi: np.ndarray, 
                        mask: np.ndarray, save_path: str,
                        alpha: float = 0.5):
    """
    Create VGI heatmap overlay on input image.
    
    Args:
        image (np.ndarray): Input RGB image
        vgi (np.ndarray): VGI values
        mask (np.ndarray): Segmentation mask
        save_path (str): Path to save the figure
        alpha (float): Overlay transparency
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Original Image
    axes[0].imshow(image)
    axes[0].set_title('Input Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Plot 2: VGI Heatmap
    cmap = create_vgi_colormap()
    im = axes[1].imshow(vgi, cmap=cmap, vmin=0.0, vmax=1.0)
    axes[1].set_title('VGI Heatmap\n(Vegetation Greenness Index)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('VGI Value', fontsize=10)
    
    # Plot 3: Overlay
    # Normalize image for display
    img_display = image.astype(np.float32) / 255.0 if image.max() > 1 else image
    
    # Create VGI color overlay
    vgi_normalized = vgi  # VGI is already in [0, 1]
    vgi_colored = cmap(vgi_normalized)[:, :, :3]
    
    # Blend images
    overlay = (1 - alpha) * img_display + alpha * vgi_colored
    overlay = np.clip(overlay, 0, 1)
    
    axes[2].imshow(overlay)
    axes[2].set_title('VGI Overlay on Image', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.suptitle('Crop Health Analysis - VGI Visualization\n(Vegetation Greenness Index for RGB Images)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ VGI overlay saved to: {save_path}")


def create_detailed_report_figure(df: pd.DataFrame, vgi: np.ndarray,
                                  mask: np.ndarray, save_path: str,
                                  healthy_thresh: float = None,
                                  moderate_thresh: float = None):
    """
    Create a detailed visual report combining multiple visualizations.
    
    Args:
        df (pd.DataFrame): Health report dataframe
        vgi (np.ndarray): VGI values
        mask (np.ndarray): Segmentation mask
        save_path (str): Path to save the figure
        healthy_thresh (float): Adaptive healthy threshold
        moderate_thresh (float): Adaptive moderate/poor threshold
    """
    # Use global thresholds if not provided
    if healthy_thresh is None:
        healthy_thresh = VGI_HEALTHY_THRESHOLD
    if moderate_thresh is None:
        moderate_thresh = VGI_MODERATE_THRESHOLD
    
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid layout
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: VGI Distribution (Histogram)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(vgi.flatten(), bins=50, color='#3498DB', edgecolor='black', alpha=0.7)
    if healthy_thresh is not None:
        ax1.axvline(x=healthy_thresh, color='green', linestyle='--', 
                    linewidth=2, label=f'Healthy (>{healthy_thresh:.3f})')
    if moderate_thresh is not None:
        ax1.axvline(x=moderate_thresh, color='orange', linestyle='--', 
                    linewidth=2, label=f'Moderate/Poor ({moderate_thresh:.3f})')
    ax1.set_xlabel('VGI Value', fontsize=11)
    ax1.set_ylabel('Pixel Count', fontsize=11)
    ax1.set_title('VGI Distribution with Adaptive Thresholds', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: Class Area Pie Chart
    ax2 = fig.add_subplot(gs[0, 1])
    df_valid = df[df['pixel_count'] > 0]
    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(df_valid)))
    wedges, texts, autotexts = ax2.pie(
        df_valid['pixel_count'], 
        labels=df_valid['class_name'],
        autopct='%1.1f%%',
        colors=colors_pie,
        explode=[0.02] * len(df_valid)
    )
    ax2.set_title('Land Cover Distribution', fontsize=12, fontweight='bold')
    
    # Plot 3: VGI Heatmap
    ax3 = fig.add_subplot(gs[1, 0])
    cmap = create_vgi_colormap()
    im = ax3.imshow(vgi, cmap=cmap, vmin=0.0, vmax=1.0)
    ax3.set_title('VGI Spatial Distribution', fontsize=12, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    
    # Plot 4: Health Summary Table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Create table data
    table_data = df_valid[['class_name', 'pixel_count', 'area_hectares', 
                           'mean_vgi', 'health_category']].values.tolist()
    columns = ['Class', 'Pixels', 'Area (ha)', 'Mean VGI', 'Health']
    
    table = ax4.table(
        cellText=table_data,
        colLabels=columns,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color cells based on health
    for i, row in enumerate(table_data):
        health = row[-1]
        color = get_health_color(health)
        table[(i+1, 4)].set_facecolor(color)
        table[(i+1, 4)].set_text_props(color='white', fontweight='bold')
    
    ax4.set_title('Health Summary Table', fontsize=12, fontweight='bold', y=0.95)
    
    plt.suptitle('Crop Health Analytics Report\n(Using VGI - Vegetation Greenness Index for RGB Images)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Detailed report figure saved to: {save_path}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main function to generate crop health analytics report.
    
    Note: This script uses VGI (Vegetation Greenness Index) instead of NDVI
    because the Sentinel-2 LULC dataset provides RGB patches only (no NIR band).
    
    VGI Formula: VGI = G / (R + G + B + ε)
    
    Health Thresholds (ADAPTIVE based on vegetation pixel statistics):
    - Healthy:  VGI > mean + 0.5*std
    - Moderate: mean - 0.5*std <= VGI <= mean + 0.5*std
    - Poor:     VGI < mean - 0.5*std
    """
    global VGI_HEALTHY_THRESHOLD, VGI_MODERATE_THRESHOLD
    
    print("=" * 70)
    print("CROP HEALTH ANALYTICS REPORT GENERATOR")
    print("Using VGI (Vegetation Greenness Index) for RGB Images")
    print("With ADAPTIVE Thresholds (mean ± 0.5*std)")
    print("=" * 70)
    
    # ==================== CONFIGURATION ====================
    # File paths
    input_image_path = "outputs/predictions/test_image.png"
    predicted_mask_path = "outputs/predictions/predicted_mask.png"
    
    # Alternative paths to check
    alt_image_paths = [
        "outputs/predictions/test_image.png",
        "SEN-2 LULC/test_images/test/10000.png",
        "SEN-2 LULC/test_images/test/10001.png",
        "SEN-2 LULC/test_images/1.png",
        "SEN-2 LULC/test_images/0.png",
    ]
    
    alt_mask_paths = [
        "outputs/predictions/predicted_mask.png",
        "SEN-2 LULC/test_masks/test/10000.png",
        "SEN-2 LULC/test_masks/test/10001.png",
        "outputs/predictions/predicted_mask_colored.png",
    ]
    
    # Output directory
    output_dir = Path("outputs/health")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ==================== LOAD DATA ====================
    print("\n" + "-" * 50)
    print("LOADING DATA...")
    print("-" * 50)
    
    # Find input image
    image = None
    for img_path in alt_image_paths:
        if os.path.exists(img_path):
            print(f"✓ Loading input image: {img_path}")
            image = np.array(Image.open(img_path).convert('RGB'))
            input_image_path = img_path
            break
    
    if image is None:
        print("⚠ No input image found, generating synthetic data for demo...")
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        # Add some structure
        image[:, :, 1] = np.clip(image[:, :, 1] + 50, 0, 255)  # More green
    
    print(f"  Image shape: {image.shape}")
    
    # Find predicted mask
    mask = None
    for mask_path in alt_mask_paths:
        if os.path.exists(mask_path):
            print(f"✓ Loading predicted mask: {mask_path}")
            mask_img = Image.open(mask_path)
            
            # Handle colored vs grayscale mask
            if mask_img.mode == 'RGB' or mask_img.mode == 'RGBA':
                # Convert colored mask to class labels
                mask_array = np.array(mask_img.convert('RGB'))
                # Use a simple heuristic - take the dominant channel or convert to grayscale
                mask = np.array(mask_img.convert('L'))
                # Normalize to class range
                mask = (mask / 255.0 * 7).astype(np.uint8)
            else:
                mask = np.array(mask_img)
            
            predicted_mask_path = mask_path
            break
    
    if mask is None:
        print("⚠ No predicted mask found, generating synthetic mask for demo...")
        # Create synthetic mask with various classes
        mask = np.random.randint(0, 8, (256, 256), dtype=np.uint8)
        # Add some structure - make crops dominant in center
        mask[64:192, 64:192] = 4  # Crops
        mask[32:64, 32:64] = 1    # Trees
        mask[200:230, 200:230] = 0  # Water
    
    print(f"  Mask shape: {mask.shape}")
    print(f"  Unique classes: {np.unique(mask)}")
    
    # Resize mask to match image if needed
    if mask.shape[:2] != image.shape[:2]:
        print(f"  Resizing mask from {mask.shape} to {image.shape[:2]}")
        mask_img = Image.fromarray(mask)
        mask_img = mask_img.resize((image.shape[1], image.shape[0]), Image.NEAREST)
        mask = np.array(mask_img)
    
    # ==================== COMPUTE VGI ====================
    print("\n" + "-" * 50)
    print("COMPUTING VGI (Vegetation Greenness Index)...")
    print("-" * 50)
    print("Note: Using VGI = G/(R+G+B) as proxy for vegetation health")
    print("      (True NDVI requires NIR band not available in RGB images)")
    
    vgi = compute_vgi_from_rgb(image)
    
    print(f"\n✓ VGI computed (all pixels):")
    print(f"  - Min: {vgi.min():.4f}")
    print(f"  - Max: {vgi.max():.4f}")
    print(f"  - Mean: {vgi.mean():.4f}")
    print(f"  - Std: {vgi.std():.4f}")
    
    # ==================== COMPUTE ADAPTIVE THRESHOLDS ====================
    print("\n" + "-" * 50)
    print("COMPUTING ADAPTIVE THRESHOLDS...")
    print("-" * 50)
    print("Method: Thresholds based on vegetation pixel statistics")
    print(f"  Formula: Healthy > mean + {ADAPTIVE_THRESHOLD_FACTOR}*std")
    print(f"           Moderate: mean - {ADAPTIVE_THRESHOLD_FACTOR}*std to mean + {ADAPTIVE_THRESHOLD_FACTOR}*std")
    print(f"           Poor < mean - {ADAPTIVE_THRESHOLD_FACTOR}*std")
    
    # Compute adaptive thresholds from vegetation pixels
    healthy_thresh, moderate_thresh, veg_mean, veg_std = compute_adaptive_thresholds(
        vgi, mask, VEGETATION_CLASSES
    )
    
    # Update global thresholds for use in other functions
    VGI_HEALTHY_THRESHOLD = healthy_thresh
    VGI_MODERATE_THRESHOLD = moderate_thresh
    
    print(f"\n✓ Adaptive thresholds computed from vegetation pixels:")
    print(f"  - Vegetation classes: {[CLASS_NAMES.get(c, c) for c in VEGETATION_CLASSES]}")
    print(f"  - Vegetation VGI Mean: {veg_mean:.4f}")
    print(f"  - Vegetation VGI Std:  {veg_std:.4f}")
    print(f"\n  📊 COMPUTED THRESHOLDS:")
    print(f"  - Healthy threshold:  {healthy_thresh:.4f} (mean + 0.5*std)")
    print(f"  - Moderate threshold: {moderate_thresh:.4f} (mean - 0.5*std)")
    print(f"\n  Classification ranges:")
    print(f"  🟢 Healthy:  VGI > {healthy_thresh:.4f}")
    print(f"  🟡 Moderate: {moderate_thresh:.4f} <= VGI <= {healthy_thresh:.4f}")
    print(f"  🔴 Poor:     VGI < {moderate_thresh:.4f}")
    
    # ==================== GENERATE HEALTH REPORT ====================
    print("\n" + "-" * 50)
    print("GENERATING HEALTH REPORT (with adaptive thresholds)...")
    print("-" * 50)
    
    # Analyze all classes with adaptive thresholds
    df_all = generate_health_report(mask, vgi, list(CLASS_NAMES.keys()),
                                    healthy_thresh, moderate_thresh)
    
    # Print summary
    print("\n" + "=" * 60)
    print("CROP HEALTH SUMMARY (Adaptive VGI Thresholds)")
    print("=" * 60)
    
    for _, row in df_all.iterrows():
        if row['pixel_count'] > 0:
            health_indicator = "🟢" if row['health_category'] == "Healthy" else \
                              "🟡" if row['health_category'] == "Moderate" else "🔴"
            print(f"\n{health_indicator} {row['class_name']} (Class {row['class_id']}):")
            print(f"   Pixels: {row['pixel_count']:,}")
            print(f"   Area: {row['area_hectares']:.4f} hectares")
            print(f"   Mean VGI: {row['mean_vgi']:.4f}")
            print(f"   Health: {row['health_category']}")
    
    # ==================== SAVE OUTPUTS ====================
    print("\n" + "-" * 50)
    print("SAVING OUTPUTS...")
    print("-" * 50)
    
    # Save CSV report
    csv_path = output_dir / "crop_health_report.csv"
    df_all.to_csv(csv_path, index=False)
    print(f"✓ CSV report saved to: {csv_path}")
    
    # Save VGI bar chart with adaptive thresholds
    bar_chart_path = output_dir / "vgi_bar_chart.png"
    plot_vgi_bar_chart(df_all, str(bar_chart_path), healthy_thresh, moderate_thresh)
    
    # Save VGI overlay
    overlay_path = output_dir / "vgi_overlay.png"
    create_vgi_overlay(image, vgi, mask, str(overlay_path))
    
    # Save detailed report figure with adaptive thresholds
    report_fig_path = output_dir / "detailed_health_report.png"
    create_detailed_report_figure(df_all, vgi, mask, str(report_fig_path), 
                                  healthy_thresh, moderate_thresh)
    
    # ==================== SUMMARY ====================
    print("\n" + "=" * 70)
    print("CROP HEALTH ANALYSIS COMPLETE!")
    print("=" * 70)
    
    print(f"\nOutput files saved to: {output_dir.absolute()}")
    print("  - crop_health_report.csv: Detailed statistics table")
    print("  - vgi_bar_chart.png: VGI comparison by class")
    print("  - vgi_overlay.png: VGI heatmap overlay")
    print("  - detailed_health_report.png: Comprehensive visual report")
    
    # Print final statistics
    total_area = df_all['area_hectares'].sum()
    vegetation_df = df_all[df_all['class_id'].isin(VEGETATION_CLASSES)]
    veg_area = vegetation_df['area_hectares'].sum()
    
    print(f"\n📊 Final Statistics:")
    print(f"   Total analyzed area: {total_area:.4f} hectares")
    print(f"   Vegetation area: {veg_area:.4f} hectares ({veg_area/total_area*100:.1f}%)")
    
    healthy_area = df_all[df_all['health_category'] == 'Healthy']['area_hectares'].sum()
    moderate_area = df_all[df_all['health_category'] == 'Moderate']['area_hectares'].sum()
    poor_area = df_all[df_all['health_category'] == 'Poor']['area_hectares'].sum()
    
    print(f"\n🌱 Health Distribution (Adaptive Thresholds):")
    print(f"   Thresholds: Healthy>{healthy_thresh:.4f}, Moderate>{moderate_thresh:.4f}")
    print(f"   🟢 Healthy: {healthy_area:.4f} ha ({healthy_area/total_area*100:.1f}%)")
    print(f"   🟡 Moderate: {moderate_area:.4f} ha ({moderate_area/total_area*100:.1f}%)")
    print(f"   🔴 Poor: {poor_area:.4f} ha ({poor_area/total_area*100:.1f}%)")
    
    return df_all, vgi, mask


if __name__ == "__main__":
    df, vgi, mask = main()

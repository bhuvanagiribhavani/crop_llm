"""
================================================================================
PREDICTION SCRIPT FOR U-NET SEMANTIC SEGMENTATION
================================================================================
This script performs inference using a trained U-Net model on Sentinel-2 
satellite imagery for LULC (Land Use Land Cover) classification.

Features:
- Load trained model weights
- Apply same preprocessing as training (resize + ImageNet normalization)
- Run GPU-accelerated inference
- Generate predicted segmentation mask
- Visualize input image vs predicted mask
- Save predictions to file

Usage:
    python predict.py
    python predict.py --image_path path/to/image.png
    python predict.py --model_path outputs/best_model.pth --visualize

Author: Deep Learning Project
Date: 2026
================================================================================
"""

import os
import argparse
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Import model architecture
from model import get_model


# ============================================================================
# CONFIGURATION
# ============================================================================

# ImageNet normalization (same as training)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Class names for LULC classification
CLASS_NAMES = [
    "Class 0 - Water",
    "Class 1 - Trees", 
    "Class 2 - Grass",
    "Class 3 - Flooded Vegetation",
    "Class 4 - Crops",
    "Class 5 - Scrub/Shrub",
    "Class 6 - Built Area",
    "Class 7 - Bare Ground"
]

# Color map for visualization (8 classes)
CLASS_COLORS = [
    [0, 0, 255],       # Class 0 - Blue (Water)
    [0, 128, 0],       # Class 1 - Dark Green (Trees)
    [144, 238, 144],   # Class 2 - Light Green (Grass)
    [0, 255, 255],     # Class 3 - Cyan (Flooded Vegetation)
    [255, 255, 0],     # Class 4 - Yellow (Crops)
    [139, 69, 19],     # Class 5 - Brown (Scrub/Shrub)
    [255, 0, 0],       # Class 6 - Red (Built Area)
    [128, 128, 128]    # Class 7 - Gray (Bare Ground)
]


# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def load_and_preprocess_image(image_path, image_size=(256, 256)):
    """
    Load and preprocess an image for inference.
    
    Applies the same preprocessing as during training:
    1. Load as RGB
    2. Resize to target size
    3. Normalize to [0, 1]
    4. Apply ImageNet normalization
    5. Convert to tensor (C, H, W)
    
    Args:
        image_path (str): Path to the input image
        image_size (tuple): Target size (height, width)
    
    Returns:
        tuple: (preprocessed_tensor, original_image_array)
    """
    # Load image
    image = Image.open(image_path)
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Store original for visualization (before resize)
    original_size = image.size
    
    # Resize to target size
    image = image.resize(image_size, Image.BILINEAR)
    
    # Convert to numpy array
    image_np = np.array(image, dtype=np.float32)
    
    # Keep a copy for visualization (before normalization)
    image_for_viz = image_np.copy()
    
    # Normalize to [0, 1]
    image_np = image_np / 255.0
    
    # Apply ImageNet normalization
    image_np = (image_np - IMAGENET_MEAN) / IMAGENET_STD
    
    # Convert to tensor: (H, W, C) -> (C, H, W)
    image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float()
    
    # Add batch dimension: (C, H, W) -> (1, C, H, W)
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor, image_for_viz


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(model_path, num_classes=8, device='cuda'):
    """
    Load trained U-Net model from checkpoint.
    
    Args:
        model_path (str): Path to the model checkpoint (.pth file)
        num_classes (int): Number of output classes
        device (str): Device to load model on ('cuda' or 'cpu')
    
    Returns:
        nn.Module: Loaded model in evaluation mode
    """
    print(f"Loading model from: {model_path}")
    
    # Create model architecture
    model = get_model(
        in_channels=3,
        num_classes=num_classes,
        model_type='unet',
        device=device
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        loss = checkpoint.get('loss', 'unknown')
        print(f"  ✓ Model loaded successfully!")
        print(f"    - Trained for: {epoch} epochs")
        print(f"    - Best loss: {loss:.4f}" if isinstance(loss, float) else f"    - Best loss: {loss}")
    else:
        # Direct state dict
        model.load_state_dict(checkpoint)
        print(f"  ✓ Model loaded successfully!")
    
    # Set to evaluation mode
    model.eval()
    
    return model


# ============================================================================
# INFERENCE
# ============================================================================

def predict(model, image_tensor, device='cuda'):
    """
    Run inference on a single image.
    
    Args:
        model (nn.Module): Trained model
        image_tensor (torch.Tensor): Preprocessed image tensor (1, C, H, W)
        device (str): Device to run inference on
    
    Returns:
        numpy.ndarray: Predicted mask (H, W) with class labels 0-7
    """
    # Move to device
    image_tensor = image_tensor.to(device)
    
    # Run inference with no gradient computation
    with torch.no_grad():
        # Forward pass
        outputs = model(image_tensor)
        
        # Get class predictions using argmax
        # outputs shape: (1, num_classes, H, W)
        # predictions shape: (1, H, W)
        predictions = torch.argmax(outputs, dim=1)
        
        # Convert to numpy: (H, W)
        predicted_mask = predictions.squeeze(0).cpu().numpy()
    
    return predicted_mask


# ============================================================================
# VISUALIZATION
# ============================================================================

def colorize_mask(mask, colors=CLASS_COLORS):
    """
    Convert class mask to RGB colored image.
    
    Args:
        mask (numpy.ndarray): Class mask (H, W) with values 0-7
        colors (list): List of RGB colors for each class
    
    Returns:
        numpy.ndarray: RGB image (H, W, 3)
    """
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in enumerate(colors):
        colored_mask[mask == class_id] = color
    
    return colored_mask


def visualize_prediction(input_image, predicted_mask, ground_truth=None, 
                         save_path=None, show=True):
    """
    Visualize input image and predicted mask side by side.
    
    Args:
        input_image (numpy.ndarray): Original input image (H, W, 3)
        predicted_mask (numpy.ndarray): Predicted class mask (H, W)
        ground_truth (numpy.ndarray, optional): Ground truth mask (H, W)
        save_path (str, optional): Path to save visualization
        show (bool): Whether to display the plot
    """
    # Colorize predicted mask
    colored_pred = colorize_mask(predicted_mask)
    
    # Determine number of subplots
    if ground_truth is not None:
        colored_gt = colorize_mask(ground_truth)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        titles = ['Input Image', 'Ground Truth', 'Prediction']
        images = [input_image.astype(np.uint8), colored_gt, colored_pred]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        titles = ['Input Image', 'Predicted Mask']
        images = [input_image.astype(np.uint8), colored_pred]
    
    # Plot images
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=14)
        ax.axis('off')
    
    # Add legend
    legend_patches = []
    for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        color_normalized = [c/255 for c in color]
        patch = plt.Rectangle((0, 0), 1, 1, facecolor=color_normalized)
        legend_patches.append(patch)
    
    fig.legend(legend_patches, CLASS_NAMES, 
               loc='lower center', ncol=4, 
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Visualization saved to: {save_path}")
    
    # Show plot
    if show:
        plt.show()
    else:
        plt.close()


# ============================================================================
# MAIN PREDICTION FUNCTION
# ============================================================================

def main(args):
    """
    Main prediction pipeline.
    
    Args:
        args: Command line arguments
    """
    print("\n" + "=" * 70)
    print("U-NET PREDICTION - SENTINEL-2 LULC CLASSIFICATION")
    print("=" * 70)
    
    # ==================== DEVICE SETUP ====================
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\n✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("\n⚠ GPU not available, using CPU")
    
    # ==================== CREATE OUTPUT DIRECTORY ====================
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"✓ Output directory: {args.output_dir}")
    
    # ==================== LOAD MODEL ====================
    print("\n" + "-" * 50)
    print("LOADING MODEL...")
    print("-" * 50)
    
    model = load_model(
        model_path=args.model_path,
        num_classes=args.num_classes,
        device=device
    )
    
    # ==================== LOAD IMAGE ====================
    print("\n" + "-" * 50)
    print("LOADING IMAGE...")
    print("-" * 50)
    
    # If no image specified, get first image from test directory
    if args.image_path is None:
        test_images_dir = os.path.join(args.data_root, 'test_images')
        
        # Find first image file
        image_files = []
        for root, dirs, files in os.walk(test_images_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    image_files.append(os.path.join(root, f))
        
        if not image_files:
            raise ValueError(f"No images found in {test_images_dir}")
        
        image_files.sort()
        args.image_path = image_files[0]
        print(f"  Using first test image: {os.path.basename(args.image_path)}")
    
    print(f"  Image path: {args.image_path}")
    
    # Load and preprocess image
    image_tensor, original_image = load_and_preprocess_image(
        args.image_path, 
        image_size=(args.image_size, args.image_size)
    )
    print(f"  ✓ Image loaded and preprocessed")
    print(f"    - Size: {args.image_size}x{args.image_size}")
    print(f"    - Tensor shape: {image_tensor.shape}")
    
    # ==================== RUN INFERENCE ====================
    print("\n" + "-" * 50)
    print("RUNNING INFERENCE...")
    print("-" * 50)
    
    predicted_mask = predict(model, image_tensor, device)
    
    print(f"  ✓ Inference complete!")
    print(f"    - Output shape: {predicted_mask.shape}")
    print(f"    - Unique classes in prediction: {np.unique(predicted_mask)}")
    
    # Calculate class distribution
    total_pixels = predicted_mask.size
    print(f"    - Class distribution:")
    for class_id in range(args.num_classes):
        count = np.sum(predicted_mask == class_id)
        percentage = (count / total_pixels) * 100
        if count > 0:
            print(f"      Class {class_id}: {percentage:.2f}% ({count} pixels)")
    
    # ==================== SAVE PREDICTED MASK ====================
    print("\n" + "-" * 50)
    print("SAVING RESULTS...")
    print("-" * 50)
    
    # Save as grayscale mask (class values 0-7)
    mask_save_path = os.path.join(args.output_dir, 'predicted_mask.png')
    mask_image = Image.fromarray(predicted_mask.astype(np.uint8))
    mask_image.save(mask_save_path)
    print(f"  ✓ Predicted mask saved to: {mask_save_path}")
    
    # Save colorized mask
    colored_mask = colorize_mask(predicted_mask)
    colored_save_path = os.path.join(args.output_dir, 'predicted_mask_colored.png')
    colored_image = Image.fromarray(colored_mask)
    colored_image.save(colored_save_path)
    print(f"  ✓ Colored mask saved to: {colored_save_path}")
    
    # ==================== VISUALIZATION ====================
    if args.visualize:
        print("\n" + "-" * 50)
        print("GENERATING VISUALIZATION...")
        print("-" * 50)
        
        # Try to load ground truth mask
        ground_truth = None
        if args.load_gt:
            try:
                # Construct ground truth path
                image_name = os.path.basename(args.image_path)
                name_without_ext = os.path.splitext(image_name)[0]
                
                # Search for mask
                test_masks_dir = os.path.join(args.data_root, 'test_masks')
                for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                    gt_path = os.path.join(test_masks_dir, name_without_ext + ext)
                    if os.path.exists(gt_path):
                        gt_image = Image.open(gt_path)
                        if gt_image.mode != 'L':
                            gt_image = gt_image.convert('L')
                        gt_image = gt_image.resize((args.image_size, args.image_size), Image.NEAREST)
                        ground_truth = np.array(gt_image, dtype=np.int64)
                        print(f"  ✓ Ground truth loaded from: {gt_path}")
                        break
                
                # Also search in subdirectories
                if ground_truth is None:
                    for root, dirs, files in os.walk(test_masks_dir):
                        for f in files:
                            if f.startswith(name_without_ext):
                                gt_path = os.path.join(root, f)
                                gt_image = Image.open(gt_path)
                                if gt_image.mode != 'L':
                                    gt_image = gt_image.convert('L')
                                gt_image = gt_image.resize((args.image_size, args.image_size), Image.NEAREST)
                                ground_truth = np.array(gt_image, dtype=np.int64)
                                print(f"  ✓ Ground truth loaded from: {gt_path}")
                                break
                        if ground_truth is not None:
                            break
                            
            except Exception as e:
                print(f"  ⚠ Could not load ground truth: {e}")
        
        # Create visualization
        viz_save_path = os.path.join(args.output_dir, 'prediction_visualization.png')
        visualize_prediction(
            input_image=original_image,
            predicted_mask=predicted_mask,
            ground_truth=ground_truth,
            save_path=viz_save_path,
            show=args.show_plot
        )
    
    # ==================== COMPLETE ====================
    print("\n" + "=" * 70)
    print("PREDICTION COMPLETE!")
    print("=" * 70)
    print(f"\nOutput files saved to: {args.output_dir}/")
    print(f"  - predicted_mask.png (grayscale class labels)")
    print(f"  - predicted_mask_colored.png (RGB visualization)")
    if args.visualize:
        print(f"  - prediction_visualization.png (side-by-side comparison)")
    print("\n✓ Success!")
    
    return predicted_mask


# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run inference with trained U-Net on Sentinel-2 imagery',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input arguments
    parser.add_argument('--image_path', type=str, default=None,
                        help='Path to input image (if not specified, uses first test image)')
    parser.add_argument('--model_path', type=str, default='outputs/best_model.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_root', type=str, default='SEN-2 LULC',
                        help='Root directory of the dataset')
    
    # Model arguments
    parser.add_argument('--num_classes', type=int, default=8,
                        help='Number of output classes')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Input image size')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='outputs/predictions',
                        help='Directory to save predictions')
    
    # Visualization arguments
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Generate visualization')
    parser.add_argument('--no_visualize', action='store_false', dest='visualize',
                        help='Disable visualization')
    parser.add_argument('--load_gt', action='store_true', default=True,
                        help='Load ground truth for comparison')
    parser.add_argument('--no_gt', action='store_false', dest='load_gt',
                        help='Do not load ground truth')
    parser.add_argument('--show_plot', action='store_true', default=False,
                        help='Display plot (requires display)')
    
    return parser.parse_args()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    args = parse_args()
    predicted_mask = main(args)

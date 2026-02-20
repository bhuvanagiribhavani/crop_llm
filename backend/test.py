"""
================================================================================
GPU-OPTIMIZED TESTING SCRIPT FOR U-NET CROP SEGMENTATION
================================================================================
Features:
- Load best trained model checkpoint
- GPU-accelerated inference with mixed precision
- Compute IoU and Dice coefficient metrics
- Visualize predictions (input image, ground truth, prediction)
- Save metrics and visualizations

Author: Deep Learning Project
Date: 2026
================================================================================
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast

# Import custom modules
from dataset import get_data_loaders, get_num_classes, CropSegmentationDataset
from model import get_model, UNet, UNetSmall
from utils import (
    set_seed,
    get_device,
    calculate_iou,
    calculate_dice,
    visualize_prediction,
    visualize_batch,
    load_checkpoint,
    denormalize_image
)


# ============================================================================
# INFERENCE FUNCTION
# ============================================================================

def test(model, test_loader, device, num_classes, use_amp=True):
    """
    Run inference on test set and compute metrics.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: Device to run inference on (GPU/CPU)
        num_classes: Number of segmentation classes
        use_amp: Whether to use automatic mixed precision
    
    Returns:
        dict: Dictionary containing:
            - predictions: Predicted masks
            - targets: Ground truth masks
            - images: Input images
            - iou: IoU scores per class and mean
            - dice: Dice scores per class and mean
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_images = []
    
    print("\n" + "-" * 50)
    print("Running GPU-accelerated inference...")
    print("-" * 50)
    
    with torch.no_grad():  # Disable gradient computation for inference
        for images, masks in tqdm(test_loader, desc='Testing', ncols=100):
            # Move data to GPU (non-blocking for efficiency)
            images = images.to(device, non_blocking=True)
            
            # Mixed precision forward pass
            if use_amp and device.type == 'cuda':
                with autocast():
                    outputs = model(images)
            else:
                outputs = model(images)
            
            # Get predictions (argmax over classes)
            predictions = torch.argmax(outputs, dim=1)
            
            # Move to CPU and store
            all_predictions.append(predictions.cpu())
            all_targets.append(masks)
            all_images.append(images.cpu())
    
    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_images = torch.cat(all_images, dim=0)
    
    print(f"\nTotal test samples: {len(all_predictions)}")
    
    # ==================== CALCULATE METRICS ====================
    print("\nCalculating metrics...")
    
    # IoU (Intersection over Union)
    iou_scores = calculate_iou(all_predictions, all_targets, num_classes)
    
    # Dice coefficient
    dice_scores = calculate_dice(all_predictions, all_targets, num_classes)
    
    # ==================== PRINT RESULTS ====================
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    
    print("\nIoU (Intersection over Union) Scores:")
    for key, value in iou_scores.items():
        if 'class' in key:
            print(f"  {key}: {value:.4f}")
    print(f"\n  >>> Mean IoU: {iou_scores['mean_iou']:.4f} <<<")
    
    print("\nDice Coefficient Scores:")
    for key, value in dice_scores.items():
        if 'class' in key:
            print(f"  {key}: {value:.4f}")
    print(f"\n  >>> Mean Dice: {dice_scores['mean_dice']:.4f} <<<")
    
    print("=" * 60)
    
    return {
        'predictions': all_predictions,
        'targets': all_targets,
        'images': all_images,
        'iou': iou_scores,
        'dice': dice_scores
    }


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_results(results, output_dir, num_samples=5):
    """
    Generate and save visualizations of test results.
    
    Args:
        results: Dictionary containing test results
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
    """
    print(f"\nGenerating visualizations for {num_samples} samples...")
    
    # Create visualization directory
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    predictions = results['predictions']
    targets = results['targets']
    images = results['images']
    
    # Limit to available samples
    num_samples = min(num_samples, len(predictions))
    
    # Visualize individual samples
    for i in range(num_samples):
        # Denormalize image for visualization
        image = denormalize_image(images[i])
        mask = targets[i].numpy()
        pred = predictions[i].numpy()
        
        # Save visualization
        save_path = os.path.join(vis_dir, f'sample_{i+1}.png')
        visualize_prediction(image, mask, pred, save_path=save_path)
    
    # Visualize batch comparison (4 samples side by side)
    if num_samples >= 4:
        batch_images = [denormalize_image(images[i]) for i in range(4)]
        batch_masks = [targets[i].numpy() for i in range(4)]
        batch_preds = [predictions[i].numpy() for i in range(4)]
        
        batch_save_path = os.path.join(vis_dir, 'batch_comparison.png')
        visualize_batch(
            np.array(batch_images),
            np.array(batch_masks),
            np.array(batch_preds),
            num_samples=4,
            save_path=batch_save_path
        )
    
    print(f"Visualizations saved to: {vis_dir}")


def save_metrics(results, output_dir):
    """
    Save test metrics to a text file.
    
    Args:
        results: Dictionary containing test results
        output_dir: Directory to save the metrics file
    """
    metrics_path = os.path.join(output_dir, 'test_metrics.txt')
    
    with open(metrics_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("CROP SEGMENTATION - TEST METRICS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("IoU (Intersection over Union) Scores:\n")
        f.write("-" * 40 + "\n")
        for key, value in results['iou'].items():
            f.write(f"  {key}: {value:.4f}\n")
        
        f.write("\nDice Coefficient Scores:\n")
        f.write("-" * 40 + "\n")
        for key, value in results['dice'].items():
            f.write(f"  {key}: {value:.4f}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Mean IoU:  {results['iou']['mean_iou']:.4f}\n")
        f.write(f"Mean Dice: {results['dice']['mean_dice']:.4f}\n")
    
    print(f"Metrics saved to: {metrics_path}")


# ============================================================================
# SINGLE IMAGE INFERENCE
# ============================================================================

def run_inference_single_image(model, image_path, device, image_size=(256, 256), use_amp=True):
    """
    Run inference on a single image file.
    
    Args:
        model: Trained model
        image_path: Path to the input image
        device: Device to run inference on
        image_size: Target image size
        use_amp: Whether to use mixed precision
    
    Returns:
        numpy.ndarray: Predicted segmentation mask
    """
    from PIL import Image
    
    # Load and preprocess image
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    original_size = image.size
    image = image.resize(image_size, Image.BILINEAR)
    image = np.array(image, dtype=np.float32)
    
    # Normalize using ImageNet statistics
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    image = image / 255.0
    image = (image - mean) / std
    
    # Convert to tensor and add batch dimension
    image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        if use_amp and device.type == 'cuda':
            with autocast():
                output = model(image_tensor)
        else:
            output = model(image_tensor)
        
        prediction = torch.argmax(output, dim=1)
    
    # Convert to numpy
    prediction = prediction.squeeze().cpu().numpy()
    
    # Resize back to original size if different
    if original_size != image_size:
        pred_image = Image.fromarray(prediction.astype(np.uint8))
        pred_image = pred_image.resize(original_size, Image.NEAREST)
        prediction = np.array(pred_image)
    
    return prediction


# ============================================================================
# MAIN TESTING FUNCTION
# ============================================================================

def main(args):
    """
    Main testing function.
    
    Args:
        args: Command line arguments
    """
    print("\n" + "=" * 70)
    print("CROP CLASSIFICATION - U-NET TESTING (GPU-OPTIMIZED)")
    print("=" * 70)
    
    # Set random seed
    set_seed(args.seed)
    
    # Get device (automatically selects GPU if available)
    device = get_device()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ==================== DETERMINE NUM CLASSES ====================
    mask_dir = os.path.join(args.data_root, 'test_masks')
    if args.num_classes is None:
        num_classes = get_num_classes(mask_dir)
    else:
        num_classes = args.num_classes
    
    print(f"\nNumber of classes: {num_classes}")
    
    # ==================== LOAD MODEL ====================
    print("\n" + "-" * 50)
    print("LOADING MODEL...")
    print("-" * 50)
    
    # Create model architecture
    model = get_model(
        in_channels=3,
        num_classes=num_classes,
        model_type=args.model_type,
        device=device
    )
    
    # Load trained weights from checkpoint
    if os.path.exists(args.model_path):
        model, _, epoch, loss = load_checkpoint(model, None, args.model_path, device)
        print(f"✓ Model loaded from epoch {epoch} with validation loss {loss:.4f}")
    else:
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")
    
    # ==================== LOAD TEST DATA ====================
    print("\n" + "-" * 50)
    print("LOADING TEST DATA...")
    print("-" * 50)
    
    # Create test dataset
    test_dataset = CropSegmentationDataset(
        image_dir=os.path.join(args.data_root, 'test_images'),
        mask_dir=os.path.join(args.data_root, 'test_masks'),
        image_size=(args.image_size, args.image_size),
        augment=False  # No augmentation for testing
    )
    
    # Create test DataLoader (GPU-optimized)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True  # Faster GPU transfer
    )
    
    # ==================== RUN INFERENCE ====================
    results = test(model, test_loader, device, num_classes, use_amp=args.use_amp)
    
    # ==================== SAVE METRICS ====================
    save_metrics(results, args.output_dir)
    
    # ==================== VISUALIZE RESULTS ====================
    if args.visualize:
        visualize_results(results, args.output_dir, num_samples=args.num_visualize)
    
    # ==================== SUMMARY ====================
    print("\n" + "=" * 70)
    print("TESTING COMPLETE!")
    print("=" * 70)
    print(f"\nResults Summary:")
    print(f"  - Mean IoU:  {results['iou']['mean_iou']:.4f}")
    print(f"  - Mean Dice: {results['dice']['mean_dice']:.4f}")
    print(f"\nResults saved to: {args.output_dir}")
    
    return results


# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Test U-Net for Crop Segmentation (GPU-Optimized)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--data_root', type=str, default='miniDataSet',
                        help='Root directory of the dataset')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size (height and width)')
    parser.add_argument('--num_classes', type=int, default=None,
                        help='Number of classes (auto-detected if not specified)')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='unet',
                        choices=['unet', 'unet_small'],
                        help='Type of model architecture')
    parser.add_argument('--model_path', type=str, default='outputs/best_model.pth',
                        help='Path to the trained model checkpoint')
    
    # Inference arguments
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for inference')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision')
    parser.add_argument('--no_amp', action='store_false', dest='use_amp',
                        help='Disable automatic mixed precision')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='test_results',
                        help='Directory to save test results')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Generate visualizations')
    parser.add_argument('--no_visualize', action='store_false', dest='visualize',
                        help='Skip visualizations')
    parser.add_argument('--num_visualize', type=int, default=10,
                        help='Number of samples to visualize')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    return parser.parse_args()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    # Parse arguments
    args = parse_args()
    
    # Print configuration
    print("\n" + "=" * 70)
    print("TEST CONFIGURATION")
    print("=" * 70)
    for arg, value in sorted(vars(args).items()):
        print(f"  {arg}: {value}")
    
    # Run testing
    results = main(args)
    
    print("\n" + "=" * 70)
    print("Testing script completed successfully!")
    print("=" * 70)

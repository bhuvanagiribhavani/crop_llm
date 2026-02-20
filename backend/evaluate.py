"""
Comprehensive Model Evaluation Report Generator
Generates detailed metrics including accuracy, precision, recall, F1-score,
confusion matrix, and per-class analysis for semantic segmentation.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime

from dataset import CropSegmentationDataset, get_num_classes
from model import get_model


# Class names for crop classification (adjust based on your dataset)
CLASS_NAMES = [
    'Background',
    'Crop Type 1',
    'Crop Type 2', 
    'Crop Type 3',
    'Crop Type 4',
    'Crop Type 5',
    'Crop Type 6'
]


def calculate_metrics(pred, target, num_classes):
    """Calculate comprehensive metrics for segmentation."""
    pred = pred.flatten()
    target = target.flatten()
    
    # Overall pixel accuracy
    correct = (pred == target).sum()
    total = len(target)
    pixel_accuracy = correct / total
    
    # Per-class metrics
    class_metrics = {}
    for c in range(num_classes):
        # True positives, false positives, false negatives
        tp = ((pred == c) & (target == c)).sum()
        fp = ((pred == c) & (target != c)).sum()
        fn = ((pred != c) & (target == c)).sum()
        tn = ((pred != c) & (target != c)).sum()
        
        # Precision, Recall, F1
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        # IoU
        iou = tp / (tp + fp + fn + 1e-10)
        
        # Dice
        dice = 2 * tp / (2 * tp + fp + fn + 1e-10)
        
        # Class accuracy
        class_acc = (tp + tn) / (tp + tn + fp + fn + 1e-10)
        
        class_metrics[c] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iou': iou,
            'dice': dice,
            'accuracy': class_acc,
            'support': int((target == c).sum())
        }
    
    return pixel_accuracy, class_metrics


def evaluate_model_incremental(model, dataloader, device, num_classes, use_amp=True):
    """Run evaluation and compute metrics incrementally (memory efficient)."""
    model.eval()
    
    # Initialize confusion matrix and counters
    confusion_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_correct = 0
    total_pixels = 0
    
    # Per-class counters for metrics
    tp = np.zeros(num_classes, dtype=np.int64)
    fp = np.zeros(num_classes, dtype=np.int64)
    fn = np.zeros(num_classes, dtype=np.int64)
    support = np.zeros(num_classes, dtype=np.int64)
    
    # Store a few samples for visualization
    sample_preds = []
    sample_targets = []
    sample_images = []
    max_samples = 8
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(dataloader, desc="Evaluating")):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            with autocast(enabled=use_amp):
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
            
            # Move to CPU for metrics calculation
            preds_np = preds.cpu().numpy()
            masks_np = masks.cpu().numpy()
            
            # Save samples for visualization
            if len(sample_preds) < max_samples:
                sample_preds.append(preds_np[0])
                sample_targets.append(masks_np[0])
                sample_images.append(images[0].cpu().numpy())
            
            # Compute metrics incrementally for each image in batch
            for i in range(preds_np.shape[0]):
                pred_flat = preds_np[i].flatten()
                target_flat = masks_np[i].flatten()
                
                # Update confusion matrix
                for t, p in zip(target_flat, pred_flat):
                    if 0 <= t < num_classes and 0 <= p < num_classes:
                        confusion_mat[t, p] += 1
                
                # Update pixel accuracy
                total_correct += (pred_flat == target_flat).sum()
                total_pixels += len(target_flat)
                
                # Update per-class metrics
                for c in range(num_classes):
                    tp[c] += ((pred_flat == c) & (target_flat == c)).sum()
                    fp[c] += ((pred_flat == c) & (target_flat != c)).sum()
                    fn[c] += ((pred_flat != c) & (target_flat == c)).sum()
                    support[c] += (target_flat == c).sum()
    
    # Calculate final metrics
    pixel_accuracy = total_correct / total_pixels
    
    class_metrics = {}
    for c in range(num_classes):
        precision = tp[c] / (tp[c] + fp[c] + 1e-10)
        recall = tp[c] / (tp[c] + fn[c] + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        iou = tp[c] / (tp[c] + fp[c] + fn[c] + 1e-10)
        dice = 2 * tp[c] / (2 * tp[c] + fp[c] + fn[c] + 1e-10)
        
        class_metrics[c] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'iou': float(iou),
            'dice': float(dice),
            'support': int(support[c])
        }
    
    samples = {
        'images': sample_images,
        'preds': sample_preds,
        'targets': sample_targets
    }
    
    return pixel_accuracy, class_metrics, confusion_mat, samples


def generate_confusion_matrix_from_mat(cm, num_classes, save_path):
    """Generate and save confusion matrix visualization from pre-computed matrix."""
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Raw counts (use scientific notation for large numbers)
    sns.heatmap(cm, annot=True, fmt='.2e', cmap='Blues', ax=axes[0],
                xticklabels=CLASS_NAMES[:num_classes],
                yticklabels=CLASS_NAMES[:num_classes])
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14)
    axes[0].set_xlabel('Predicted', fontsize=12)
    axes[0].set_ylabel('True', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='y', rotation=0)
    
    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=axes[1],
                xticklabels=CLASS_NAMES[:num_classes],
                yticklabels=CLASS_NAMES[:num_classes])
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14)
    axes[1].set_xlabel('Predicted', fontsize=12)
    axes[1].set_ylabel('True', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return cm, cm_normalized


def generate_metrics_bar_chart(class_metrics, num_classes, save_path):
    """Generate bar chart comparing metrics across classes."""
    metrics = ['precision', 'recall', 'f1', 'iou', 'dice']
    x = np.arange(num_classes)
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
    
    for i, metric in enumerate(metrics):
        values = [class_metrics[c][metric] for c in range(num_classes)]
        bars = ax.bar(x + i * width, values, width, label=metric.upper(), color=colors[i])
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Metrics Comparison', fontsize=14)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(CLASS_NAMES[:num_classes], rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_report(args):
    """Generate comprehensive evaluation report."""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    
    print("\n" + "=" * 70)
    print("MODEL EVALUATION REPORT GENERATOR")
    print("=" * 70)
    print(f"✓ Using: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model first to get correct num_classes
    print("\n" + "-" * 50)
    print("LOADING MODEL...")
    print("-" * 50)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Get num_classes from checkpoint or detect from model weights
    if 'num_classes' in checkpoint:
        num_classes = checkpoint['num_classes']
    else:
        # Detect num_classes from the output layer weights
        for key in checkpoint['model_state_dict']:
            if 'outc.conv.weight' in key or 'outc.conv.bias' in key:
                num_classes = checkpoint['model_state_dict'][key].shape[0]
                break
        else:
            num_classes = 8  # Default fallback
    
    print(f"Number of classes (detected from model): {num_classes}")
    
    # Update class names if needed
    global CLASS_NAMES
    if num_classes > len(CLASS_NAMES):
        CLASS_NAMES = [f'Class {i}' for i in range(num_classes)]
    
    model = get_model(in_channels=3, num_classes=num_classes, model_type=args.model_type, device=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded from: {args.model_path}")
    print(f"  Trained for {checkpoint['epoch']} epochs")
    if 'val_loss' in checkpoint:
        print(f"  Best validation loss: {checkpoint['val_loss']:.4f}")
    elif 'loss' in checkpoint:
        print(f"  Best loss: {checkpoint['loss']:.4f}")
    
    # Load test data
    print("\n" + "-" * 50)
    print("LOADING TEST DATA...")
    print("-" * 50)
    
    test_dataset = CropSegmentationDataset(
        image_dir=os.path.join(args.data_root, 'test_images'),
        mask_dir=os.path.join(args.data_root, 'test_masks'),
        image_size=(args.image_size, args.image_size),
        augment=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"✓ Test samples: {len(test_dataset)}")
    
    # Run evaluation (memory-efficient incremental)
    print("\n" + "-" * 50)
    print("RUNNING EVALUATION...")
    print("-" * 50)
    
    pixel_accuracy, class_metrics, confusion_mat, samples = evaluate_model_incremental(
        model, test_loader, device, num_classes, args.use_amp
    )
    
    # Calculate mean metrics
    print("\nCalculating aggregate metrics...")
    mean_precision = np.mean([class_metrics[c]['precision'] for c in range(num_classes)])
    mean_recall = np.mean([class_metrics[c]['recall'] for c in range(num_classes)])
    mean_f1 = np.mean([class_metrics[c]['f1'] for c in range(num_classes)])
    mean_iou = np.mean([class_metrics[c]['iou'] for c in range(num_classes)])
    mean_dice = np.mean([class_metrics[c]['dice'] for c in range(num_classes)])
    
    # Weighted metrics (by support)
    total_support = sum([class_metrics[c]['support'] for c in range(num_classes)])
    weighted_precision = sum([class_metrics[c]['precision'] * class_metrics[c]['support'] for c in range(num_classes)]) / total_support
    weighted_recall = sum([class_metrics[c]['recall'] * class_metrics[c]['support'] for c in range(num_classes)]) / total_support
    weighted_f1 = sum([class_metrics[c]['f1'] * class_metrics[c]['support'] for c in range(num_classes)]) / total_support
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Confusion matrix (use pre-computed matrix)
    cm, cm_norm = generate_confusion_matrix_from_mat(
        confusion_mat, num_classes,
        os.path.join(args.output_dir, 'confusion_matrix.png')
    )
    print("  ✓ Confusion matrix saved")
    
    # Metrics bar chart
    generate_metrics_bar_chart(
        class_metrics, num_classes,
        os.path.join(args.output_dir, 'metrics_comparison.png')
    )
    print("  ✓ Metrics comparison chart saved")
    
    # Generate text report
    report_path = os.path.join(args.output_dir, 'evaluation_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("CROP CLASSIFICATION MODEL - EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Test Dataset: {args.data_root}/test_images\n")
        f.write(f"Test Samples: {len(test_dataset)}\n")
        f.write(f"Image Size: {args.image_size}x{args.image_size}\n")
        f.write(f"Number of Classes: {num_classes}\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("OVERALL METRICS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"{'Metric':<25} {'Score':>10}\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Pixel Accuracy':<25} {pixel_accuracy:>10.4f} ({pixel_accuracy*100:.2f}%)\n")
        f.write(f"{'Mean IoU (mIoU)':<25} {mean_iou:>10.4f} ({mean_iou*100:.2f}%)\n")
        f.write(f"{'Mean Dice':<25} {mean_dice:>10.4f} ({mean_dice*100:.2f}%)\n")
        f.write(f"{'Mean Precision':<25} {mean_precision:>10.4f} ({mean_precision*100:.2f}%)\n")
        f.write(f"{'Mean Recall':<25} {mean_recall:>10.4f} ({mean_recall*100:.2f}%)\n")
        f.write(f"{'Mean F1-Score':<25} {mean_f1:>10.4f} ({mean_f1*100:.2f}%)\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Weighted Precision':<25} {weighted_precision:>10.4f} ({weighted_precision*100:.2f}%)\n")
        f.write(f"{'Weighted Recall':<25} {weighted_recall:>10.4f} ({weighted_recall*100:.2f}%)\n")
        f.write(f"{'Weighted F1-Score':<25} {weighted_f1:>10.4f} ({weighted_f1*100:.2f}%)\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("PER-CLASS METRICS\n")
        f.write("=" * 70 + "\n\n")
        
        header = f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'IoU':>10} {'Dice':>10} {'Support':>10}\n"
        f.write(header)
        f.write("-" * 75 + "\n")
        
        for c in range(num_classes):
            m = class_metrics[c]
            class_name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f'Class {c}'
            f.write(f"{class_name:<15} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['iou']:>10.4f} {m['dice']:>10.4f} {m['support']:>10d}\n")
        
        f.write("-" * 75 + "\n")
        f.write(f"{'Macro Avg':<15} {mean_precision:>10.4f} {mean_recall:>10.4f} {mean_f1:>10.4f} {mean_iou:>10.4f} {mean_dice:>10.4f} {total_support:>10d}\n")
        f.write(f"{'Weighted Avg':<15} {weighted_precision:>10.4f} {weighted_recall:>10.4f} {weighted_f1:>10.4f} {'-':>10} {'-':>10} {total_support:>10d}\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("CONFUSION MATRIX (Row: True, Col: Predicted)\n")
        f.write("=" * 70 + "\n\n")
        
        # Header
        f.write(f"{'':>12}")
        for c in range(num_classes):
            f.write(f"{'C'+str(c):>10}")
        f.write("\n")
        f.write("-" * (12 + 10 * num_classes) + "\n")
        
        for i in range(num_classes):
            f.write(f"{'C'+str(i)+' (True)':<12}")
            for j in range(num_classes):
                f.write(f"{cm[i, j]:>10d}")
            f.write("\n")
        
        f.write("\n")
        f.write("=" * 70 + "\n")
        f.write("CLASS DISTRIBUTION\n")
        f.write("=" * 70 + "\n\n")
        
        for c in range(num_classes):
            support = class_metrics[c]['support']
            pct = support / total_support * 100
            class_name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f'Class {c}'
            bar = '█' * int(pct / 2)
            f.write(f"{class_name:<15} {support:>10} pixels ({pct:>5.2f}%) {bar}\n")
        
        f.write("\n")
        f.write("=" * 70 + "\n")
        f.write("INTERPRETATION GUIDE\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("• Pixel Accuracy: Overall percentage of correctly classified pixels\n")
        f.write("• IoU (Intersection over Union): Overlap between prediction and ground truth\n")
        f.write("• Dice Coefficient: F1-score variant for segmentation (2*TP / (2*TP + FP + FN))\n")
        f.write("• Precision: Ratio of true positives to all predicted positives\n")
        f.write("• Recall: Ratio of true positives to all actual positives\n")
        f.write("• F1-Score: Harmonic mean of precision and recall\n\n")
        
        f.write("Performance Thresholds:\n")
        f.write("  - Excellent: IoU > 0.7, Dice > 0.8\n")
        f.write("  - Good: IoU 0.5-0.7, Dice 0.65-0.8\n")
        f.write("  - Fair: IoU 0.3-0.5, Dice 0.45-0.65\n")
        f.write("  - Poor: IoU < 0.3, Dice < 0.45\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")
    
    # Print summary to console
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\n{'OVERALL METRICS':^40}")
    print("-" * 40)
    print(f"  Pixel Accuracy:    {pixel_accuracy*100:6.2f}%")
    print(f"  Mean IoU (mIoU):   {mean_iou*100:6.2f}%")
    print(f"  Mean Dice:         {mean_dice*100:6.2f}%")
    print(f"  Mean Precision:    {mean_precision*100:6.2f}%")
    print(f"  Mean Recall:       {mean_recall*100:6.2f}%")
    print(f"  Mean F1-Score:     {mean_f1*100:6.2f}%")
    print("-" * 40)
    print(f"  Weighted F1:       {weighted_f1*100:6.2f}%")
    
    print(f"\n{'PER-CLASS PERFORMANCE':^50}")
    print("-" * 50)
    print(f"{'Class':<15} {'IoU':>8} {'Dice':>8} {'F1':>8} {'Support':>10}")
    print("-" * 50)
    
    for c in range(num_classes):
        m = class_metrics[c]
        class_name = CLASS_NAMES[c][:12] if c < len(CLASS_NAMES) else f'Class {c}'
        print(f"{class_name:<15} {m['iou']*100:>7.2f}% {m['dice']*100:>7.2f}% {m['f1']*100:>7.2f}% {m['support']:>10}")
    
    print("-" * 50)
    
    print("\n" + "=" * 70)
    print("SAVED FILES")
    print("=" * 70)
    print(f"  ✓ evaluation_report.txt - Detailed text report")
    print(f"  ✓ confusion_matrix.png  - Confusion matrix visualization")
    print(f"  ✓ metrics_comparison.png - Per-class metrics chart")
    print(f"\n  All files saved to: {args.output_dir}/")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70 + "\n")
    
    return {
        'pixel_accuracy': pixel_accuracy,
        'mean_iou': mean_iou,
        'mean_dice': mean_dice,
        'mean_precision': mean_precision,
        'mean_recall': mean_recall,
        'mean_f1': mean_f1,
        'weighted_f1': weighted_f1,
        'class_metrics': class_metrics
    }


def main():
    parser = argparse.ArgumentParser(description='Generate Model Evaluation Report')
    
    parser.add_argument('--model_path', type=str, default='outputs/best_model.pth',
                        help='Path to trained model')
    parser.add_argument('--model_type', type=str, default='unet',
                        help='Model architecture')
    parser.add_argument('--data_root', type=str, default='miniDataSet',
                        help='Root directory of dataset')
    parser.add_argument('--output_dir', type=str, default='evaluation_report',
                        help='Directory to save report')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for evaluation')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use mixed precision')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    generate_report(args)


if __name__ == '__main__':
    main()

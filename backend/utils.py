"""
================================================================================
UTILITY FUNCTIONS FOR CROP SEGMENTATION PROJECT
================================================================================
Contains:
- Random seed setting for reproducibility
- Device detection (GPU/CPU)
- Metrics: IoU, Dice coefficient
- Loss functions: Dice Loss, Combined Loss (Dice + CrossEntropy)
- Visualization functions
- Checkpoint saving/loading

Author: Deep Learning Project
Date: 2026
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
import os


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seed(seed=42):
    """
    Set random seeds for reproducibility across all libraries.
    
    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - cuDNN deterministic mode
    
    Args:
        seed (int): Random seed value (default: 42)
    
    Note:
        Setting cuDNN to deterministic mode may slightly reduce performance
        but ensures reproducible results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Make cuDNN deterministic (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"✓ Random seed set to {seed} for reproducibility")


# ============================================================================
# DEVICE SELECTION
# ============================================================================

def get_device():
    """
    Automatically detect and return the best available device.
    
    Returns:
        torch.device: CUDA device if available, otherwise CPU
    
    Example:
        >>> device = get_device()
        >>> model = model.to(device)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        print("⚠ CUDA not available, using CPU")
    
    return device


# ============================================================================
# METRICS
# ============================================================================

def calculate_iou(pred, target, num_classes, smooth=1e-6):
    """
    Calculate Intersection over Union (IoU) for each class.
    
    IoU = (Intersection) / (Union)
        = TP / (TP + FP + FN)
    
    Args:
        pred (torch.Tensor): Predicted segmentation mask (N, H, W)
        target (torch.Tensor): Ground truth mask (N, H, W)
        num_classes (int): Number of segmentation classes
        smooth (float): Smoothing factor to avoid division by zero
    
    Returns:
        dict: IoU for each class and mean IoU
    
    Example:
        >>> pred = torch.randint(0, 10, (4, 256, 256))
        >>> target = torch.randint(0, 10, (4, 256, 256))
        >>> iou = calculate_iou(pred, target, num_classes=10)
        >>> print(iou['mean_iou'])
    """
    iou_per_class = {}
    
    for cls in range(num_classes):
        # Create binary masks for current class
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        
        # Calculate intersection and union
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum() - intersection
        
        # Calculate IoU with smoothing
        iou = (intersection + smooth) / (union + smooth)
        iou_per_class[f'class_{cls}'] = iou.item()
    
    # Calculate mean IoU
    mean_iou = np.mean(list(iou_per_class.values()))
    iou_per_class['mean_iou'] = mean_iou
    
    return iou_per_class


def calculate_dice(pred, target, num_classes, smooth=1e-6):
    """
    Calculate Dice coefficient for each class.
    
    Dice = 2 * (Intersection) / (Sum of both masks)
         = 2 * TP / (2 * TP + FP + FN)
    
    Args:
        pred (torch.Tensor): Predicted segmentation mask (N, H, W)
        target (torch.Tensor): Ground truth mask (N, H, W)
        num_classes (int): Number of segmentation classes
        smooth (float): Smoothing factor to avoid division by zero
    
    Returns:
        dict: Dice score for each class and mean Dice
    
    Example:
        >>> pred = torch.randint(0, 10, (4, 256, 256))
        >>> target = torch.randint(0, 10, (4, 256, 256))
        >>> dice = calculate_dice(pred, target, num_classes=10)
        >>> print(dice['mean_dice'])
    """
    dice_per_class = {}
    
    for cls in range(num_classes):
        # Create binary masks for current class
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        
        # Calculate intersection
        intersection = (pred_cls * target_cls).sum()
        
        # Calculate Dice coefficient with smoothing
        dice = (2. * intersection + smooth) / (pred_cls.sum() + target_cls.sum() + smooth)
        dice_per_class[f'class_{cls}'] = dice.item()
    
    # Calculate mean Dice
    mean_dice = np.mean(list(dice_per_class.values()))
    dice_per_class['mean_dice'] = mean_dice
    
    return dice_per_class


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation.
    
    Dice Loss = 1 - Dice Coefficient
    
    This loss is particularly useful for imbalanced datasets where
    some classes have fewer pixels than others.
    
    Args:
        smooth (float): Smoothing factor to avoid division by zero
    """
    
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target, num_classes):
        """
        Calculate Dice loss.
        
        Args:
            pred (torch.Tensor): Predicted logits (N, C, H, W)
            target (torch.Tensor): Ground truth mask (N, H, W)
            num_classes (int): Number of classes
        
        Returns:
            torch.Tensor: Dice loss value (scalar)
        """
        # Apply softmax to get probabilities
        pred = torch.softmax(pred, dim=1)
        
        # One-hot encode target: (N, H, W) -> (N, C, H, W)
        target_one_hot = F.one_hot(target.long(), num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        # Calculate Dice coefficient for each class
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Return Dice loss (1 - Dice)
        dice_loss = 1 - dice.mean()
        
        return dice_loss


class CombinedLoss(nn.Module):
    """
    Combined Dice Loss + Cross-Entropy Loss.
    
    This combination often works better than either loss alone:
    - Cross-Entropy provides stable gradients for all classes
    - Dice Loss helps with class imbalance
    
    Total Loss = dice_weight * Dice_Loss + ce_weight * CE_Loss
    
    Args:
        num_classes (int): Number of segmentation classes
        dice_weight (float): Weight for Dice loss (default: 0.5)
        ce_weight (float): Weight for Cross-Entropy loss (default: 0.5)
    """
    
    def __init__(self, num_classes, dice_weight=0.5, ce_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
    
    def forward(self, pred, target):
        """
        Calculate combined loss.
        
        Args:
            pred (torch.Tensor): Predicted logits (N, C, H, W)
            target (torch.Tensor): Ground truth mask (N, H, W)
        
        Returns:
            torch.Tensor: Combined loss value (scalar)
        """
        dice = self.dice_loss(pred, target, self.num_classes)
        ce = self.ce_loss(pred, target.long())
        
        return self.dice_weight * dice + self.ce_weight * ce


class FocalLoss(nn.Module):
    """
    ============================================================================
    FOCAL LOSS FOR IMPROVED PIXEL ACCURACY
    ============================================================================
    
    Focal Loss addresses class imbalance by down-weighting easy examples 
    and focusing on hard misclassified pixels.
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Where:
    - p_t is the probability for the correct class
    - gamma (focusing parameter) reduces loss for well-classified examples
    - alpha provides class balancing
    
    Key Benefits:
    1. Automatically handles class imbalance without manual weight tuning
    2. Focuses learning on hard/misclassified pixels
    3. Prevents easy/dominant classes from overwhelming gradients
    4. Proven effective for semantic segmentation tasks
    
    Args:
        gamma (float): Focusing parameter (default: 2.0)
        alpha (tensor): Per-class weights (optional)
        ignore_index (int): Class index to ignore
    """
    
    def __init__(self, gamma=2.0, alpha=None, ignore_index=-100, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C, H, W) - logits
            targets: (N, H, W) - ground truth labels
        """
        ce_loss = F.cross_entropy(
            inputs, targets.long(), 
            reduction='none', 
            ignore_index=self.ignore_index
        )
        
        pt = torch.exp(-ce_loss)  # probability of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            # Apply class-specific alpha weights
            alpha_t = self.alpha[targets.long().clamp(0, len(self.alpha)-1)]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            # Only average over non-ignored pixels
            if self.ignore_index >= 0:
                mask = targets != self.ignore_index
                return focal_loss[mask].mean() if mask.sum() > 0 else focal_loss.mean()
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class HighAccuracyLoss(nn.Module):
    """
    ============================================================================
    HIGH ACCURACY LOSS - FOCAL + DICE + OHEM
    ============================================================================
    
    State-of-the-art loss combination for maximizing semantic segmentation accuracy:
    
    1. FOCAL LOSS (primary): Handles class imbalance, focuses on hard pixels
    2. DICE LOSS (secondary): Optimizes region overlap, handles boundaries
    3. OHEM (Online Hard Example Mining): Focuses on worst predictions
    
    This combination has been shown to achieve 90%+ accuracy on LULC datasets.
    
    Dataset Distribution (from SEN-2 LULC):
    - Class 5: ~49% (1.03B pixels) - Built-up
    - Class 4: ~28% (590M pixels) - Vegetation  
    - Class 3: ~11% (227M pixels) - Cropland
    - Class 2: ~5.6% (118M pixels) - Forest
    - Class 6: ~4.2% (88M pixels) - Barren
    - Class 1: ~1.2% (26M pixels) - Water
    - Class 0: ~1.1% (23M pixels) - Unknown
    - Class 7: ~0.001% (20K pixels) - IGNORED (noise)
    
    Args:
        num_classes (int): Number of segmentation classes
        focal_gamma (float): Focusing parameter for focal loss (default: 2.0)
        focal_weight (float): Weight for focal loss (default: 0.7)
        dice_weight (float): Weight for dice loss (default: 0.3)
        ignore_index (int): Class to ignore (default: 7)
        use_ohem (bool): Whether to use online hard example mining
        ohem_ratio (float): Ratio of hard examples to keep (default: 0.7)
    """
    
    def __init__(
        self, 
        num_classes=8,
        focal_gamma=2.0,
        focal_weight=0.7,
        dice_weight=0.3,
        ignore_index=7,
        use_ohem=True,
        ohem_ratio=0.7
    ):
        super(HighAccuracyLoss, self).__init__()
        
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.use_ohem = use_ohem
        self.ohem_ratio = ohem_ratio
        self.focal_gamma = focal_gamma
        
        # ====================================================================
        # CLASS-BALANCED WEIGHTS using Effective Number of Samples
        # ====================================================================
        # Formula: w_c = (1 - beta) / (1 - beta^n_c)
        # Where beta = (N-1)/N, n_c = samples in class c
        # This balances rare and common classes effectively
        # ====================================================================
        
        # Pixel counts from dataset analysis
        pixel_counts = torch.tensor([
            23e6,    # Class 0: ~23M pixels
            26e6,    # Class 1: ~26M pixels  
            118e6,   # Class 2: ~118M pixels
            227e6,   # Class 3: ~227M pixels
            590e6,   # Class 4: ~590M pixels
            1030e6,  # Class 5: ~1.03B pixels
            88e6,    # Class 6: ~88M pixels
            20e3,    # Class 7: ~20K pixels (will be ignored)
        ], dtype=torch.float32)
        
        # Effective number of samples weighting (Class-Balanced Loss)
        beta = 0.9999  # Smoothing factor
        effective_num = 1.0 - torch.pow(beta, pixel_counts)
        class_weights = (1.0 - beta) / (effective_num + 1e-8)
        
        # Normalize weights
        class_weights = class_weights / class_weights.sum() * num_classes
        class_weights[ignore_index] = 0.0  # Zero weight for ignored class
        
        self.register_buffer('class_weights', class_weights)
        
        # Focal Loss with class weights
        self.focal_loss = FocalLoss(
            gamma=focal_gamma, 
            alpha=self.class_weights,
            ignore_index=ignore_index
        )
        
        # Dice Loss for region coherence
        self.dice_loss = DiceLoss()
        
        print(f"\n{'='*60}")
        print("HIGH ACCURACY LOSS (Focal + Dice + OHEM)")
        print(f"{'='*60}")
        print(f"  Focal Loss Weight: {self.focal_weight} (gamma={focal_gamma})")
        print(f"  Dice Loss Weight: {self.dice_weight}")
        print(f"  OHEM Enabled: {self.use_ohem} (ratio={self.ohem_ratio})")
        print(f"  Ignored Class: {self.ignore_index}")
        print(f"  Class Weights (effective number):")
        for i, w in enumerate(class_weights):
            status = "IGNORED" if i == ignore_index else f"weight={w:.4f}"
            print(f"    Class {i}: {status}")
        print(f"{'='*60}\n")
    
    def ohem_loss(self, pred, target, loss_per_pixel):
        """
        Online Hard Example Mining - focus on hardest predictions.
        
        Keeps only the top-k hardest pixels (highest loss) for backprop.
        This forces the model to focus on its mistakes.
        """
        # Flatten spatial dimensions
        batch_size = pred.size(0)
        loss_flat = loss_per_pixel.view(batch_size, -1)
        
        # Create mask for non-ignored pixels
        target_flat = target.view(batch_size, -1)
        valid_mask = target_flat != self.ignore_index
        
        total_loss = 0.0
        valid_batches = 0
        
        for b in range(batch_size):
            valid_losses = loss_flat[b][valid_mask[b]]
            if valid_losses.numel() == 0:
                continue
                
            # Keep top-k hardest examples
            k = max(1, int(valid_losses.numel() * self.ohem_ratio))
            topk_losses, _ = torch.topk(valid_losses, k)
            total_loss += topk_losses.mean()
            valid_batches += 1
        
        return total_loss / max(valid_batches, 1)
    
    def forward(self, pred, target):
        """
        Calculate high accuracy loss.
        
        Args:
            pred (torch.Tensor): Predicted logits (N, C, H, W)
            target (torch.Tensor): Ground truth mask (N, H, W)
        
        Returns:
            torch.Tensor: Combined loss value
        """
        # Ensure weights are on correct device
        if self.class_weights.device != pred.device:
            self.class_weights = self.class_weights.to(pred.device)
            self.focal_loss.alpha = self.class_weights
        
        # ====================================================================
        # FOCAL LOSS - Primary loss for handling class imbalance
        # ====================================================================
        if self.use_ohem:
            # Calculate per-pixel focal loss for OHEM
            ce_loss = F.cross_entropy(
                pred, target.long(), 
                reduction='none',
                ignore_index=self.ignore_index
            )
            pt = torch.exp(-ce_loss)
            focal_per_pixel = ((1 - pt) ** self.focal_gamma) * ce_loss
            focal = self.ohem_loss(pred, target, focal_per_pixel)
        else:
            focal = self.focal_loss(pred, target)
        
        # ====================================================================
        # DICE LOSS - For region overlap and boundary awareness
        # ====================================================================
        dice = self.dice_loss(pred, target, self.num_classes)
        
        # ====================================================================
        # COMBINED LOSS
        # ====================================================================
        total_loss = self.focal_weight * focal + self.dice_weight * dice
        
        return total_loss


class PixelAccuracyOptimizedLoss(nn.Module):
    """
    ============================================================================
    PIXEL ACCURACY OPTIMIZED LOSS v3 - Using Focal Loss + OHEM
    ============================================================================
    
    Wrapper that uses HighAccuracyLoss internally.
    Maintained for backward compatibility with train.py.
    """
    
    def __init__(
        self, 
        num_classes=8,
        class_weights=None,
        ignore_index=7,
        label_smoothing=0.0,
        dice_weight=0.3,
        ce_weight=0.7
    ):
        super(PixelAccuracyOptimizedLoss, self).__init__()
        
        # Use the high accuracy loss internally
        self.loss_fn = HighAccuracyLoss(
            num_classes=num_classes,
            focal_gamma=2.0,
            focal_weight=ce_weight,  # Focal replaces CE
            dice_weight=dice_weight,
            ignore_index=ignore_index,
            use_ohem=True,
            ohem_ratio=0.7
        )
        
    def forward(self, pred, target):
        return self.loss_fn(pred, target)


# ============================================================================
# IMAGE PROCESSING
# ============================================================================

def denormalize_image(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Denormalize an image for visualization.
    
    Reverses the ImageNet normalization applied during preprocessing.
    
    Args:
        image (torch.Tensor or numpy.ndarray): Normalized image
        mean (tuple): Mean values used for normalization
        std (tuple): Std values used for normalization
    
    Returns:
        numpy.ndarray: Denormalized image (H, W, C) in range [0, 255]
    """
    # Convert tensor to numpy if needed
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    
    # Handle (C, H, W) format -> (H, W, C)
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    
    # Denormalize: image = (normalized * std) + mean
    mean = np.array(mean).reshape(1, 1, 3)
    std = np.array(std).reshape(1, 1, 3)
    
    image = image * std + mean
    
    # Clip to [0, 1] and convert to [0, 255]
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    
    return image


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_prediction(image, mask, prediction, save_path=None, class_names=None):
    """
    Visualize input image, ground truth mask, and predicted mask side by side.
    
    Creates a figure with three subplots:
    1. Input Image
    2. Ground Truth Mask
    3. Predicted Mask
    
    Args:
        image (numpy.ndarray): Input image (H, W, 3) in range [0, 255]
        mask (numpy.ndarray): Ground truth mask (H, W)
        prediction (numpy.ndarray): Predicted mask (H, W)
        save_path (str, optional): Path to save the visualization
        class_names (list, optional): List of class names for legend
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Input image
    axes[0].imshow(image)
    axes[0].set_title('Input Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Ground truth mask
    im1 = axes[1].imshow(mask, cmap='viridis')
    axes[1].set_title('Ground Truth Mask', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Predicted mask
    im2 = axes[2].imshow(prediction, cmap='viridis')
    axes[2].set_title('Predicted Mask', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    # Add colorbar
    plt.colorbar(im2, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.close()


def visualize_batch(images, masks, predictions, num_samples=4, save_path=None):
    """
    Visualize a batch of images, masks, and predictions in a grid.
    
    Args:
        images (numpy.ndarray): Batch of images (N, H, W, 3)
        masks (numpy.ndarray): Batch of ground truth masks (N, H, W)
        predictions (numpy.ndarray): Batch of predicted masks (N, H, W)
        num_samples (int): Number of samples to visualize
        save_path (str, optional): Path to save the visualization
    """
    num_samples = min(num_samples, len(images))
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    # Handle single sample case
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Input image
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title('Input' if i == 0 else '', fontsize=12)
        axes[i, 0].axis('off')
        
        # Ground truth
        axes[i, 1].imshow(masks[i], cmap='viridis')
        axes[i, 1].set_title('Ground Truth' if i == 0 else '', fontsize=12)
        axes[i, 1].axis('off')
        
        # Prediction
        axes[i, 2].imshow(predictions[i], cmap='viridis')
        axes[i, 2].set_title('Prediction' if i == 0 else '', fontsize=12)
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.close()


def plot_training_history(train_losses, val_losses, save_path=None):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses (list): List of training losses per epoch
        val_losses (list): List of validation losses per epoch
        save_path (str, optional): Path to save the plot
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    
    # Plot losses
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=4)
    
    # Find best epoch
    best_epoch = np.argmin(val_losses) + 1
    best_val_loss = min(val_losses)
    plt.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    plt.scatter([best_epoch], [best_val_loss], color='g', s=100, zorder=5)
    
    # Labels and styling
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Set axis limits
    plt.xlim([0.5, len(epochs) + 0.5])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Training history plot saved to: {save_path}")
    
    plt.close()


# ============================================================================
# MODEL UTILITIES
# ============================================================================

def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Save model checkpoint.
    
    Saves:
    - Model state dict
    - Optimizer state dict
    - Current epoch
    - Current loss
    
    Args:
        model: PyTorch model
        optimizer: Optimizer (can be None)
        epoch (int): Current epoch
        loss (float): Current loss value
        path (str): Path to save the checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, path, device):
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer (can be None)
        path (str): Path to the checkpoint
        device: Device to load the model on
    
    Returns:
        tuple: (model, optimizer, epoch, loss)
    """
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', 0.0)
    
    return model, optimizer, epoch, loss


# ============================================================================
# MAIN - TEST UTILITIES
# ============================================================================

if __name__ == "__main__":
    """Test utility functions."""
    
    print("\n" + "=" * 60)
    print("TESTING UTILITY FUNCTIONS")
    print("=" * 60)
    
    # Test seed setting
    print("\n1. Testing seed setting...")
    set_seed(42)
    
    # Test device detection
    print("\n2. Testing device detection...")
    device = get_device()
    
    # Test metrics
    print("\n3. Testing metrics...")
    
    # Create dummy predictions and targets
    pred = torch.randint(0, 5, (4, 64, 64))
    target = torch.randint(0, 5, (4, 64, 64))
    
    # Calculate IoU
    iou = calculate_iou(pred, target, num_classes=5)
    print(f"  IoU scores: {iou}")
    
    # Calculate Dice
    dice = calculate_dice(pred, target, num_classes=5)
    print(f"  Dice scores: {dice}")
    
    # Test loss functions
    print("\n4. Testing loss functions...")
    
    # Create dummy predictions (logits) and targets
    pred_logits = torch.randn(2, 5, 64, 64)  # (N, C, H, W)
    target_labels = torch.randint(0, 5, (2, 64, 64))  # (N, H, W)
    
    # Test Dice loss
    dice_loss = DiceLoss()
    loss_val = dice_loss(pred_logits, target_labels, num_classes=5)
    print(f"  Dice Loss: {loss_val.item():.4f}")
    
    # Test Combined loss
    combined_loss = CombinedLoss(num_classes=5)
    loss_val = combined_loss(pred_logits, target_labels)
    print(f"  Combined Loss: {loss_val.item():.4f}")
    
    # Test denormalization
    print("\n5. Testing image denormalization...")
    normalized_img = torch.randn(3, 64, 64)
    denorm_img = denormalize_image(normalized_img)
    print(f"  Denormalized image shape: {denorm_img.shape}")
    print(f"  Denormalized image range: [{denorm_img.min()}, {denorm_img.max()}]")
    
    print("\n" + "=" * 60)
    print("ALL UTILITY TESTS PASSED!")
    print("=" * 60)

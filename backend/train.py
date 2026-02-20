"""
================================================================================
GPU-OPTIMIZED TRAINING SCRIPT FOR U-NET CROP SEGMENTATION
================================================================================
Features:
- Automatic GPU detection and utilization
- Mixed precision training (torch.cuda.amp) for faster training
- cuDNN benchmark mode for optimal performance
- Epoch-wise training and validation with progress bars
- Learning rate scheduling (ReduceLROnPlateau / CosineAnnealing)
- Best model checkpointing based on validation loss
- Training history plotting and logging

Author: Deep Learning Project
Date: 2026
================================================================================
"""

import os
import argparse
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

# Import custom modules
from dataset import get_data_loaders, get_num_classes
from model import get_model
from utils import (
    set_seed,
    get_device,
    CombinedLoss,
    PixelAccuracyOptimizedLoss,
    HighAccuracyLoss,
    save_checkpoint,
    plot_training_history,
    count_parameters
)


# ============================================================================
# GPU OPTIMIZATION SETTINGS
# ============================================================================

def setup_gpu_optimizations():
    """
    Configure PyTorch for optimal GPU performance.
    
    Optimizations:
    1. cuDNN benchmark: Finds fastest convolution algorithms
    2. cuDNN enabled: Uses cuDNN for convolutions
    3. TF32 precision: Allows TF32 on Ampere GPUs for faster training
    """
    if torch.cuda.is_available():
        # Enable cuDNN benchmark mode for optimal performance
        # This finds the best algorithm for your specific input sizes
        torch.backends.cudnn.benchmark = True
        
        # Ensure cuDNN is enabled
        torch.backends.cudnn.enabled = True
        
        # Allow TF32 on Ampere+ GPUs (RTX 30xx, A100, etc.)
        # Provides ~2x speedup with minimal accuracy loss
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        print("GPU Optimizations Enabled:")
        print("  - cuDNN benchmark: True")
        print("  - cuDNN enabled: True")
        print("  - TF32 allowed: True")
    else:
        print("Warning: CUDA not available, running on CPU")


# ============================================================================
# TRAINING FUNCTION (SINGLE EPOCH)
# ============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device, 
                    epoch, num_epochs, scaler, use_amp=True):
    """
    Train the model for one epoch with mixed precision support.
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device (GPU/CPU)
        epoch: Current epoch number
        num_epochs: Total number of epochs
        scaler: GradScaler for mixed precision
        use_amp: Whether to use automatic mixed precision
    
    Returns:no
        float: Average training loss for the epoch
    """
    model.train()
    running_loss = 0.0
    
    # Progress bar with epoch information
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs} [Train]', 
                leave=True, ncols=100)
    
    for batch_idx, (images, masks) in enumerate(pbar):
        # Move data to GPU (non-blocking for efficiency with pin_memory)
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        # Zero gradients
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        
        # Mixed precision forward pass
        if use_amp and device.type == 'cuda':
            with autocast():
                # Forward pass
                outputs = model(images)
                # Calculate loss
                loss = criterion(outputs, masks)
            
            # Mixed precision backward pass
            scaler.scale(loss).backward()
            
            # Unscale gradients and update weights
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard precision training
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
        
        # Accumulate loss
        running_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # ================================================================
        # FREQUENT CHECKPOINT SAVING (every 100 batches ~1-2 minutes)
        # ================================================================
        if (batch_idx + 1) % 100 == 0:
            # Save mid-epoch checkpoint for resume capability
            mid_checkpoint = {
                'epoch': epoch,
                'batch_idx': batch_idx + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'running_loss': running_loss,
                'scaler_state_dict': scaler.state_dict() if scaler else None,
            }
            torch.save(mid_checkpoint, 'outputs/mid_epoch_checkpoint.pth')
    
    # Calculate average loss
    avg_loss = running_loss / len(train_loader)
    
    return avg_loss


# ============================================================================
# VALIDATION FUNCTION
# ============================================================================

def validate(model, val_loader, criterion, device, epoch, num_epochs, use_amp=True):
    """
    Validate the model on validation set.
    
    Args:
        model: Neural network model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device (GPU/CPU)
        epoch: Current epoch number
        num_epochs: Total number of epochs
        use_amp: Whether to use automatic mixed precision
    
    Returns:
        float: Average validation loss
    """
    model.eval()
    running_loss = 0.0
    
    # Progress bar for validation
    pbar = tqdm(val_loader, desc=f'Epoch {epoch}/{num_epochs} [Val]  ', 
                leave=True, ncols=100)
    
    with torch.no_grad():  # Disable gradient computation
        for images, masks in pbar:
            # Move data to GPU
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            # Mixed precision forward pass
            if use_amp and device.type == 'cuda':
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)
            else:
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            # Accumulate loss
            running_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Calculate average loss
    avg_loss = running_loss / len(val_loader)
    
    return avg_loss


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train(args):
    """
    Main training function with all optimizations.
    
    Args:
        args: Command line arguments containing all training configuration
    
    Returns:
        tuple: (trained_model, train_losses, val_losses)
    """
    print("\n" + "=" * 70)
    print("CROP CLASSIFICATION - U-NET TRAINING (GPU-OPTIMIZED)")
    print("=" * 70)
    
    # ==================== SETUP ====================
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Get device (automatically selects GPU if available)
    device = get_device()
    
    # Setup GPU optimizations
    setup_gpu_optimizations()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nOutput directory: {args.output_dir}")
    
    # ==================== DATA ====================
    print("\n" + "-" * 50)
    print("LOADING DATA...")
    print("-" * 50)
    
    # Get data loaders (GPU-optimized with pin_memory)
    loaders = get_data_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        image_size=(args.image_size, args.image_size),
        num_workers=args.num_workers,
        pin_memory=True  # Essential for GPU training
    )
    
    train_loader = loaders['train']
    val_loader = loaders['val']
    
    # Determine number of classes
    mask_dir = os.path.join(args.data_root, 'train_masks')
    if args.num_classes is None:
        num_classes = get_num_classes(mask_dir)
    else:
        num_classes = args.num_classes
    
    print(f"\nNumber of classes: {num_classes}")
    
    # ==================== MODEL ====================
    print("\n" + "-" * 50)
    print("CREATING MODEL...")
    print("-" * 50)
    
    model = get_model(
        in_channels=3,
        num_classes=num_classes,
        model_type=args.model_type,
        device=device
    )
    
    # ==================== LOSS & OPTIMIZER ====================
    print("-" * 50)
    print("SETTING UP LOSS AND OPTIMIZER...")
    print("-" * 50)
    
    # Loss function
    if args.loss_type == 'ce':
        criterion = nn.CrossEntropyLoss()
        print("Loss: CrossEntropyLoss")
    elif args.loss_type == 'combined':
        criterion = CombinedLoss(num_classes=num_classes, dice_weight=0.5, ce_weight=0.5)
        print("Loss: Combined (Dice + CrossEntropy)")
    elif args.loss_type == 'pixel_accuracy':
        # ====================================================================
        # HIGH ACCURACY LOSS v3 - FOCAL + DICE + OHEM
        # ====================================================================
        # State-of-the-art loss combination for maximum accuracy:
        # - Focal Loss (70%): Handles class imbalance, focuses on hard pixels
        # - Dice Loss (30%): Region overlap and boundary awareness
        # - OHEM: Online Hard Example Mining for difficult samples
        # - Class-balanced weights using effective number of samples
        # - Ignores Class 7 (too rare/noisy)
        # ====================================================================
        criterion = HighAccuracyLoss(
            num_classes=num_classes,
            focal_gamma=2.0,          # Focusing parameter
            focal_weight=0.7,         # 70% Focal Loss
            dice_weight=0.3,          # 30% Dice Loss
            ignore_index=7,           # Ignore rare/noisy class
            use_ohem=True,            # Online Hard Example Mining
            ohem_ratio=0.7            # Keep 70% hardest pixels
        )
        print("Loss: High Accuracy (Focal + Dice + OHEM)")
    else:
        raise ValueError(f"Unknown loss type: {args.loss_type}")
    
    # Optimizer (Adam with configurable learning rate and weight decay)
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    print(f"Optimizer: Adam (lr={args.learning_rate}, weight_decay={args.weight_decay})")
    
    # Learning rate scheduler
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min',      # Minimize validation loss
            factor=0.5,      # Reduce LR by half
            patience=5,      # Wait 5 epochs before reducing
            min_lr=1e-7
        )
        print("Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)")
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
        print("Scheduler: CosineAnnealingLR")
    else:
        scheduler = None
        print("Scheduler: None")
    
    # ==================== MIXED PRECISION SETUP ====================
    # GradScaler for mixed precision training
    scaler = GradScaler() if args.use_amp and device.type == 'cuda' else None
    print(f"Mixed Precision (AMP): {'Enabled' if args.use_amp and device.type == 'cuda' else 'Disabled'}")
    
    # ==================== RESUME FROM CHECKPOINT ====================
    start_epoch = 1
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    
    # Find checkpoint to resume from
    resume_path = args.resume
    if resume_path is None and args.auto_resume:
        # Auto-find latest checkpoint
        checkpoint_files = [f for f in os.listdir(args.output_dir) 
                           if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
        if checkpoint_files:
            # Sort by epoch number and get the latest
            checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].replace('.pth', '')))
            resume_path = os.path.join(args.output_dir, checkpoint_files[-1])
            print(f"\n✓ Auto-resume: Found checkpoint {checkpoint_files[-1]}")
    
    # Load checkpoint if available
    if resume_path and os.path.exists(resume_path):
        print(f"\n" + "-" * 50)
        print("RESUMING FROM CHECKPOINT...")
        print("-" * 50)
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('loss', float('inf'))
        best_epoch = checkpoint.get('epoch', 0)
        
        # Load training history if exists
        history_path = os.path.join(args.output_dir, 'training_history.csv')
        if os.path.exists(history_path):
            import csv
            with open(history_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if int(row['epoch']) < start_epoch:
                        train_losses.append(float(row['train_loss']))
                        val_losses.append(float(row['val_loss']))
        
        print(f"  ✓ Resumed from epoch {checkpoint['epoch']}")
        print(f"  ✓ Best val loss so far: {best_val_loss:.4f}")
        print(f"  ✓ Continuing from epoch {start_epoch}")
    
    # ==================== TRAINING LOOP ====================
    print("\n" + "=" * 70)
    print("STARTING TRAINING...")
    print("=" * 70)
    
    # Start timer
    start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        print(f"\n{'='*70}")
        
        # Train one epoch
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, args.epochs, scaler, use_amp=args.use_amp
        )
        
        # Validate
        val_loss = validate(
            model, val_loader, criterion, device,
            epoch, args.epochs, use_amp=args.use_amp
        )
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start
        
        # Store losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs} Summary:")
        print(f"  Train Loss:     {train_loss:.4f}")
        print(f"  Val Loss:       {val_loss:.4f}")
        print(f"  Learning Rate:  {optimizer.param_groups[0]['lr']:.2e}")
        print(f"  Epoch Time:     {epoch_time:.2f}s")
        
        # Print GPU memory usage
        if device.type == 'cuda':
            print(f"  GPU Memory:     {torch.cuda.memory_allocated(0) / 1024**2:.0f} MB")
        
        # Update learning rate scheduler
        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Save best model (based on validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            
            # Save checkpoint
            checkpoint_path = os.path.join(args.output_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
            print(f"  ✓ NEW BEST MODEL SAVED! (Val Loss: {val_loss:.4f})")
        
        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
        
        # Save training history after each epoch (for resume support)
        history_path = os.path.join(args.output_dir, 'training_history.csv')
        with open(history_path, 'w') as f:
            f.write("epoch,train_loss,val_loss\n")
            for ep, (t_loss, v_loss) in enumerate(zip(train_losses, val_losses), 1):
                f.write(f"{ep},{t_loss:.6f},{v_loss:.6f}\n")
    
    # ==================== TRAINING COMPLETE ====================
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nTraining Summary:")
    print(f"  - Total training time: {total_time / 60:.2f} minutes")
    print(f"  - Best validation loss: {best_val_loss:.4f} (Epoch {best_epoch})")
    print(f"  - Final training loss: {train_losses[-1]:.4f}")
    print(f"  - Final validation loss: {val_losses[-1]:.4f}")
    print(f"\nModel saved to: {os.path.join(args.output_dir, 'best_model.pth')}")
    
    # Save final model
    final_path = os.path.join(args.output_dir, 'final_model.pth')
    save_checkpoint(model, optimizer, args.epochs, val_losses[-1], final_path)
    print(f"Final model saved to: {final_path}")
    
    # Plot and save training history
    history_plot_path = os.path.join(args.output_dir, 'training_history.png')
    plot_training_history(train_losses, val_losses, save_path=history_plot_path)
    
    # Save training history as CSV
    history_path = os.path.join(args.output_dir, 'training_history.csv')
    with open(history_path, 'w') as f:
        f.write("epoch,train_loss,val_loss\n")
        for epoch, (t_loss, v_loss) in enumerate(zip(train_losses, val_losses), 1):
            f.write(f"{epoch},{t_loss:.6f},{v_loss:.6f}\n")
    print(f"Training history saved to: {history_path}")
    
    return model, train_losses, val_losses


# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train U-Net for Crop Segmentation (GPU-Optimized)',
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
                        help='Type of model to use')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for regularization')
    
    # Loss and scheduler arguments
    parser.add_argument('--loss_type', type=str, default='combined',
                        choices=['ce', 'combined', 'pixel_accuracy'],
                        help='Loss function type (ce=CrossEntropy, combined=Dice+CE, pixel_accuracy=Optimized for Pixel Accuracy)')
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine', 'none'],
                        help='Learning rate scheduler')
    
    # GPU optimization arguments
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision (AMP)')
    parser.add_argument('--no_amp', action='store_false', dest='use_amp',
                        help='Disable automatic mixed precision')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save outputs')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    # Resume arguments
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (e.g., outputs/checkpoint_epoch_10.pth)')
    parser.add_argument('--auto_resume', action='store_true', default=True,
                        help='Automatically resume from latest checkpoint if available')
    
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
    # Parse command line arguments
    args = parse_args()
    
    # Print configuration
    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    for arg, value in sorted(vars(args).items()):
        print(f"  {arg}: {value}")
    
    # Train the model
    model, train_losses, val_losses = train(args)
    
    print("\n" + "=" * 70)
    print("Training script completed successfully!")
    print("=" * 70)

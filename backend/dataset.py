"""
================================================================================
CROP CLASSIFICATION DATASET - SENTINEL-2 SATELLITE IMAGERY
================================================================================
Custom PyTorch Dataset for semantic segmentation of satellite images.

Features:
- RGB image loading and preprocessing
- Image and mask resizing to 256x256
- ImageNet normalization
- Data augmentation (horizontal/vertical flip, rotation) for training
- GPU-optimized DataLoader configuration

Author: Deep Learning Project
Date: 2026
================================================================================
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random


# ============================================================================
# CUSTOM DATASET CLASS
# ============================================================================

class CropSegmentationDataset(Dataset):
    """
    Custom PyTorch Dataset for loading satellite images and segmentation masks.
    
    This dataset handles:
    1. Loading RGB images from specified directory
    2. Loading corresponding segmentation masks
    3. Resizing to target size (256x256)
    4. Normalizing images using ImageNet statistics
    5. Applying data augmentation (training only)
    
    Args:
        image_dir (str): Path to directory containing images
        mask_dir (str): Path to directory containing masks
        image_size (tuple): Target size for resizing (height, width)
        augment (bool): Whether to apply data augmentation
        mean (tuple): Mean values for normalization (ImageNet defaults)
        std (tuple): Standard deviation values for normalization
    
    Example:
        >>> dataset = CropSegmentationDataset(
        ...     image_dir='miniDataSet/train_images',
        ...     mask_dir='miniDataSet/train_masks',
        ...     augment=True
        ... )
        >>> image, mask = dataset[0]
        >>> print(image.shape)  # torch.Size([3, 256, 256])
        >>> print(mask.shape)   # torch.Size([256, 256])
    """
    
    def __init__(
        self,
        image_dir,
        mask_dir,
        image_size=(256, 256),
        augment=False,
        mean=(0.485, 0.456, 0.406),  # ImageNet mean
        std=(0.229, 0.224, 0.225)     # ImageNet std
    ):
        """Initialize the dataset with paths and configuration."""
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.augment = augment
        
        # Store normalization parameters as arrays for efficient computation
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        
        # Get list of image files
        self.image_files = self._get_image_files()
        
        # Print dataset information
        print(f"  Dataset initialized:")
        print(f"    - Image directory: {image_dir}")
        print(f"    - Mask directory: {mask_dir}")
        print(f"    - Number of samples: {len(self.image_files)}")
        print(f"    - Image size: {image_size}")
        print(f"    - Augmentation: {augment}")
    
    def _get_image_files(self):
        """
        Get list of image files from the image directory.
        Supports nested directory structures.
        
        Returns:
            list: Sorted list of image file paths
        
        Raises:
            ValueError: If no images are found in the directory
        """
        image_files = []
        supported_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
        
        # Walk through directory to handle nested structure
        for root, dirs, files in os.walk(self.image_dir):
            for file in files:
                if file.lower().endswith(supported_extensions):
                    image_files.append(os.path.join(root, file))
        
        # Sort for reproducibility
        image_files.sort()
        
        if len(image_files) == 0:
            raise ValueError(f"No images found in {self.image_dir}")
        
        return image_files
    
    def _get_mask_path(self, image_path):
        """
        Get the corresponding mask path for an image.
        
        Args:
            image_path (str): Path to the image file
        
        Returns:
            str: Path to the corresponding mask file
        
        Raises:
            FileNotFoundError: If no matching mask is found
        """
        filename = os.path.basename(image_path)
        name_without_ext = os.path.splitext(filename)[0]
        
        # Search for mask with same name (any supported extension)
        for root, dirs, files in os.walk(self.mask_dir):
            # First, try exact filename match
            if filename in files:
                return os.path.join(root, filename)
            
            # Then, try matching with different extensions
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
                mask_filename = name_without_ext + ext
                if mask_filename in files:
                    return os.path.join(root, mask_filename)
        
        raise FileNotFoundError(f"No mask found for image: {image_path}")
    
    def _load_image(self, path):
        """
        Load and preprocess an RGB image.
        
        Args:ot
            path (str): Path to the image file
        
        Returns:
            numpy.ndarray: Image as float32 array (H, W, 3)
        """
        # Load image using PIL
        image = Image.open(path)
        
        # Convert to RGB if necessary (handles grayscale, RGBA, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to target size using bilinear interpolation
        image = image.resize(self.image_size, Image.BILINEAR)
        
        # Convert to numpy array as float32
        image = np.array(image, dtype=np.float32)
        
        return image
    
    def _load_mask(self, path):
        """
        Load and preprocess a segmentation mask.
        
        Masks are loaded as single-channel integer arrays.
        Nearest neighbor interpolation preserves label values.
        Labels are shifted to start from 0 if they start from 1.
        
        Args:
            path (str): Path to the mask file
        
        Returns:
            numpy.ndarray: Mask as int64 array (H, W)
        """
        # Load mask using PIL
        mask = Image.open(path)
        
        # Convert to grayscale (single channel) if necessary
        if mask.mode != 'L':
            mask = mask.convert('L')
        
        # Resize using nearest neighbor to preserve label values
        mask = mask.resize(self.image_size, Image.NEAREST)
        
        # Convert to numpy array as integer (NOT normalized!)
        mask = np.array(mask, dtype=np.int64)
        
        # Shift labels to start from 0 (PyTorch CrossEntropyLoss expects 0-indexed labels)
        # If min label is 1, subtract 1 to make it 0-indexed
        if mask.min() >= 1:
            mask = mask - 1
        
        return mask
    
    def _normalize_image(self, image):
        """
        Normalize image using ImageNet mean and standard deviation.
        
        Steps:
        1. Scale pixel values from [0, 255] to [0, 1]
        2. Apply mean and std normalization
        
        Args:
            image (numpy.ndarray): Image array (H, W, 3) in range [0, 255]
        
        Returns:
            numpy.ndarray: Normalized image
        """
        # Scale to [0, 1]
        image = image / 255.0
        
        # Apply ImageNet normalization: (pixel - mean) / std
        image = (image - self.mean) / self.std
        
        return image
    
    def _augment(self, image, mask):
        """
        Apply random data augmentation to image and mask.
        
        Augmentations (each with 50% probability):
        1. Random horizontal flip
        2. Random vertical flip
        3. Random 90-degree rotations (0, 90, 180, or 270 degrees)
        
        Args:
            image (numpy.ndarray): Image array (H, W, 3)
            mask (numpy.ndarray): Mask array (H, W)
        
        Returns:
            tuple: Augmented (image, mask)
        """
        # Random horizontal flip (50% probability)
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
        
        # Random vertical flip (50% probability)
        if random.random() > 0.5:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()
        
        # Random 90-degree rotation (0, 90, 180, or 270 degrees)
        k = random.randint(0, 3)  # Number of 90-degree rotations
        if k > 0:
            image = np.rot90(image, k).copy()
            mask = np.rot90(mask, k).copy()
        
        return image, mask
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
        
        Returns:
            tuple: (image_tensor, mask_tensor)
                - image_tensor: Float tensor of shape (3, H, W)
                - mask_tensor: Long tensor of shape (H, W) with class labels
        """
        # Get paths
        image_path = self.image_files[idx]
        mask_path = self._get_mask_path(image_path)
        
        # Load image and mask
        image = self._load_image(image_path)
        mask = self._load_mask(mask_path)
        
        # Apply augmentation BEFORE normalization (only for training)
        if self.augment:
            image, mask = self._augment(image, mask)
        
        # Normalize image AFTER augmentation
        image = self._normalize_image(image)
        
        # Convert to PyTorch tensors
        # Image: (H, W, C) -> (C, H, W) for PyTorch format
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        # Mask: Keep as (H, W) with long dtype for CrossEntropyLoss
        mask_tensor = torch.from_numpy(mask).long()
        
        return image_tensor, mask_tensor
    
    def get_original_image(self, idx):
        """
        Get the original (non-normalized) image for visualization.
        
        Args:
            idx (int): Index of the sample
        
        Returns:
            numpy.ndarray: Original image (H, W, 3) in range [0, 255]
        """
        image_path = self.image_files[idx]
        image = self._load_image(image_path)
        return image.astype(np.uint8)


# ============================================================================
# DATA LOADER FACTORY FUNCTION
# ============================================================================

def get_data_loaders(
    data_root,
    batch_size=8,
    image_size=(256, 256),
    num_workers=4,
    pin_memory=True
):
    """
    Create GPU-optimized DataLoaders for training, validation, and testing.
    
    GPU Optimizations:
    - pin_memory=True: Faster host-to-device transfers
    - num_workers > 0: Parallel data loading
    - persistent_workers: Reduces worker startup overhead
    
    Args:
        data_root (str): Root directory containing train/val/test folders
        batch_size (int): Batch size for data loaders (default: 8)
        image_size (tuple): Target image size (height, width)
        num_workers (int): Number of worker processes
        pin_memory (bool): Pin memory for faster GPU transfer
    
    Returns:
        dict: Dictionary containing DataLoaders and Datasets
    """
    # Define directory paths
    train_image_dir = os.path.join(data_root, 'train_images')
    train_mask_dir = os.path.join(data_root, 'train_masks')
    val_image_dir = os.path.join(data_root, 'val_images')
    val_mask_dir = os.path.join(data_root, 'val_masks')
    test_image_dir = os.path.join(data_root, 'test_images')
    test_mask_dir = os.path.join(data_root, 'test_masks')
    
    print("=" * 60)
    print("CREATING DATASETS")
    print("=" * 60)
    
    # Create training dataset WITH augmentation
    print("\n[1/3] Training Dataset:")
    train_dataset = CropSegmentationDataset(
        image_dir=train_image_dir,
        mask_dir=train_mask_dir,
        image_size=image_size,
        augment=True  # ONLY training data gets augmentation
    )
    
    # Create validation dataset WITHOUT augmentation
    print("\n[2/3] Validation Dataset:")
    val_dataset = CropSegmentationDataset(
        image_dir=val_image_dir,
        mask_dir=val_mask_dir,
        image_size=image_size,
        augment=False
    )
    
    # Create test dataset WITHOUT augmentation
    print("\n[3/3] Test Dataset:")
    test_dataset = CropSegmentationDataset(
        image_dir=test_image_dir,
        mask_dir=test_mask_dir,
        image_size=image_size,
        augment=False
    )
    
    print("=" * 60)
    
    # Determine persistent workers setting
    use_persistent = num_workers > 0
    
    # Create GPU-optimized DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,               # Shuffle for training
        num_workers=num_workers,    # Parallel data loading
        pin_memory=pin_memory,      # Faster GPU transfer
        drop_last=True,             # Drop incomplete batch
        persistent_workers=use_persistent
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent
    )
    
    # Print summary
    print("\nDataLoaders Created (GPU-Optimized):")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Num workers: {num_workers}")
    print(f"  - Pin memory: {pin_memory}")
    print(f"  - Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"  - Val:   {len(val_loader)} batches ({len(val_dataset)} samples)")
    print(f"  - Test:  {len(test_loader)} batches ({len(test_dataset)} samples)")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset
    }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_num_classes(mask_dir):
    """
    Determine the number of unique classes by scanning mask files.
    
    Args:
        mask_dir (str): Path to directory containing mask files
    
    Returns:
        int: Number of unique classes found in masks
    """
    unique_classes = set()
    
    for root, dirs, files in os.walk(mask_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                mask_path = os.path.join(root, file)
                mask = Image.open(mask_path)
                if mask.mode != 'L':
                    mask = mask.convert('L')
                mask_array = np.array(mask)
                unique_classes.update(np.unique(mask_array).tolist())
    
    num_classes = len(unique_classes)
    print(f"Found {num_classes} unique classes: {sorted(unique_classes)}")
    
    return num_classes


# ============================================================================
# MAIN - TEST THE DATASET
# ============================================================================

if __name__ == "__main__":
    """Test the dataset and dataloader functionality."""
    
    print("\n" + "=" * 60)
    print("TESTING CROP SEGMENTATION DATASET")
    print("=" * 60)
    
    # Dataset root path
    data_root = "miniDataSet"
    
    # Check if dataset exists
    if not os.path.exists(data_root):
        print(f"Error: Dataset not found at {data_root}")
        print("Please ensure the miniDataSet folder exists with train/val/test splits.")
        exit(1)
    
    # Get number of classes
    mask_dir = os.path.join(data_root, 'train_masks')
    if os.path.exists(mask_dir):
        num_classes = get_num_classes(mask_dir)
    
    # Create data loaders
    try:
        loaders = get_data_loaders(data_root, batch_size=4, num_workers=2)
        
        # Test loading a batch
        print("\n" + "-" * 40)
        print("Testing Data Loading...")
        print("-" * 40)
        
        for images, masks in loaders['train']:
            print(f"\nBatch Information:")
            print(f"  - Image batch shape: {images.shape}")
            print(f"  - Mask batch shape: {masks.shape}")
            print(f"  - Image dtype: {images.dtype}")
            print(f"  - Mask dtype: {masks.dtype}")
            print(f"  - Image value range: [{images.min():.3f}, {images.max():.3f}]")
            print(f"  - Mask unique values: {torch.unique(masks).tolist()}")
            break
        
        print("\n" + "=" * 60)
        print("DATASET TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

"""
================================================================================
U-NET MODEL FOR SEMANTIC SEGMENTATION
================================================================================
Implementation of the classic U-Net architecture from scratch with:
- Encoder (contracting path) with max pooling
- Decoder (expanding path) with upsampling
- Skip connections between encoder and decoder
- Batch normalization and ReLU activations
- GPU-optimized operations

Reference: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical 
           Image Segmentation", MICCAI 2015

Author: Deep Learning Project
Date: 2026
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# BUILDING BLOCKS
# ============================================================================

class DoubleConv(nn.Module):
    """
    Double Convolution Block: (Conv -> BatchNorm -> ReLU) x 2
    
    This is the basic building block of U-Net, used in both encoder and decoder.
    Each convolution uses 3x3 kernels with padding=1 to preserve spatial dimensions.
    
    Architecture:
        Input -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU -> Output
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        mid_channels (int, optional): Number of channels after first conv
    
    Example:
        >>> block = DoubleConv(64, 128)
        >>> x = torch.randn(1, 64, 256, 256)
        >>> output = block(x)
        >>> print(output.shape)  # torch.Size([1, 128, 256, 256])
    """
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        
        # If mid_channels not specified, use out_channels
        if mid_channels is None:
            mid_channels = out_channels
        
        # Sequential block of two convolutions
        self.double_conv = nn.Sequential(
            # First convolution: in_channels -> mid_channels
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            
            # Second convolution: mid_channels -> out_channels
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """Forward pass through double convolution."""
        return self.double_conv(x)


class EncoderBlock(nn.Module):
    """
    Encoder Block: MaxPool2x2 -> DoubleConv
    
    Downsamples the feature map by factor of 2 and extracts higher-level features.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    
    Example:
        >>> encoder = EncoderBlock(64, 128)
        >>> x = torch.randn(1, 64, 256, 256)
        >>> output = encoder(x)
        >>> print(output.shape)  # torch.Size([1, 128, 128, 128])
    """
    
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),  # Halve spatial dimensions
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        """Forward pass with downsampling."""
        return self.maxpool_conv(x)


class DecoderBlock(nn.Module):
    """
    Decoder Block: Upsample -> Concatenate (skip connection) -> DoubleConv
    
    Upsamples the feature map and combines with corresponding encoder features
    via skip connections for better localization.
    
    Args:
        in_channels (int): Number of input channels (from lower layer)
        out_channels (int): Number of output channels
        bilinear (bool): Use bilinear upsampling (True) or transposed conv (False)
    
    Example:
        >>> decoder = DecoderBlock(256, 128)
        >>> x1 = torch.randn(1, 256, 32, 32)   # From lower layer
        >>> x2 = torch.randn(1, 128, 64, 64)   # Skip connection
        >>> output = decoder(x1, x2)
        >>> print(output.shape)  # torch.Size([1, 128, 64, 64])
    """
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(DecoderBlock, self).__init__()
        
        if bilinear:
            # Bilinear upsampling: faster, uses less memory
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # Transposed convolution: learnable upsampling
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 
                                          kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        """
        Forward pass with skip connection.
        
        Args:
            x1: Feature map from lower decoder layer (to be upsampled)
            x2: Feature map from encoder (skip connection)
        
        Returns:
            torch.Tensor: Concatenated and convolved feature map
        """
        # Upsample x1 to match x2's spatial dimensions
        x1 = self.up(x1)
        
        # Handle size mismatch (can occur with odd dimensions)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        
        # Pad x1 if necessary
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        
        # Concatenate along channel dimension (skip connection)
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)


class OutConv(nn.Module):
    """
    Output Convolution: 1x1 Conv to map features to class predictions.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels (= number of classes)
    
    Example:
        >>> out = OutConv(64, 10)
        >>> x = torch.randn(1, 64, 256, 256)
        >>> output = out(x)
        >>> print(output.shape)  # torch.Size([1, 10, 256, 256])
    """
    
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        """Forward pass through 1x1 convolution."""
        return self.conv(x)


# ============================================================================
# U-NET MODEL
# ============================================================================

class UNet(nn.Module):
    """
    U-Net Architecture for Semantic Segmentation.
    
    The architecture consists of:
    1. ENCODER (Contracting Path): Captures context through convolutions and pooling
       - 4 encoder blocks, each doubles the number of channels
       - Max pooling reduces spatial dimensions by 2 at each level
    
    2. BOTTLENECK: The deepest layer connecting encoder and decoder
       - Processes the most compressed representation
    
    3. DECODER (Expanding Path): Enables precise localization through upsampling
       - 4 decoder blocks, each halves the number of channels
       - Upsampling doubles spatial dimensions at each level
    
    4. SKIP CONNECTIONS: Connect encoder layers to corresponding decoder layers
       - Concatenate encoder features with decoder features
       - Preserves fine-grained spatial information
    
    Architecture Diagram:
    
        Input (3, 256, 256)
              |
         DoubleConv (64)  ─────────────────────┐
              |                                 │ Skip 1
         Encoder1 (128)  ──────────────────┐   │
              |                             │   │ Skip 2
         Encoder2 (256)  ─────────────┐    │   │
              |                        │    │   │ Skip 3
         Encoder3 (512)  ────────┐    │    │   │
              |                   │    │    │   │ Skip 4
         Encoder4 (1024)         │    │    │   │
              |                   │    │    │   │
         Decoder1 (512)  ←───────┘    │    │   │
              |                        │    │   │
         Decoder2 (256)  ←────────────┘    │   │
              |                             │   │
         Decoder3 (128)  ←─────────────────┘   │
              |                                 │
         Decoder4 (64)   ←──────────────────────┘
              |
          OutConv (num_classes)
              |
        Output (num_classes, 256, 256)
    
    Args:
        in_channels (int): Number of input channels (3 for RGB images)
        num_classes (int): Number of segmentation classes
        base_features (int): Number of features in first layer (64 by default)
        bilinear (bool): Use bilinear upsampling (True) or transposed conv (False)
    
    Example:
        >>> model = UNet(in_channels=3, num_classes=10)
        >>> x = torch.randn(1, 3, 256, 256)
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([1, 10, 256, 256])
    """
    
    def __init__(self, in_channels=3, num_classes=2, base_features=64, bilinear=True):
        super(UNet, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        
        # Factor for channel reduction when using bilinear upsampling
        factor = 2 if bilinear else 1
        
        # ==================== ENCODER (Contracting Path) ====================
        # Initial double convolution (no downsampling)
        self.inc = DoubleConv(in_channels, base_features)  # 3 -> 64
        
        # Encoder blocks (each halves spatial dimensions, doubles channels)
        self.down1 = EncoderBlock(base_features, base_features * 2)       # 64 -> 128
        self.down2 = EncoderBlock(base_features * 2, base_features * 4)   # 128 -> 256
        self.down3 = EncoderBlock(base_features * 4, base_features * 8)   # 256 -> 512
        self.down4 = EncoderBlock(base_features * 8, base_features * 16 // factor)  # 512 -> 1024
        
        # ==================== DECODER (Expanding Path) ====================
        # Decoder blocks (each doubles spatial dimensions, halves channels)
        self.up1 = DecoderBlock(base_features * 16, base_features * 8 // factor, bilinear)
        self.up2 = DecoderBlock(base_features * 8, base_features * 4 // factor, bilinear)
        self.up3 = DecoderBlock(base_features * 4, base_features * 2 // factor, bilinear)
        self.up4 = DecoderBlock(base_features * 2, base_features, bilinear)
        
        # ==================== OUTPUT ====================
        # Final 1x1 convolution to get class logits
        self.outc = OutConv(base_features, num_classes)
    
    def forward(self, x):
        """
        Forward pass through U-Net.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (N, C, H, W)
        
        Returns:
            torch.Tensor: Output logits of shape (N, num_classes, H, W)
        """
        # ==================== ENCODER PATH ====================
        # Level 0: Initial convolution (no downsampling)
        x1 = self.inc(x)      # (N, 64, H, W)
        
        # Level 1-4: Encoder blocks with downsampling
        x2 = self.down1(x1)   # (N, 128, H/2, W/2)
        x3 = self.down2(x2)   # (N, 256, H/4, W/4)
        x4 = self.down3(x3)   # (N, 512, H/8, W/8)
        x5 = self.down4(x4)   # (N, 1024, H/16, W/16) - Bottleneck
        
        # ==================== DECODER PATH ====================
        # Decoder blocks with skip connections
        x = self.up1(x5, x4)  # (N, 512, H/8, W/8)
        x = self.up2(x, x3)   # (N, 256, H/4, W/4)
        x = self.up3(x, x2)   # (N, 128, H/2, W/2)
        x = self.up4(x, x1)   # (N, 64, H, W)
        
        # ==================== OUTPUT ====================
        # Final 1x1 convolution to get class logits
        logits = self.outc(x)  # (N, num_classes, H, W)
        
        return logits
    
    def predict(self, x):
        """
        Get class predictions (argmax of logits).
        
        Args:
            x (torch.Tensor): Input image tensor of shape (N, C, H, W)
        
        Returns:
            torch.Tensor: Predicted class labels of shape (N, H, W)
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


# ============================================================================
# SMALLER U-NET VARIANT
# ============================================================================

class UNetSmall(nn.Module):
    """
    Smaller U-Net variant with fewer parameters.
    
    Useful for:
    - Limited GPU memory
    - Smaller datasets
    - Faster training/inference
    
    Has 3 encoder/decoder levels instead of 4, and starts with 32 features.
    
    Args:
        in_channels (int): Number of input channels (3 for RGB)
        num_classes (int): Number of segmentation classes
        base_features (int): Number of features in first layer (32 by default)
    """
    
    def __init__(self, in_channels=3, num_classes=2, base_features=32):
        super(UNetSmall, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Encoder
        self.inc = DoubleConv(in_channels, base_features)          # 3 -> 32
        self.down1 = EncoderBlock(base_features, base_features * 2)       # 32 -> 64
        self.down2 = EncoderBlock(base_features * 2, base_features * 4)   # 64 -> 128
        self.down3 = EncoderBlock(base_features * 4, base_features * 8)   # 128 -> 256
        
        # Decoder
        self.up1 = DecoderBlock(base_features * 8, base_features * 4, bilinear=True)
        self.up2 = DecoderBlock(base_features * 4, base_features * 2, bilinear=True)
        self.up3 = DecoderBlock(base_features * 2, base_features, bilinear=True)
        
        # Output
        self.outc = OutConv(base_features, num_classes)
    
    def forward(self, x):
        """Forward pass through smaller U-Net."""
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Decoder with skip connections
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        # Output
        return self.outc(x)
    
    def predict(self, x):
        """Get class predictions."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


# ============================================================================
# MODEL FACTORY FUNCTION
# ============================================================================

def get_model(in_channels=3, num_classes=2, model_type='unet', device='cpu'):
    """
    Factory function to create and initialize a segmentation model.
    
    Args:
        in_channels (int): Number of input channels (3 for RGB)
        num_classes (int): Number of output classes
        model_type (str): Type of model ('unet' or 'unet_small')
        device (str or torch.device): Device to place the model on
    
    Returns:
        nn.Module: Initialized model on specified device
    
    Example:
        >>> model = get_model(in_channels=3, num_classes=10, 
        ...                   model_type='unet', device='cuda')
    """
    # Create model based on type
    if model_type == 'unet':
        model = UNet(in_channels=in_channels, num_classes=num_classes)
    elif model_type == 'unet_small':
        model = UNetSmall(in_channels=in_channels, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'unet' or 'unet_small'.")
    
    # Move model to specified device
    model = model.to(device)
    
    # Calculate and print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*60}")
    print(f"MODEL: {model_type.upper()}")
    print(f"{'='*60}")
    print(f"  Input channels:      {in_channels}")
    print(f"  Output classes:      {num_classes}")
    print(f"  Total parameters:    {total_params:,}")
    print(f"  Trainable params:    {trainable_params:,}")
    print(f"  Model size:          {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    print(f"  Device:              {device}")
    print(f"{'='*60}\n")
    
    return model


# ============================================================================
# MAIN - TEST THE MODEL
# ============================================================================

if __name__ == "__main__":
    """Test the U-Net model functionality."""
    
    print("\n" + "=" * 60)
    print("TESTING U-NET MODEL")
    print("=" * 60)
    
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create model
    model = get_model(in_channels=3, num_classes=10, model_type='unet', device=device)
    
    # Test forward pass
    batch_size = 2
    height, width = 256, 256
    
    # Create dummy input
    x = torch.randn(batch_size, 3, height, width).to(device)
    print(f"Input shape: {x.shape}")
    
    # Forward pass (with no gradient computation for testing)
    with torch.no_grad():
        output = model(x)
        predictions = model.predict(x)
    
    print(f"Output shape (logits): {output.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Predictions dtype: {predictions.dtype}")
    print(f"Unique predictions: {torch.unique(predictions).tolist()[:10]}")
    
    # Test small model
    print("\n" + "=" * 60)
    print("Testing UNet Small...")
    model_small = get_model(in_channels=3, num_classes=10, model_type='unet_small', device=device)
    
    with torch.no_grad():
        output_small = model_small(x)
    
    print(f"Output shape (small): {output_small.shape}")
    
    # GPU memory usage (if CUDA)
    if device.type == 'cuda':
        print(f"\nGPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024 / 1024:.2f} MB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1024 / 1024:.2f} MB")
    
    print("\n" + "=" * 60)
    print("MODEL TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)

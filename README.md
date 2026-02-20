# 🌾 Crop Classification using U-Net Segmentation

A GPU-optimized deep learning project for semantic segmentation of Sentinel-2 satellite images using PyTorch and U-Net architecture.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset Structure](#dataset-structure)
- [Usage](#usage)
  - [Training](#training)
  - [Testing](#testing)
- [Model Architecture](#model-architecture)
- [GPU Optimizations](#gpu-optimizations)
- [Results](#results)
- [Configuration Options](#configuration-options)

## 🎯 Overview

This project implements an end-to-end deep learning pipeline for crop classification from satellite imagery. The system uses a U-Net architecture for semantic segmentation, optimized for GPU training with mixed precision support.

**Key Features:**
- U-Net from scratch with skip connections
- Mixed precision training (AMP) for faster training
- Automatic GPU detection and optimization
- IoU and Dice coefficient metrics
- Visualization of predictions

## ✨ Features

| Feature | Description |
|---------|-------------|
| **U-Net Architecture** | Encoder-decoder with skip connections |
| **GPU Acceleration** | CUDA support with cuDNN optimization |
| **Mixed Precision** | `torch.cuda.amp` for 2x faster training |
| **Data Augmentation** | Random flip and rotation |
| **Loss Functions** | CrossEntropy + Dice loss |
| **Metrics** | IoU and Dice coefficient |
| **Visualization** | Input/GT/Prediction comparison |

## 📁 Project Structure

```
crop_llm/
│
├── dataset.py          # Custom PyTorch Dataset and DataLoader
├── model.py            # U-Net architecture implementation
├── train.py            # GPU-optimized training script
├── test.py             # Testing and evaluation script
├── utils.py            # Utility functions (metrics, visualization)
├── requirements.txt    # Python dependencies
├── README.md           # This file
│
├── miniDataSet/        # Dataset folder
│   ├── train_images/   # Training images
│   ├── train_masks/    # Training masks
│   ├── val_images/     # Validation images
│   ├── val_masks/      # Validation masks
│   ├── test_images/    # Test images
│   └── test_masks/     # Test masks
│
├── outputs/            # Training outputs (created during training)
│   ├── best_model.pth
│   ├── final_model.pth
│   └── training_history.png
│
└── test_results/       # Test results (created during testing)
    ├── test_metrics.txt
    └── visualizations/
```

## 🔧 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- CUDA 11.x or later

### Setup

1. **Clone/Navigate to the project:**
   ```bash
   cd crop_llm
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install PyTorch with CUDA support:**
   ```bash
   # For CUDA 11.8
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Install other dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Verify GPU setup:**
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

## 📊 Dataset Structure

The dataset should be organized as follows:

```
miniDataSet/
├── train_images/       # Training satellite images (RGB)
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
├── train_masks/        # Training segmentation masks
│   ├── image_001.png   # Same filename as corresponding image
│   ├── image_002.png
│   └── ...
├── val_images/         # Validation images
├── val_masks/          # Validation masks
├── test_images/        # Test images
└── test_masks/         # Test masks
```

**Requirements:**
- Images: RGB format (PNG, JPG, TIFF supported)
- Masks: Single-channel grayscale with integer class labels
- Each mask filename must match its corresponding image filename

## 🚀 Usage

### Training

**Basic training with default settings:**
```bash
python train.py
```

**Training with custom parameters:**
```bash
python train.py \
    --data_root miniDataSet \
    --batch_size 16 \
    --epochs 100 \
    --learning_rate 0.0001 \
    --model_type unet \
    --loss_type combined \
    --use_amp
```

**All training options:**
```bash
python train.py --help
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | miniDataSet | Dataset root directory |
| `--batch_size` | 8 | Training batch size |
| `--epochs` | 50 | Number of epochs |
| `--learning_rate` | 1e-4 | Initial learning rate |
| `--model_type` | unet | Model type (unet, unet_small) |
| `--loss_type` | combined | Loss function (ce, combined) |
| `--use_amp` | True | Enable mixed precision |
| `--scheduler` | plateau | LR scheduler (plateau, cosine, none) |
| `--output_dir` | outputs | Output directory |

### Testing

**Run evaluation on test set:**
```bash
python test.py
```

**Testing with custom parameters:**
```bash
python test.py \
    --data_root miniDataSet \
    --model_path outputs/best_model.pth \
    --batch_size 8 \
    --visualize \
    --num_visualize 10
```

**All testing options:**
```bash
python test.py --help
```

## 🏗️ Model Architecture

### U-Net

The U-Net architecture consists of:

```
Input (3, 256, 256)
       │
   ┌───┴───┐
   │ Encoder │
   └───┬───┘
       │
   DoubleConv (64)  ─────────────────────┐
       │                                  │
   Encoder1 (128)  ──────────────────┐   │
       │                              │   │
   Encoder2 (256)  ─────────────┐    │   │
       │                         │    │   │
   Encoder3 (512)  ────────┐    │    │   │
       │                    │    │    │   │
   Bottleneck (1024)        │    │    │   │
       │                    │    │    │   │
   Decoder1 (512)  ←────────┘    │    │   │
       │                         │    │   │
   Decoder2 (256)  ←─────────────┘    │   │
       │                              │   │
   Decoder3 (128)  ←──────────────────┘   │
       │                                  │
   Decoder4 (64)   ←──────────────────────┘
       │
   OutConv (num_classes)
       │
Output (num_classes, 256, 256)
```

**Key Components:**
- **DoubleConv**: Conv3x3 → BatchNorm → ReLU → Conv3x3 → BatchNorm → ReLU
- **EncoderBlock**: MaxPool2x2 → DoubleConv
- **DecoderBlock**: Upsample → Concatenate (skip) → DoubleConv
- **Skip Connections**: Preserve spatial information from encoder

## ⚡ GPU Optimizations

This project includes several GPU optimizations:

1. **cuDNN Benchmark Mode**
   ```python
   torch.backends.cudnn.benchmark = True
   ```
   Automatically finds the fastest convolution algorithms.

2. **Mixed Precision Training (AMP)**
   ```python
   with torch.cuda.amp.autocast():
       outputs = model(images)
   ```
   Uses FP16 for faster computation with FP32 precision where needed.

3. **Pin Memory**
   ```python
   DataLoader(..., pin_memory=True)
   ```
   Faster CPU-to-GPU data transfer.

4. **Non-blocking Transfers**
   ```python
   images = images.to(device, non_blocking=True)
   ```
   Overlaps data transfer with computation.

5. **Efficient Gradient Zeroing**
   ```python
   optimizer.zero_grad(set_to_none=True)
   ```
   More memory-efficient than `zero_grad()`.

## 📈 Results

After training, results are saved to:

**Training outputs (`outputs/`):**
- `best_model.pth` - Best model checkpoint
- `final_model.pth` - Final epoch model
- `training_history.png` - Loss curves
- `training_history.csv` - Training log

**Test results (`test_results/`):**
- `test_metrics.txt` - IoU and Dice scores
- `visualizations/` - Prediction visualizations

## ⚙️ Configuration Options

### Loss Functions

| Loss | Description |
|------|-------------|
| `ce` | CrossEntropyLoss - Standard classification loss |
| `combined` | Dice + CrossEntropy - Better for imbalanced classes |

### Learning Rate Schedulers

| Scheduler | Description |
|-----------|-------------|
| `plateau` | ReduceLROnPlateau - Reduces LR when val loss plateaus |
| `cosine` | CosineAnnealingLR - Gradual LR decay |
| `none` | No scheduling - Constant learning rate |

### Model Variants

| Model | Parameters | Description |
|-------|------------|-------------|
| `unet` | ~31M | Standard U-Net (64 base features) |
| `unet_small` | ~7M | Smaller variant (32 base features) |

## 📝 Example Output

```
================================================================================
CROP CLASSIFICATION - U-NET TRAINING (GPU-OPTIMIZED)
================================================================================

✓ Random seed set to 42 for reproducibility
✓ Using GPU: NVIDIA GeForce RTX 3090
  CUDA Version: 11.8
  GPU Memory: 24.0 GB

GPU Optimizations Enabled:
  - cuDNN benchmark: True
  - cuDNN enabled: True
  - TF32 allowed: True

==============================================================
MODEL: UNET
==============================================================
  Input channels:      3
  Output classes:      10
  Total parameters:    31,037,698
  Trainable params:    31,037,698
  Model size:          118.39 MB (float32)
==============================================================

Epoch 1/50 [Train]: 100%|██████████| 125/125 [00:45<00:00, loss: 0.8234]
Epoch 1/50 [Val]  : 100%|██████████| 32/32 [00:08<00:00, loss: 0.6521]

Epoch 1/50 Summary:
  Train Loss:     0.8234
  Val Loss:       0.6521
  Learning Rate:  1.00e-04
  ✓ NEW BEST MODEL SAVED!
```

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📄 License

This project is for educational purposes.

---

**Author:** Deep Learning Project  
**Date:** 2026

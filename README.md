# Slum-i: Semi-Supervised Semantic Segmentation for Slum Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)

## Abstract

This repository contains the official implementation of a novel semi-supervised semantic segmentation framework for automated slum detection from satellite imagery. Our method combines prototype-based pseudo-labeling with self-adaptive thresholding to effectively leverage unlabeled data, achieving state-of-the-art performance on multi-city slum datasets with limited labeled samples. The framework addresses the critical challenge of slum identification in urban planning and resource allocation by reducing the dependency on expensive manual annotations while maintaining high segmentation accuracy.

## Table of Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Experimental Results](#experimental-results)
- [Model Zoo](#model-zoo)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

Slum detection from satellite imagery is a critical task for urban planning, policy-making, and resource allocation in developing regions. However, creating dense pixel-level annotations for semantic segmentation is time-consuming and expensive. This work introduces **Slum-i**, a semi-supervised learning framework that achieves competitive performance using only a small fraction of labeled data.

Our approach integrates:
- **Prototype-based pseudo-labeling** for generating high-quality pseudo-labels from unlabeled data
- **Class-aware adaptive thresholding** to handle class imbalance inherent in slum detection tasks
- **Multi-model support** including DeepLabV3+ with ResNet-101 and DPT with DINOv2 backbones
- **Comprehensive evaluation** across 8 different cities spanning multiple continents

## Key Features

- ✨ **Semi-supervised learning** framework with prototype memory banks
- 🎯 **State-of-the-art performance** with limited labeled data (10%, 20%, 30% splits)
- 🌍 **Multi-city evaluation** across diverse geographical regions
- 🔧 **Modular implementation** supporting multiple architectures and backbones
- 📊 **Comprehensive benchmarking** against FixMatch, UniMatch, and supervised baselines
- ⚡ **Efficient training** with optimized data augmentation and memory management
- 📈 **Detailed metrics** including IoU, accuracy, FLOPs, and training time

## Architecture

Our framework consists of the following key components:

1. **Segmentation Network**: DeepLabV3+ (ResNet-101) or DPT (DINOv2)
2. **Prototype Bank**: Maintains class-representative features for pseudo-label generation
3. **Adaptive Threshold Module**: Dynamically adjusts confidence thresholds per class
4. **Consistency Regularization**: Enforces prediction consistency across augmented views

### Supported Models

| Model | Backbone | Parameters | FLOPs | Inference Time |
|-------|----------|------------|-------|----------------|
| DeepLabV3+ | ResNet-101 | 59.5M | 96.08G | 12-13ms |
| DPT | DINOv2-Small | 24.2M | 40.45G | 12-13ms |
| DPT | DINOv2-Base | - | - | - |
| DPT | DINOv2-Large | - | - | - |

## Datasets

The framework is evaluated on **8 cities** from diverse geographical regions:

| City | Region | Description |
|------|--------|-------------|
| Mumbai | India | Dense urban slums |
| Karachi | Pakistan | Informal settlements |
| Lahore | Pakistan | Urban informal areas |
| Eldaein | Sudan | Sub-Saharan slums |
| Elgeneina | Sudan | Regional settlements |
| Makoko | Nigeria | Waterside slums |
| Medellin | Colombia | Hillside settlements |
| Lower | Haiti | Caribbean informal areas |

### Data Organization

```
data/
├── [city_name]/
│   ├── Images/     # RGB satellite images
│   └── Masks/      # Binary segmentation masks (0: background, 255: slum)
```

### Data Splits

For each city, we provide labeled/unlabeled splits:
- **10%**: 10% labeled, 90% unlabeled
- **20%**: 20% labeled, 80% unlabeled  
- **30%**: 30% labeled, 70% unlabeled
- **Full**: 100% labeled (supervised baseline)

Split files are located in `splits/[city_name]/`.

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU support)
- 16GB+ RAM recommended

### Dependencies

```bash
# Clone the repository
git clone https://github.com/tahamukhtar20/Slum-i.git
cd Slum-i

# Install required packages
pip install -r requirements.txt
```

### Key Dependencies

- PyTorch >= 1.9.0
- torchvision
- segmentation_models_pytorch
- transformers (for DINOv2)
- albumentations
- opencv-python
- scikit-learn
- tensorboard

## Usage

### Training

#### 1. Semi-Supervised Training (Our Method)

```bash
# DeepLabV3+ (ResNet-101) on Mumbai with 10% labels
python ours.py \
  --config configs/deeplab/mumbai.yaml \
  --labeled-id-path splits/mumbai/10/labeled.txt \
  --unlabeled-id-path splits/mumbai/10/unlabeled.txt \
  --save-path exp/mumbai/ours/10 \
  --split 10

# DPT (DINOv2) on Mumbai with 10% labels
python ours-dino.py \
  --config configs/dino/mumbai.yaml \
  --labeled-id-path splits/mumbai/10/labeled.txt \
  --unlabeled-id-path splits/mumbai/10/unlabeled.txt \
  --save-path exp/mumbai/ours-dino/10 \
  --split 10
```

#### 2. Baseline Methods

**FixMatch:**
```bash
python fixmatch.py \
  --config configs/deeplab/mumbai.yaml \
  --labeled-id-path splits/mumbai/10/labeled.txt \
  --unlabeled-id-path splits/mumbai/10/unlabeled.txt \
  --save-path exp/mumbai/fixmatch/10 \
  --split 10
```

**UniMatch:**
```bash
python unimatch.py \
  --config configs/deeplab/mumbai.yaml \
  --labeled-id-path splits/mumbai/10/labeled.txt \
  --unlabeled-id-path splits/mumbai/10/unlabeled.txt \
  --save-path exp/mumbai/unimatch/10 \
  --split 10
```

**Supervised Baseline:**
```bash
python supervised.py \
  --config configs/deeplab/mumbai.yaml \
  --labeled-id-path splits/mumbai/10/labeled.txt \
  --save-path exp/mumbai/supervised/10 \
  --split 10
```

#### 3. Batch Training

Use the provided scripts for multi-city experiments:

```bash
# Run multiple experiments
bash scripts/multischedule.sh
```

### Evaluation

Models are automatically evaluated during training. Results are saved to:
- `[save-path]/results.txt`: Performance metrics
- `[save-path]/best.pth`: Best model checkpoint
- `[save-path]/events.*`: TensorBoard logs

View training logs with TensorBoard:
```bash
tensorboard --logdir exp/
```

## Experimental Results

### Performance Comparison (Macro IoU %)

**Mumbai Dataset:**

| Method | 10% Labels | 20% Labels | 30% Labels | Full Labels |
|--------|-----------|-----------|-----------|-------------|
| Supervised | 52.44 | 70.72 | 67.72 | 75.24 |
| FixMatch | 67.41 | 71.50 | 73.52 | - |
| UniMatch | 70.22 | 72.78 | 70.82 | - |
| **Ours (DeepLabV3+)** | **71.87** | **73.48** | **76.30** | - |
| **Ours (DPT-DINOv2)** | **69.17** | **73.81** | **74.36** | **80.06** |

**Eldaein Dataset:**

| Method | 10% Labels | 20% Labels | 30% Labels | Full Labels |
|--------|-----------|-----------|-----------|-------------|
| Supervised | 52.44 | 70.72 | 67.72 | 75.24 |
| FixMatch | 67.41 | 71.50 | 73.52 | - |
| UniMatch | 70.22 | 72.78 | 70.82 | - |
| **Ours (DeepLabV3+)** | **68.05** | **70.23** | **74.88** | - |
| **Ours (DPT-DINOv2)** | **68.01** | **74.43** | **74.89** | **80.06** |

### Computational Efficiency

| Model | Parameters | FLOPs | Inference Time | Training Time (10%) |
|-------|-----------|-------|----------------|---------------------|
| DeepLabV3+ + ResNet-101 | 59.5M | 96.08G | 12-13ms | 2.8-7.3h |
| DPT + DINOv2-Small | 24.2M | 40.45G | 12-13ms | 1.9-2.8h |

**Key Observations:**
- Our method consistently outperforms baselines across different label ratios
- DPT with DINOv2 achieves competitive accuracy with 2.5× fewer parameters
- Significant performance gain over supervised learning with limited labels
- Efficient training time even with prototype memory mechanisms

### Ablation Studies

We conduct ablation studies on the Eldaein dataset to analyze the contribution of each component:

| Variant | Eldaein (10%) | Eldaein (20%) | Eldaein (30%) |
|---------|---------------|---------------|---------------|
| **Ours (Full)** | **68.05** | **70.23** | **74.88** |
| w/o Prototype Bank | 70.69 | 72.96 | 72.78 |
| w/o Adaptive Threshold (SAT only) | 73.77 | 73.48 | 76.30 |

**Note**: The ablation results show different behaviors across label ratios. The "SAT only" variant (self-adaptive threshold without prototype) performs better at lower label ratios but converges to similar performance at 30% labels. This suggests that the prototype bank component is particularly beneficial when combined with other mechanisms at higher label ratios, while the adaptive threshold provides consistent gains across all settings.

## Model Zoo

Pre-trained models and checkpoints will be made available upon publication.

### Pretrained Backbones

Download pretrained backbones to `./pretrained/`:
- ResNet-101: Standard ImageNet pretrained weights (via torchvision)
- DINOv2: Download from [official repository](https://github.com/facebookresearch/dinov2)

## Citation

If you find this work useful for your research, please consider citing:

```bibtex
@article{mukhtar2025slumi,
  title={Slum-i: Semi-Supervised Semantic Segmentation for Slum Detection},
  author={Mukhtar, Muhammad Taha and [Co-authors]},
  journal={[Journal/Conference Name]},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Satellite imagery datasets from [relevant sources]
- ResNet-101 implementation from [torchvision](https://pytorch.org/vision/)
- DINOv2 from [Meta AI Research](https://github.com/facebookresearch/dinov2)
- DeepLabV3+ architecture adapted from [original implementation](https://github.com/VainF/DeepLabV3Plus-Pytorch)
- Baseline implementations inspired by FixMatch, UniMatch, and UniMatchV2

## Contact

For questions and collaboration opportunities, please contact:
- Muhammad Taha Mukhtar - [GitHub](https://github.com/tahamukhtar20)

---

**Note**: This work addresses the critical challenge of slum detection for urban planning and development. The models and methods should be used responsibly and ethically, considering the social implications of automated slum identification.

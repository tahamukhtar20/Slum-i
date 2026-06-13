# SLUM-i: Semi-supervised Slum Segmentation with Data Quality Benchmarking

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## Environment setup

### Variables

```bash
export SLUM_I_ROOT=/path/to/Slum-i
export DATA_ROOT="$SLUM_I_ROOT/data_pipeline/datasets"
```

### Container (DFKI Pegasus cluster)

```bash
# All training scripts use this container image:
nvcr.io/nvidia/pytorch:25.02-py3

# Install dependencies inside container:
pip install -r "$SLUM_I_ROOT/requirements.txt"
```

### Pretrained backbones

Download and place under `pretrained/`:

| Backbone | Source | Path |
|---|---|---|
| ResNet-101 | [Google Drive](https://drive.google.com/file/d/1Rx0legsMolCWENpfvE2jUScT3ogalMO8/view) | `pretrained/resnet101.pth` |
| DINOv2-Small | [FAIR](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth) | `pretrained/dinov2_small.pth` |

## Dataset

### Cities

| City | Region | Source |
|---|---|---|
| Lahore | Pakistan | Original annotation (this work); boundary in `data_pipeline/kmls/lahore.kml` |
| Karachi | Pakistan | Derived from Karachi Cartography 2024 |
| Mumbai | India | Derived from Slum Rehabilitation Authority boundaries (2015) |
| El Daein | Sudan | [Gram-Hansen et al. 2019](https://drive.google.com/drive/u/0/folders/1yhDwR4zyPQO78x040uGCPqFarTDQ3yQm) |
| El Geneina | Sudan | [Gram-Hansen et al. 2019](https://drive.google.com/drive/u/0/folders/1yhDwR4zyPQO78x040uGCPqFarTDQ3yQm) |
| N. Nairobi (Lower) | Kenya | [Gram-Hansen et al. 2019](https://drive.google.com/drive/u/0/folders/1yhDwR4zyPQO78x040uGCPqFarTDQ3yQm) |
| Medellín | Colombia | [Gram-Hansen et al. 2019](https://drive.google.com/drive/u/0/folders/1yhDwR4zyPQO78x040uGCPqFarTDQ3yQm) |

> **Images and masks are not distributed.** Only dataset split files (`.txt`) are included. Obtain the original imagery from the sources above or contact the authors. Datasets can be reproduced (Image, Mask pairs) using the scripts present in `data_pipeline/` folder.

### Split files

```
data_pipeline/datasets/{city}/splits/
    train.txt         # all training tiles
    val.txt           # validation tiles
    test.txt          # test tiles
    {10,20,30}/
        labeled.txt   # labeled subset for that label budget
        unlabeled.txt # unlabeled pool
```

## Training

### Single run

```bash
python ours.py \
  --config "$SLUM_I_ROOT/configs/deeplab/benchmark/lahore.yaml" \
  --labeled-id-path   "$DATA_ROOT/lahore/splits/10/labeled.txt" \
  --unlabeled-id-path "$DATA_ROOT/lahore/splits/10/unlabeled.txt" \
  --save-path         "$SLUM_I_ROOT/exp/lahore/ours/10/seed42" \
  --split 10 --seed 42
```

Pass `--no-wandb` to disable W&B logging. Set `WANDB_API_KEY` in your environment to enable it.

### Full sweep (SLURM array)

```bash
# Dry run — show what would be submitted
python scripts/sweep.py --methods ours unimatch fixmatch supervised --dry-run

# Submit (caps at 16 concurrent jobs)
python scripts/sweep.py --methods ours unimatch --datasets lahore karachi mumbai \
  --splits 10 20 30 --concurrency 16
```

### Smoke test

```bash
bash scripts/smoke_test.sh   # 2-epoch sanity check on lower/split10
```

## DINO filter (ablation component)

```bash
# 1. Extract DINOv2 embeddings for all cities (requires GPU)
sbatch scripts/extract_embeddings.sh

# 2. Generate filtered unlabeled splits (keep top-80% by cosine similarity)
python scripts/filter_unlabeled.py --all --keep 0.8
```

## Reproducing figures

```bash
# Data quality figure (Fig. 1)
bash analysis/run.sh  # or: python analysis/make_figure.py

# Ablation heatmap
python scripts/plot_ablation_heatmap.py

# Qualitative segmentation figure
# Step 1: run inference (requires GPU + checkpoints)
python scripts/run_qualitative_inference.py
# Step 2: plot
python scripts/plot_qualitative_figure.py
```

## Results

All per-seed results are in `tables/results.csv`.


Columns in `tables/results.csv`: `method`, `city`, `split`, `seed`, `macro_iou`, `bg_iou`, `fg_iou`.

## Citation

```bibtex
@misc{mukhtar2026slumisemisupervisedlearningurban,
      title={SLUM-i: Semi-supervised Learning for Urban Mapping of Informal Settlements and Data Quality Benchmarking}, 
      author={Muhammad Taha Mukhtar and Syed Musa Ali Kazmi and Khola Naseem and Muhammad Ali Chattha and Andreas Dengel and Sheraz Ahmed and Muhammad Naseer Bajwa and Muhammad Imran Malik},
      year={2026},
      eprint={2602.04525},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.04525}, 
}
```

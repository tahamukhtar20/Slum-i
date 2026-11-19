# SLUM-i

Code, configurations, spatial splits and per-seed results for the paper "SLUM-i: Semi-Supervised
Learning for Urban Mapping of Informal Settlements" ([arXiv:2602.04525](https://arxiv.org/abs/2602.04525)).

Semi-supervised segmentation methods such as FixMatch, UniMatch and UniMatch V2 keep a pseudo-labelled
pixel only if its confidence exceeds a fixed threshold. Slums are the minority class and the model is
less confident on them, so the fixed threshold discards slum pixels far more often than background
pixels. We keep the self-adaptive threshold of FreeMatch but compute its class term from the mean
confidence of the pixels predicted as each class. It adds no hyperparameter and plugs into all three
methods. The paper evaluates it on Lahore, Karachi and Mumbai, with Esri World Imagery (0.5 m) and
Sentinel-2 (10 m), at 10, 20 and 30% labels.

The code of earlier versions of the paper is under the tag
[`v1.0`](https://github.com/tahamukhtar20/Slum-i/tree/v1.0).

## Contents

```
supervised.py, supervised-dino.py   supervised baselines (DeepLabV3+ ResNet-101, DPT DINOv2-S)
fixmatch_db.py, unimatch_db.py,     FixMatch, UniMatch and UniMatch V2, each with the pseudo-label
unimatchv2_db.py                    schemes of the paper selected in the config
util/debias.py                      the thresholds and the plug-in schemes (SelfAdaptiveThreshold is ours)
model/, dataset/, util/             networks, data loading, helpers (from UniMatch and UniMatch V2)
configs/deeplab/, configs/dino/     one config per city, sensor and method
scripts/runs/table2.txt, table3.txt every run of Tables II and III, one per line
scripts/train.sh                    runs one line of a run list, locally or as a Slurm array task
scripts/data/                       rebuilds the datasets from the polygons, mapcv and Earth Search
data/lahore.kml                     the 266 Lahore settlement polygons (new annotation)
data/mapcv/<city>/                  mapcv configs; Lahore also has its label and region GeoJSON
data/tiles/<dataset>/splits/        train, val and test lists and the 10/20/30% labelled subsets
data/s2_scenes.csv                  the Sentinel-2 L2A scenes used per city
results/runs.csv                    test scores of every run
analysis/                           scripts that compute the tables and statistics from results/
RESULTS.md                          which script and file give each number in the paper
```

## Setup

All runs used the NVIDIA PyTorch 25.02 container (`nvcr.io/nvidia/pytorch:25.02-py3`) on one GPU.
Elsewhere, install PyTorch and torchvision first, then

```bash
pip install -r requirements.txt
```

Download the pretrained backbones into `pretrained/`:

| Backbone | Source | Path | MD5 |
|---|---|---|---|
| ResNet-101 | [UniMatch (Google Drive)](https://drive.google.com/file/d/1Rx0legsMolCWENpfvE2jUScT3ogalMO8/view) | `pretrained/resnet101.pth` | `5dbecba3b82523cda5fd26171b267c49` |
| DINOv2-S | [Meta](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth) | `pretrained/dinov2_small.pth` | `2e405cee1bad14912278296d4f42e993` |

## Data

Esri World Imagery and the Karachi map may not be redistributed, so the repository holds what is needed
to rebuild the datasets instead of the images and masks.

| City | Labels | In this repository |
|---|---|---|
| Lahore | new annotation of the 266 settlements in the records of the Katchi Abadis Directorate | polygons, mapcv configs, region files, splits |
| Karachi | [Katchi Abadi map](https://www.karachicartography.org/katchi-abadi-map) of Karachi Cartography (529 polygons; the four over 500 ha are ignored) | mapcv configs, splits |
| Mumbai | [Slum cluster map](https://data.opencity.in/dataset/mumbai-slum-cluster-map) of the Slum Rehabilitation Authority (2,541 polygons) | mapcv configs, splits |

The datasets are built with [mapcv](https://github.com/tahamukhtar20/mapcv) 0.3.0. Esri tiles are
512 x 512 RGB at zoom 18 within 250 m of the polygons; the splits are spatial, by blocks of
2048 x 2048 pixels. Each Sentinel-2 sample is a 64 x 64 window of ten bands centred on an Esri tile,
so the two sensors share tiles, splits and labelled subsets. To rebuild a city:

```bash
pip install -r requirements-data.txt
bash scripts/data/build_city.sh lahore
bash scripts/data/build_city.sh karachi /path/to/karachi.kml
bash scripts/data/build_city.sh mumbai /path/to/mumbai.kml
```

The script downloads the imagery, writes `data/tiles/<dataset>/{Images,Masks}` and checks that the
rebuilt split lists equal the shipped ones. Training reads these folders:

| Dataset folder | City | Sensor | Tiles |
|---|---|---|---|
| `lahore_esri_z18`, `lahore_s2` | Lahore | Esri, Sentinel-2 | 2,006 |
| `karachi_esri_z18_f`, `karachi_s2` | Karachi | Esri, Sentinel-2 | 792 (of 1,850 before dropping tiles more than half unlabelled) |
| `mumbai_esri_z18`, `mumbai_s2` | Mumbai | Esri, Sentinel-2 | 3,559 |

## Training

Each line of `scripts/runs/table2.txt` (780 runs) and `scripts/runs/table3.txt` (90 runs) is one run:
script, config, labelled list, output folder, label budget, seed and unlabelled list. From the repo root:

```bash
bash scripts/train.sh scripts/runs/table2.txt 0
```

```bash
sbatch --array=0-779 --gpus=1 --mem=64G --time=24:00:00 scripts/train.sh scripts/runs/table2.txt
```

Each run writes its checkpoints and a `results.json` with the test scores to `runs/...`. The config
keys that select a pseudo-label scheme are:

| Scheme | Config keys |
|---|---|
| fixed threshold (FixMatch, UniMatch, UniMatch V2) | `conf_thresh: 0.95` |
| ours | `pl_scheme: freematch`, `sat_local: conf` |
| FreeMatch (without its fairness loss) | `pl_scheme: freematch` |
| SoftMatch | `pl_scheme: softmatch` |
| CDMAD | `cdmad: True`, `conf_thresh: 0.0` |
| DWL ranking | `rank_weight: True`, `rank_floor: 0.3` |
| ours, global term only | `pl_scheme: freematch`, `sat_local: none` |
| ours, fixed class thresholds | `pl_scheme: fixed_class`, `class_thresh: [0.87, 0.74]` |

## Results

`results/runs.csv` holds the test mIoU, slum IoU, background IoU and accuracy of every run, after the
last epoch. The scripts in `analysis/` turn it into the tables and statistics of the paper; see
[RESULTS.md](RESULTS.md). To score your own runs:

```bash
python analysis/collect_results.py
```

## Citation

```bibtex
@article{mukhtar2026slumi,
  title   = {{SLUM-i}: Semi-Supervised Learning for Urban Mapping of Informal Settlements},
  author  = {Mukhtar, Muhammad Taha and Kazmi, Syed Musa Ali and Naseem, Khola and Chattha, Muhammad Ali
             and Dengel, Andreas and Ahmed, Sheraz and Bajwa, Muhammad Naseer and Malik, Muhammad Imran},
  journal = {arXiv preprint arXiv:2602.04525},
  year    = {2026}
}
```

## License

MIT, see [LICENSE](LICENSE). The model code builds on [UniMatch](https://github.com/LiheYoung/UniMatch)
and [UniMatch V2](https://github.com/LiheYoung/UniMatch-V2) (MIT) and DINOv2 (Apache 2.0). The Lahore
polygons are released with this repository; the Karachi and Mumbai maps and the imagery keep the terms
of their providers.

# Where each number in the paper comes from

Run every script from the repo root. All of them read `results/runs.csv` or `results/all.json`, which
`analysis/collect_results.py` writes from the `results.json` of each run. Scores are those of the model
after the last epoch, without checkpoint selection.

| Paper | Script | Input | Output |
|---|---|---|---|
| Table I, tile counts | line counts of the split lists | `data/tiles/{lahore_esri_z18,karachi_esri_z18_f,mumbai_esri_z18}/splits/` | tiles per split and labelled subset |
| Table I, areas and slum pixel shares | mapcv's `patches.geojson` of the rebuilt tiles | `data/tiles/<dataset>/` after `scripts/data/build_city.sh` | per-tile class pixel counts |
| Table II (all results) | `analysis/table2.py` | `results/all.json` | `results/table2.tex`; bold and underline per backbone block |
| Table III (plug-in schemes and ablations) | `analysis/table3.py` | `results/runs.csv` | printed table |
| Abstract and Results: 49 of 54 settings, mean and median change, per sensor, method, budget and city, per-setting t-tests | `analysis/significance.py` | `results/all.json` | printed summary |
| Abstract and Results: 95% intervals (hierarchical bootstrap over datasets, settings and seeds), with and without FixMatch | `analysis/hboot.py` | `results/all.json` | printed summary |
| Results: best slum IoU in 33 of 36 backbone blocks | `analysis/table2.py` | `results/all.json` | "block wins" line (slum: Esri 17 of 18, Sentinel-2 16 of 18) |
| Results: 13 settings above the all-label supervised model against 5, the five slum IoU drops, the ten mIoU drops | `analysis/results_counts.py` | `results/all.json` | printed lists |

## Table II

`results/all.json` maps `<sensor>|<city>|<method>|<budget>|<seed>` to `[mIoU, slum IoU]` for the 780 runs
of `scripts/runs/table2.txt`: 3 cities, 2 sensors, 5 seeds, the two supervised baselines at 10, 20, 30
and 100% labels, and FixMatch, UniMatch and UniMatch V2 with and without our threshold at 10, 20 and
30%.

| Method folder | Paper row |
|---|---|
| `sup_r101` | Supervised (R101) |
| `fixmatch`, `fixmatch_freematchc` | FixMatch, Ours (FixMatch) |
| `unimatch`, `unimatch_freematchc` | UniMatch, Ours (UniMatch) |
| `sup_dino` | Supervised (DINOv2) |
| `unimatchv2`, `unimatchv2_freematchc` | UniMatch V2, Ours (UniMatch V2) |

## Table III

Esri imagery, 10% labels, UniMatch V2. Each row is paired by seed with the UniMatch V2 runs of Table II.

| Method folder | Paper row | Config |
|---|---|---|
| `unimatchv2_rank` | DWL ranking | `configs/dino/<city>_ft_rank.yaml` |
| `freematch` | FreeMatch | `configs/dino/<city>_ft_freematch.yaml` |
| `softmatch` | SoftMatch | `configs/dino/<city>_ft_softmatch.yaml` |
| `cdmad` | CDMAD | `configs/dino/<city>_ft_cdmad.yaml` |
| `unimatchv2_freematchc` | Ours (from Table II) | `configs/dino/<city>_ft_freematchc.yaml` |
| `satglobal` | Ours, global term only | `configs/dino/<city>_ft_satglobal.yaml` |
| `fixedcls` | Ours, fixed class thresholds | `configs/dino/<city>_ft_fixedcls.yaml` |

The paper reports all five seeds (0, 42, 123, 999, 1337), the default. `--seeds` selects a subset.

## Not reproduced by a script here

Fig. 1 (splits map) and Fig. 3 (qualitative results) need the rebuilt tiles and, for Fig. 3, the trained
checkpoints. Fig. 2 is a schematic of the threshold.

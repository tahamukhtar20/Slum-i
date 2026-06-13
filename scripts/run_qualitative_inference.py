"""
Run inference on test sets for qualitative figure.

For each city at split=10, picks the best seed (by test macro_iou) for each
of four methods: supervised, unimatch, fixmatch, ours.

Saves to $SLUM_I_ROOT/scratch/qualitative/{city}/:
  - input/{tile_id}       — original RGB PNG
  - gt/{tile_id}          — GT mask PNG (slum=255, bg=0)
  - {method}/{tile_id}    — predicted mask PNG (slum=255, bg=0)
  - tile_scores.csv       — per-tile IoU scores for all methods

Memory-efficient: writes each tile to disk immediately during inference;
never holds more than one batch in RAM at a time.

Usage (on cluster):
  srun --partition=RTXA6000 --mem=64G --cpus-per-task=4 --gres=gpu:1 \\
    --container-image=/enroot/nvcr.io_nvidia_pytorch_25.02-py3.sqsh \\
    --container-mounts=/netscratch:/netscratch \\
    --container-workdir=/netscratch/mukhtar/Slum-i \\
    python3 scripts/run_qualitative_inference.py
"""

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(os.environ.get("SLUM_I_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))

if "DATA_ROOT" not in os.environ:
    os.environ["DATA_ROOT"] = str(ROOT / "data_pipeline" / "datasets")

from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from util.utils import load_config

CITIES = ["eldaein", "elgeneina", "lower", "medellin", "mumbai", "lahore", "karachi"]
SPLIT = "10"
SEEDS = [0, 42, 123, 999, 1337]

METHODS = {
    "supervised": lambda city: ROOT / "archive" / "benchmark" / city / "supervised" / SPLIT,
    "unimatch":   lambda city: ROOT / "archive" / "benchmark" / city / "unimatch"   / SPLIT,
    "fixmatch":   lambda city: ROOT / "archive" / "benchmark" / city / "fixmatch"   / SPLIT,
    "ours":       lambda city: ROOT / "exp_dev"  / "gate8_full" / city / f"split{SPLIT}",
}

OUT_ROOT = ROOT / "scratch" / "qualitative"

IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def best_seed_path(base: Path):
    best_iou, best_dir = -1.0, None
    for seed in SEEDS:
        rj = base / f"seed{seed}" / "results.json"
        if not rj.exists():
            continue
        data = json.loads(rj.read_text())
        iou = data.get("best", {}).get("model", {}).get("macro_iou", -1)
        if iou > best_iou:
            best_iou = iou
            best_dir = base / f"seed{seed}"
    return best_dir


def load_model(cfg, ckpt_path: Path):
    model = DeepLabV3Plus(cfg).cuda()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model


def tile_fg_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = int(((pred == 1) & (gt == 1)).sum())
    union = int(((pred == 1) | (gt == 1)).sum())
    return inter / (union + 1e-10) * 100.0


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for city in CITIES:
        print(f"\n{'='*60}\n{city}\n{'='*60}", flush=True)
        city_out = OUT_ROOT / city

        cfg_path = ROOT / "configs" / "deeplab" / "benchmark" / f"{city}.yaml"
        cfg = load_config(str(cfg_path))
        data_root = str(cfg["data_root"]).replace("${DATA_ROOT}", os.environ["DATA_ROOT"])
        cfg["data_root"] = data_root

        testset = SemiDataset(cfg["dataset"], data_root, "test")
        loader  = DataLoader(testset, batch_size=1, pin_memory=False,
                             num_workers=2, drop_last=False)
        n_tiles = len(testset)
        print(f"  {n_tiles} test tiles", flush=True)

        # ── Collect tile ids, input images, GT masks (one pass, written immediately) ──
        (city_out / "input").mkdir(parents=True, exist_ok=True)
        (city_out / "gt").mkdir(parents=True, exist_ok=True)

        tile_ids = []
        slum_fracs = []

        # We need input/gt saved once; collect tile_ids in order
        input_written = (city_out / "input" / ".done").exists()
        if not input_written:
            for img_t, mask_t, tid in loader:
                fname = Path(tid[0]).name
                tile_ids.append(fname)
                gt = mask_t.squeeze(0).numpy().astype(np.uint8)
                slum_fracs.append(float(gt.mean()))

                # Save input
                img_np = img_t.squeeze(0).permute(1, 2, 0).numpy()
                img_np = (img_np * IMG_STD + IMG_MEAN).clip(0, 1)
                Image.fromarray((img_np * 255).astype(np.uint8)).save(
                    city_out / "input" / fname)

                # Save GT
                Image.fromarray((gt * 255).astype(np.uint8)).save(
                    city_out / "gt" / fname)

            (city_out / "input" / ".done").touch()
            # Save tile order for later methods
            (city_out / "tile_order.txt").write_text("\n".join(tile_ids))
            np.save(city_out / "slum_fracs.npy", np.array(slum_fracs))
            print(f"  input/gt written", flush=True)
        else:
            tile_ids = (city_out / "tile_order.txt").read_text().splitlines()
            slum_fracs = np.load(city_out / "slum_fracs.npy").tolist()
            print(f"  input/gt already present", flush=True)

        # ── Per-method inference: write predictions immediately, only keep IoUs ──
        method_ious = {tid: {} for tid in tile_ids}

        for method, base_fn in METHODS.items():
            out_dir = city_out / method
            if (out_dir / ".done").exists():
                print(f"  [{method}] already done, loading IoUs", flush=True)
                # Recompute IoUs from saved masks
                for fname in tile_ids:
                    pred = np.array(Image.open(out_dir / fname)) > 127
                    gt   = np.array(Image.open(city_out / "gt" / fname)) > 127
                    method_ious[fname][method] = round(tile_fg_iou(
                        pred.astype(np.uint8), gt.astype(np.uint8)), 2)
                continue

            ckpt_dir = best_seed_path(base_fn(city))
            if ckpt_dir is None:
                print(f"  [{method}] no checkpoints found, skipping", flush=True)
                continue
            print(f"  [{method}] ckpt={ckpt_dir}", flush=True)

            out_dir.mkdir(parents=True, exist_ok=True)
            model = load_model(cfg, ckpt_dir / "best.pth")

            with torch.no_grad():
                for img_t, mask_t, tid in loader:
                    fname = Path(tid[0]).name
                    pred_t = model(img_t.cuda()).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                    gt     = mask_t.squeeze(0).numpy().astype(np.uint8)

                    Image.fromarray((pred_t * 255).astype(np.uint8)).save(out_dir / fname)
                    method_ious[fname][method] = round(tile_fg_iou(pred_t, gt), 2)

            (out_dir / ".done").touch()
            del model
            torch.cuda.empty_cache()
            print(f"  [{method}] done", flush=True)

        # ── Write tile_scores.csv ──
        available_methods = [m for m in METHODS if (city_out / m / ".done").exists()]
        fieldnames = ["tile_id", "slum_frac"] + [f"{m}_fg_iou" for m in available_methods]
        csv_path = city_out / "tile_scores.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for fname, sf in zip(tile_ids, slum_fracs):
                row = {"tile_id": fname, "slum_frac": round(sf, 4)}
                for m in available_methods:
                    row[f"{m}_fg_iou"] = method_ious[fname].get(m, "")
                w.writerow(row)
        print(f"  tile_scores.csv written ({len(tile_ids)} tiles)", flush=True)

    print("\nDone.")


if __name__ == "__main__":
    main()

"""Compute per-tile data quality metrics for all 7 cities.

For each (image, mask) tile:
  - fractal dim of mask boundary       (boundary complexity)
  - fractal dim of image edges in mask (texture complexity)
  - complexity ratio = fd_boundary / fd_edges
  - separability (Fisher-like contrast between slum & non-slum gray values)
  - label displacement statistics (boundary pixel → nearest image edge)
  - area ratio, num components

Output: a single CSV with one row per tile, columns above + city.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

CITIES = [
    'lahore', 'karachi', 'mumbai',
    'eldaein', 'elgeneina', 'lower', 'medellin',
]

DATA_ROOT = Path(os.environ.get('DATA_ROOT', Path(__file__).resolve().parents[2] / 'data_pipeline' / 'datasets'))


def fractal_dim(binary_img: np.ndarray, n_box_sizes: int = 8) -> float:
    """Box-counting fractal dimension of a binary image. Returns 0 if too few pixels."""
    Z = binary_img > 0
    if Z.sum() < 20:
        return 0.0

    p = min(Z.shape)
    max_n = int(np.floor(np.log2(p)))
    sizes = 2 ** np.arange(max_n, max(0, max_n - n_box_sizes), -1)

    counts = []
    for s in sizes:
        s = int(s)
        H = (Z.shape[0] // s) * s
        W = (Z.shape[1] // s) * s
        if H == 0 or W == 0:
            continue
        trimmed = Z[:H, :W]
        reshaped = trimmed.reshape(H // s, s, W // s, s)
        block_any = reshaped.any(axis=(1, 3))
        counts.append(block_any.sum())

    if len(sizes) < 2 or len(counts) < 2:
        return 0.0
    coeffs = np.polyfit(np.log(sizes[:len(counts)]), np.log(np.maximum(counts, 1)), 1)
    return float(-coeffs[0])


def displacement_stats(mask: np.ndarray, image_gray: np.ndarray) -> dict:
    """Compute boundary pixel → nearest Canny edge distance distribution."""
    blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges_inv = 255 - edges
    dist = cv2.distanceTransform(edges_inv, cv2.DIST_L2, 5)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    boundary = np.zeros_like(mask)
    cv2.drawContours(boundary, contours, -1, 255, 1)

    devs = dist[boundary == 255]
    if devs.size == 0:
        return {
            'disp_n': 0, 'disp_mean': np.nan, 'disp_median': np.nan,
            'disp_p95': np.nan, 'disp_pct_le2': np.nan,
            'disp_pct_le5': np.nan, 'disp_pct_le15': np.nan,
        }

    return {
        'disp_n':       int(devs.size),
        'disp_mean':    float(devs.mean()),
        'disp_median':  float(np.median(devs)),
        'disp_p95':     float(np.percentile(devs, 95)),
        'disp_pct_le2': float((devs <= 2).mean()),
        'disp_pct_le5': float((devs <= 5).mean()),
        'disp_pct_le15':float((devs <= 15).mean()),
    }


def separability(image_gray: np.ndarray, mask: np.ndarray) -> float:
    """Fisher-like separability between in-mask and out-mask gray values."""
    inside = image_gray[mask > 0]
    outside = image_gray[mask == 0]
    if inside.size < 50 or outside.size < 50:
        return np.nan
    mu_i, mu_o = inside.mean(), outside.mean()
    var_i, var_o = inside.var(), outside.var()
    denom = var_i + var_o + 1e-6
    return float((mu_i - mu_o) ** 2 / denom)


def process_tile(args) -> dict | None:
    img_path, mask_path, city = args
    cv2.setNumThreads(0)

    image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        return None
    if mask.max() > 1:
        mask = (mask > 127).astype(np.uint8) * 255
    else:
        mask = (mask > 0).astype(np.uint8) * 255

    area_ratio = float((mask > 0).mean())
    if area_ratio == 0.0:
        # pure background tile — record area but skip geometry/displacement
        return {
            'city': city, 'file': img_path.name,
            'area_ratio': 0.0, 'n_components': 0,
            'fd_boundary': np.nan, 'fd_edges': np.nan,
            'complexity_ratio': np.nan, 'separability': np.nan,
            **displacement_stats(mask, image),
        }

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    boundary = np.zeros_like(mask)
    cv2.drawContours(boundary, contours, -1, 255, 1)

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)
    edges_in_mask = cv2.bitwise_and(edges, mask)

    fd_b = fractal_dim(boundary)
    fd_e = fractal_dim(edges_in_mask)
    cr = fd_b / fd_e if fd_e > 0.1 else np.nan
    sep = separability(image, mask)

    return {
        'city': city, 'file': img_path.name,
        'area_ratio': area_ratio,
        'n_components': len(contours),
        'fd_boundary': fd_b,
        'fd_edges': fd_e,
        'complexity_ratio': cr,
        'separability': sep,
        **displacement_stats(mask, image),
    }


def gather_tasks(data_root: Path):
    tasks = []
    for city in CITIES:
        city_dir = data_root / city
        img_dir = city_dir / 'Images'
        mask_dir = city_dir / 'Masks'
        if not (img_dir.exists() and mask_dir.exists()):
            print(f'[warn] missing dir for {city}')
            continue
        for img_path in img_dir.glob('*.png'):
            mask_path = mask_dir / img_path.name
            if mask_path.exists():
                tasks.append((img_path, mask_path, city))
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path('analysis/metrics.csv'))
    ap.add_argument('--workers', type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument('--sample-per-city', type=int, default=None,
                    help='Optionally subsample N tiles per city for fast iteration')
    args = ap.parse_args()

    tasks = gather_tasks(DATA_ROOT)
    if args.sample_per_city:
        from collections import defaultdict
        import random
        by_city = defaultdict(list)
        for t in tasks:
            by_city[t[2]].append(t)
        tasks = []
        for city, ts in by_city.items():
            random.seed(42)
            random.shuffle(ts)
            tasks.extend(ts[:args.sample_per_city])

    print(f'processing {len(tasks)} tiles with {args.workers} workers')

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_tile, t) for t in tasks]
        for f in tqdm(as_completed(futures), total=len(futures), ncols=80):
            r = f.result()
            if r:
                rows.append(r)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f'\nwrote {len(df)} rows → {args.out}')

    print('\nper-city counts:')
    print(df['city'].value_counts())


if __name__ == '__main__':
    main()

"""
Score unlabeled images by mean cosine similarity to labeled images,
then write filtered unlabeled split files.

Usage:
  python filter_unlabeled.py --city lower --split 10 --keep 0.8
  python filter_unlabeled.py --all --keep 0.8 0.7 0.6
"""
import argparse
import os
import numpy as np

import sys
from pathlib import Path

ROOT      = Path(os.environ.get('SLUM_I_ROOT', Path(__file__).resolve().parents[1]))
DATA_ROOT = ROOT / 'data_pipeline' / 'datasets'
EMB_DIR   = ROOT / 'exp_dev' / 'filter' / 'embeddings'
OUT_DIR   = ROOT / 'exp_dev' / 'filter' / 'splits'

CITIES = ['eldaein', 'elgeneina', 'karachi', 'lahore', 'lower', 'makoko', 'medellin', 'mumbai']
SPLITS = ['10', '20', '30']


def load_embeddings(city):
    npz = np.load(f'{EMB_DIR}/{city}_embeddings.npz')
    filenames  = list(npz['filenames'])
    embeddings = npz['embeddings']          # (N, 384), L2-normalised
    fn2idx = {fn: i for i, fn in enumerate(filenames)}
    return filenames, embeddings, fn2idx


def filter_city_split(city, split, keep_ratio):
    filenames, embeddings, fn2idx = load_embeddings(city)

    labeled_path   = f'{DATA_ROOT}/{city}/splits/{split}/labeled.txt'
    unlabeled_path = f'{DATA_ROOT}/{city}/splits/{split}/unlabeled.txt'

    with open(labeled_path)   as f: labeled_fns   = [l.strip() for l in f if l.strip()]
    with open(unlabeled_path) as f: unlabeled_fns = [l.strip() for l in f if l.strip()]

    # Labeled embedding matrix
    lab_idx  = [fn2idx[fn] for fn in labeled_fns if fn in fn2idx]
    lab_embs = embeddings[lab_idx]          # (n_lab, 384)

    # Score each unlabeled image: mean cosine sim to all labeled
    ulb_idx  = [fn2idx[fn] for fn in unlabeled_fns if fn in fn2idx]
    ulb_embs = embeddings[ulb_idx]          # (n_ulb, 384)

    # cosine sim = dot product (already L2-normalised)
    scores = ulb_embs @ lab_embs.T          # (n_ulb, n_lab)
    mean_scores = scores.mean(axis=1)       # (n_ulb,)

    # Keep top keep_ratio fraction
    n_keep = max(1, int(len(unlabeled_fns) * keep_ratio))
    top_idx = np.argsort(mean_scores)[::-1][:n_keep]
    kept_fns = [unlabeled_fns[i] for i in sorted(top_idx)]

    out_path = f'{OUT_DIR}/{city}/split{split}/keep{int(keep_ratio*100)}/unlabeled.txt'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('\n'.join(kept_fns) + '\n')

    print(f'[{city} split={split} keep={keep_ratio}] {len(unlabeled_fns)} -> {len(kept_fns)} unlabeled')
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city',  type=str, default=None)
    parser.add_argument('--split', type=str, default=None)
    parser.add_argument('--keep',  type=float, nargs='+', default=[0.9, 0.8, 0.7])
    parser.add_argument('--all',   action='store_true')
    args = parser.parse_args()

    cities = CITIES if args.all else [args.city]
    splits = SPLITS if args.all else [args.split]

    for city in cities:
        for split in splits:
            for keep in args.keep:
                filter_city_split(city, split, keep)

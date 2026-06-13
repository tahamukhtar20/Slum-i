#!/usr/bin/env python3
"""Generate and submit a Slum-i training sweep as a SLURM array job.

Creates one run script per (method, dataset, split, seed) combination,
then submits all as a single sbatch array with a concurrency cap.

Usage:
    # Dry run — show what would be submitted
    python scripts/sweep.py --methods slum-i fixmatch --dry-run

    # Submit with max 8 concurrent jobs
    python scripts/sweep.py --methods slum-i fixmatch unimatch supervised --concurrency 8

    # Only specific datasets / splits
    python scripts/sweep.py --methods slum-i --datasets lahore mumbai --splits 10 20
"""

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOT = Path(os.environ.get('SLUM_I_ROOT', Path(__file__).resolve().parents[1]))
DATA = ROOT / 'data_pipeline' / 'datasets'

DATASETS = ['lahore', 'karachi', 'mumbai', 'eldaein', 'elgeneina', 'lower', 'makoko', 'medellin']
SPLITS = ['10', '20', '30']
SEEDS = [42, 1337, 0]

# backbone tag → config subfolder and script suffix
METHODS = {
    'fixmatch':       {'config': 'deeplab/benchmark', 'semi': True},
    'unimatch':       {'config': 'deeplab/benchmark', 'semi': True},
    'ours':           {'config': 'deeplab/benchmark', 'semi': True},
    'supervised':     {'config': 'deeplab/benchmark', 'semi': False},
    'supervised-dino':{'config': 'dino/benchmark',    'semi': False},
    'unimatchv2':     {'config': 'dino/benchmark',    'semi': True},
    'ours-dino':      {'config': 'dino/benchmark',    'semi': True},
}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--methods',     nargs='+', default=list(METHODS), choices=list(METHODS))
parser.add_argument('--datasets',    nargs='+', default=DATASETS)
parser.add_argument('--splits',      nargs='+', default=SPLITS)
parser.add_argument('--seeds',       nargs='+', type=int, default=SEEDS)
parser.add_argument('--concurrency', type=int, default=16,
                    help='Max simultaneous array tasks')
parser.add_argument('--sweep-dir',   type=str, default=None,
                    help='Directory to write run scripts (auto-named if omitted)')
parser.add_argument('--exp-dir',     type=str, default='exp',
                    help='Base experiment directory under ROOT')
parser.add_argument('--extra-args',  nargs=argparse.REMAINDER, default=[],
                    help='Additional args appended to each training command; put this last')
parser.add_argument('--spw-mode',    type=str, default=None,
                    choices=['none', 'rank', 'prior', 'boundary', 'combined'],
                    help='Append --spw-mode to each training command')
parser.add_argument('--spw-rank-keep', type=float, default=None)
parser.add_argument('--spw-rank-min-conf', type=float, default=None)
parser.add_argument('--spw-prior-strength', type=float, default=None)
parser.add_argument('--spw-prior-clip', type=float, default=None)
parser.add_argument('--spw-boundary-gamma', type=float, default=None)
parser.add_argument('--spw-conf-warmup-epochs', type=int, default=None,
                    help='Forward --spw-conf-warmup-epochs to training script')
parser.add_argument('--spw-conf-warmup-start', type=float, default=None,
                    help='Forward --spw-conf-warmup-start to training script')
parser.add_argument('--no-wandb',    action='store_true',
                    help='Disable wandb logging (enabled by default)')
parser.add_argument('--wandb-tag',   type=str, default=None,
                    help='Optional tag appended to wandb run tags for grouping')
parser.add_argument('--dry-run',     action='store_true')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def data_path(dataset: str) -> Path:
    return DATA / dataset


def split_files(dataset: str, split: str, semi: bool):
    base = data_path(dataset) / 'splits'
    if semi:
        unlabeled = str(base / split / 'unlabeled.txt')
        labeled   = str(base / split / 'labeled.txt')
    elif split == 'full':
        labeled   = str(base / 'train.txt')
        unlabeled = ''
    else:
        labeled   = str(base / split / 'labeled.txt')
        unlabeled = ''
    return labeled, unlabeled


def make_run_script(method: str, dataset: str, split: str, seed: int,
                    exp_dir: str = 'exp', extra_args=None, no_wandb=False) -> str:
    info = METHODS[method]
    config = str(ROOT / 'configs' / info['config'] / f'{dataset}.yaml')
    labeled, unlabeled = split_files(dataset, split, info['semi'])
    split_tag = split
    save_path = str(ROOT / exp_dir / dataset / method / split_tag / f'seed{seed}')
    extra_args = list(extra_args) if extra_args else []
    if no_wandb:
        extra_args = ['--no-wandb'] + extra_args
    extra = ''.join(f' {shlex.quote(str(arg))}' for arg in extra_args)

    return (
        f'#!/bin/bash\n'
        f'# method={method} dataset={dataset} split={split_tag} seed={seed}\n'
        f'scripts/exec.sh '
        f'"{method}" "{config}" "{labeled}" "{unlabeled}" '
        f'"{save_path}" "{split_tag}" "{seed}"'
        f'{extra}\n'
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else (
        ROOT / 'sweeps' / timestamp
    )
    sweep_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for method in args.methods:
        info = METHODS[method]
        # SSL methods cannot use 'full' (no labeled/unlabeled split there).
        # Supervised methods can use any split the user specifies.
        if info['semi']:
            split_list = [s for s in args.splits if s != 'full']
        else:
            split_list = list(args.splits)
        for dataset in args.datasets:
            config_file = ROOT / 'configs' / info['config'] / f'{dataset}.yaml'
            if not config_file.exists():
                print(f'[skip] no config: {config_file}', file=sys.stderr)
                continue
            for split in split_list:
                for seed in args.seeds:
                    runs.append((method, dataset, split, seed))

    explicit_extra = []
    if args.spw_mode is not None:
        explicit_extra += ['--spw-mode', args.spw_mode]
    for name in ['spw_rank_keep', 'spw_rank_min_conf', 'spw_prior_strength', 'spw_prior_clip', 'spw_boundary_gamma', 'spw_conf_warmup_epochs', 'spw_conf_warmup_start']:
        value = getattr(args, name)
        if value is not None:
            explicit_extra += ['--' + name.replace('_', '-'), str(value)]
    extra_args = explicit_extra + list(args.extra_args)

    print(f'Sweep: {len(runs)} runs → {sweep_dir}')

    for i, (method, dataset, split, seed) in enumerate(runs):
        script = make_run_script(method, dataset, split, seed,
                                 exp_dir=args.exp_dir, extra_args=extra_args,
                                 no_wandb=args.no_wandb)
        run_file = sweep_dir / f'run_{i:04d}.sh'
        run_file.write_text(script)
        run_file.chmod(0o755)

    if args.dry_run:
        print('Dry run — not submitting.')
        print(f'First run:\n{(sweep_dir / "run_0000.sh").read_text()}')
        return

    n = len(runs)
    cmd = [
        'sbatch',
        f'--array=0-{n - 1}%{args.concurrency}',
        'scripts/array.sh',
        str(sweep_dir),
    ]
    print('Submitting:', ' '.join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Submit the full baseline benchmark sweep with seed-level dependency chaining.

Structure:
  - One SLURM array per (method, seed) — 5 methods × 5 seeds = 25 arrays total
  - All methods for seed N run in parallel, capped at --concurrency each (~80 total)
  - Seed N+1 only starts after ALL of seed N's arrays complete (afterok chaining)
  - Job name = slumi-{method}-s{seed} for targeted cancellation

Cancellation:
    scancel --name=slumi-unimatch-s42        # cancel unimatch seed 42
    scancel --name=slumi-unimatch-s1337      # cancel unimatch seed 1337
    scancel --name=slumi-fixmatch-s0         # cancel fixmatch seed 0

Rerun a specific city after failure:
    python scripts/rerun.py --sweep-dir sweeps/benchmark_XXXX/ \\
        --method unimatch --seed 42 --datasets lahore

Usage:
    python scripts/benchmark.py              # submit everything
    python scripts/benchmark.py --dry-run    # verify without submitting
    python scripts/benchmark.py --methods unimatch fixmatch --seeds 42
"""

import argparse
import os
import csv
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(os.environ.get('SLUM_I_ROOT', Path(__file__).resolve().parents[1]))
DATA = ROOT / 'data_pipeline' / 'datasets'

DATASETS = ['lahore', 'karachi', 'mumbai', 'eldaein', 'elgeneina', 'lower', 'makoko', 'medellin']
SEEDS    = [42, 1337, 0, 123, 999]

# method -> (script_stem, config_subdir, splits, is_ssl)
METHODS = {
    'supervised':      ('supervised',      'deeplab/benchmark', ['full', '10', '20', '30'], False),
    'supervised-dino': ('supervised-dino', 'dino/benchmark',    ['full', '10', '20', '30'], False),
    'fixmatch':        ('fixmatch',        'deeplab/benchmark', ['10', '20', '30'],          True),
    'unimatch':        ('unimatch',        'deeplab/benchmark', ['10', '20', '30'],          True),
    'unimatchv2':      ('unimatchv2',      'dino/benchmark',    ['10', '20', '30'],          True),
}


def labeled_path(dataset: str, split: str) -> str:
    if split == 'full':
        return str(DATA / dataset / 'splits' / 'train.txt')
    return str(DATA / dataset / 'splits' / split / 'labeled.txt')


def unlabeled_path(dataset: str, split: str) -> str:
    return str(DATA / dataset / 'splits' / split / 'unlabeled.txt')


def make_run_script(method: str, script: str, config_sub: str, dataset: str,
                    split: str, seed: int, is_ssl: bool, exp_dir: str) -> str:
    config    = str(ROOT / 'configs' / config_sub / f'{dataset}.yaml')
    labeled   = labeled_path(dataset, split)
    unlabeled = unlabeled_path(dataset, split) if is_ssl else ''
    save_path = str(ROOT / exp_dir / dataset / method / split / f'seed{seed}')
    return (
        f'#!/bin/bash\n'
        f'# method={method} dataset={dataset} split={split} seed={seed}\n'
        f'scripts/exec.sh '
        f'"{script}" "{config}" "{labeled}" "{unlabeled}" '
        f'"{save_path}" "{split}" "{seed}" '
        f'"--no-wandb"\n'
    )


def sbatch(cmd: list[str], dry_run: bool) -> str | None:
    """Run sbatch, return job ID string or None on dry-run/failure."""
    if dry_run:
        print(f'  [dry] {" ".join(cmd)}')
        return 'DRY'
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'  ERROR: {result.stderr.strip()}', file=sys.stderr)
        sys.exit(1)
    # "Submitted batch job 1234567"
    m = re.search(r'(\d+)', result.stdout)
    job_id = m.group(1) if m else result.stdout.strip()
    print(f'  submitted job {job_id}')
    return job_id


def submit_seed(method: str, seed: int, datasets: list[str], sweep_root: Path,
                exp_dir: str, concurrency: int, dep_job_ids: list[str],
                dry_run: bool) -> str | None:
    script, config_sub, splits, is_ssl = METHODS[method]

    runs = [
        (dataset, split)
        for dataset in datasets
        if (ROOT / 'configs' / config_sub / f'{dataset}.yaml').exists()
        for split in splits
    ]
    if not runs:
        return None

    method_seed_dir = sweep_root / method / f'seed{seed}'
    method_seed_dir.mkdir(parents=True, exist_ok=True)

    for i, (dataset, split) in enumerate(runs):
        content = make_run_script(method, script, config_sub, dataset, split, seed, is_ssl, exp_dir)
        run_file = method_seed_dir / f'run_{i:04d}.sh'
        run_file.write_text(content)
        run_file.chmod(0o755)

    # Write manifest for rerun.py
    with open(method_seed_dir / 'manifest.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['task_id', 'method', 'dataset', 'split', 'seed'])
        for i, (dataset, split) in enumerate(runs):
            w.writerow([i, method, dataset, split, seed])

    n = len(runs)
    job_name = f'slumi-{method}-s{seed}'

    cmd = [
        'sbatch',
        f'--array=0-{n - 1}%{concurrency}',
        f'--job-name={job_name}',
    ]
    if dep_job_ids:
        dep_str = ':'.join(dep_job_ids)
        cmd.append(f'--dependency=afterok:{dep_str}')
    cmd += ['scripts/array.sh', str(method_seed_dir)]

    print(f'  {method} seed={seed}: {n} runs, dep={dep_job_ids or "none"}')
    return sbatch(cmd, dry_run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods',     nargs='+', default=list(METHODS), choices=list(METHODS))
    ap.add_argument('--datasets',    nargs='+', default=DATASETS)
    ap.add_argument('--seeds',       nargs='+', type=int, default=SEEDS)
    ap.add_argument('--concurrency', type=int,  default=16,
                    help='Max simultaneous jobs per (method, seed) array (default: 16 → ~80 total)')
    ap.add_argument('--exp-dir',     type=str,  default='exp_benchmark')
    ap.add_argument('--sweep-dir',   type=str,  default=None)
    ap.add_argument('--dry-run',     action='store_true')
    args = ap.parse_args()

    timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
    sweep_root = Path(args.sweep_dir) if args.sweep_dir else (ROOT / 'sweeps' / f'benchmark_{timestamp}')
    sweep_root.mkdir(parents=True, exist_ok=True)

    print(f'Sweep root : {sweep_root}')
    print(f'Methods    : {args.methods}')
    print(f'Seeds      : {args.seeds}')
    print(f'Concurrency: {args.concurrency} per (method, seed) → '
          f'~{args.concurrency * len(args.methods)} max total')
    print()

    # prev_seed_ids: job IDs from the previous seed's arrays (all methods)
    prev_seed_ids: list[str] = []

    for seed in args.seeds:
        print(f'--- Seed {seed} ---')
        this_seed_ids: list[str] = []
        for method in args.methods:
            job_id = submit_seed(
                method, seed, args.datasets, sweep_root,
                args.exp_dir, args.concurrency,
                dep_job_ids=prev_seed_ids,
                dry_run=args.dry_run,
            )
            if job_id:
                this_seed_ids.append(job_id)
        prev_seed_ids = this_seed_ids
        print()

    print(f'All submitted. Sweep dir: {sweep_root}')
    print('Monitor  : squeue -u mukhtar')
    print('Cancel method+seed: scancel --name=slumi-<method>-s<seed>')
    print(f'Rerun city: python scripts/rerun.py --sweep-dir {sweep_root} --method <m> --seed <s> --datasets <city>')


if __name__ == '__main__':
    main()

"""
Collect filter_only (exp_dev/ablation) and caat (exp_dev/gate8_full)
per-seed results and append them to results.csv.

Output: results.csv updated with method='filter_only' and method='caat'
        (existing filter_only/caat rows are replaced)

Usage:
    python3 scripts/collect_ablation_results.py
"""

import json
import glob
import csv
import os
from pathlib import Path

ROOT = Path(os.environ.get('SLUM_I_ROOT', Path(__file__).resolve().parents[1]))

sources = {
    'filter_only': ROOT / 'exp_dev' / 'ablation',
    'caat':        ROOT / 'exp_dev' / 'gate8_full',
}

patterns = {
    'filter_only': '*/filter_only/split*/seed*/results.json',
    'caat':        '*/split*/seed*/results.json',
}

COLS = ['method', 'city', 'split', 'seed', 'backbone', 'macro_iou', 'bg_iou', 'fg_iou']

new_rows = []

for method, base in sources.items():
    count = 0
    for f in sorted(glob.glob(str(base / patterns[method]))):
        d = json.load(open(f))
        best = d.get('best', {}).get('model', {})
        val = best.get('macro_iou')
        if val is None:
            continue
        seed = next((p.replace('seed', '') for p in Path(f).parts if p.startswith('seed')), '')
        new_rows.append({
            'method':    method,
            'city':      d['dataset'],
            'split':     str(d['split']),
            'seed':      seed,
            'backbone':  d.get('backbone', 'resnet101'),
            'macro_iou': round(float(val), 4),
            'bg_iou':    round(float(best.get('bg_iou', 0)), 4),
            'fg_iou':    round(float(best.get('fg_iou', 0)), 4),
        })
        count += 1
    print(f'{method}: {count} runs loaded')

# Merge into results.csv (replace old filter_only/caat rows)
results_csv = ROOT / 'tables/results.csv'
existing = []
with open(results_csv) as f:
    for row in csv.DictReader(f):
        if row['method'] not in ('filter_only', 'caat'):
            existing.append({c: row[c] for c in COLS})

all_rows = existing + new_rows
with open(results_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=COLS)
    w.writeheader()
    w.writerows(all_rows)

print(f'Updated {results_csv} — {len(all_rows)} total rows ({len(new_rows)} new)')

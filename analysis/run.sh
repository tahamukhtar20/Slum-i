#!/bin/bash
# Run the data quality analysis: compute metrics, then render the figure.
#
# Submit on the cluster:
#   bash scripts/run.sh --mem 96 --time 6 'bash analysis/run.sh'
#
set -euo pipefail

ROOT="${SLUM_I_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"

pip install -q opencv-python-headless seaborn scikit-learn scikit-image tqdm scipy pandas matplotlib

echo "=== [1/2] Computing per-tile metrics (this is the heavy step) ==="
python -u analysis/compute_metrics.py \
    --out "$ROOT/analysis/metrics.csv"

echo "=== [2/2] Building figure ==="
python -u analysis/make_figure.py \
    --csv "$ROOT/analysis/metrics.csv" \
    --out "$ROOT/analysis/fig_data_quality.pdf"

echo "=== Done. Output in analysis/ ==="
ls -la "$ROOT/analysis/"

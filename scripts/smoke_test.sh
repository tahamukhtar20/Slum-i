#!/bin/bash
# Smoke test: verify ours.py runs end-to-end (2 epochs, small dataset).
# Exits non-zero on any failure.
#
# Usage:
#   bash scripts/run.sh --gpu "bash scripts/smoke_test.sh"
#   # or directly inside container:
#   bash scripts/smoke_test.sh
#
set -euo pipefail

ROOT="${SLUM_I_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
export DATA_ROOT="$ROOT/data_pipeline/datasets"

cd "$ROOT"
pip install -q -r requirements.txt

echo "=== Smoke test: ours on lower 10% seed42 ==="
python -u ours.py \
    --config configs/deeplab/benchmark/lower.yaml \
    --labeled-id-path   "$DATA_ROOT/lower/splits/10/labeled.txt" \
    --unlabeled-id-path "$DATA_ROOT/lower/splits/10/unlabeled.txt" \
    --save-path $ROOT/tmp/ours_smoke \
    --split 10 \
    --seed 42 \
    --epochs 2 \
    --no-wandb

echo "=== Smoke test PASSED ==="

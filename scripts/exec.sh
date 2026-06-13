#!/bin/bash
# Inside-container entrypoint for a Slum-i training run.
#
# Args:
#   $1  method      e.g. "slum-i", "fixmatch", "unimatch", "supervised"
#   $2  config      e.g. "configs/deeplab/benchmark/lahore.yaml"
#   $3  labeled     e.g. "data_pipeline/datasets/lahore/splits/10/labeled.txt"
#   $4  unlabeled   e.g. "data_pipeline/datasets/lahore/splits/10/unlabeled.txt"
#                   (pass "" for supervised methods)
#   $5  save_path   e.g. "exp/lahore/slum-i/r101/10/seed42"
#   $6  split       e.g. "10"
#   $7  seed        e.g. "42"
#   $8+ extra args  e.g. "--no-wandb"

set -euo pipefail

METHOD="$1"
CONFIG="$2"
LABELED="$3"
UNLABELED="$4"
SAVE_PATH="$5"
SPLIT="${6:-full}"
SEED="${7:-42}"
shift 7 || true
EXTRA_ARGS=("$@")

ROOT="${SLUM_I_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
export DATA_ROOT="$ROOT/data_pipeline/datasets"

# Source secrets if available (wandb API key etc.)
# Set WANDB_API_KEY in your environment, or source a secrets file below.
# Alternatively, pass --no-wandb to disable wandb logging.
if [[ -f "${WANDB_ENV_FILE:-/dev/null}" ]]; then
    source "${WANDB_ENV_FILE}"
fi

echo "--- Slum-i run metadata ---"
echo "method:    $METHOD"
echo "config:    $CONFIG"
echo "labeled:   $LABELED"
echo "unlabeled: ${UNLABELED:-(none)}"
echo "save_path: $SAVE_PATH"
echo "split:     $SPLIT"
echo "seed:      $SEED"
echo "host:      $(hostname)"
echo "date:      $(date -Iseconds)"
echo "python:    $(python --version 2>&1)"
echo "---------------------------"

pip install --quiet --disable-pip-version-check -r "$ROOT/requirements.txt"

mkdir -p "$SAVE_PATH"

PY_ARGS=(
    --config "$CONFIG"
    --labeled-id-path "$LABELED"
    --save-path "$SAVE_PATH"
    --split "$SPLIT"
    --seed "$SEED"
)

if [[ -n "$UNLABELED" ]]; then
    PY_ARGS+=(--unlabeled-id-path "$UNLABELED")
fi

python -u "$ROOT/${METHOD}.py" "${PY_ARGS[@]}" "${EXTRA_ARGS[@]}" 2>&1

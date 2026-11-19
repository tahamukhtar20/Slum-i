#!/bin/bash
# Run one line of a run manifest (scripts/runs/table2.txt or table3.txt) from the repo root.
#   bash scripts/train.sh scripts/runs/table2.txt 0          # first run, on the local GPU
#   sbatch --array=0-779 --gpus=1 --mem=64G scripts/train.sh scripts/runs/table2.txt   # one task per run
# Each line is: <script> <config> <labelled list> <output dir> <budget> <seed> [<unlabelled list>].
# A run whose results.json exists is skipped, so an array can be resubmitted after a timeout.
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
I=${2:-$SLURM_ARRAY_TASK_ID}
LINE=$(sed -n "$((I + 1))p" "$1")
read -r METH CFG LAB SAVE SPLIT SEED UNLAB <<< "$LINE"
if [ -f "$SAVE/results.json" ]; then echo "done already: $SAVE"; exit 0; fi
mkdir -p "$SAVE"
python -u "$METH.py" --config "$CFG" --labeled-id-path "$LAB" --save-path "$SAVE" \
  --split "$SPLIT" --seed "$SEED" ${UNLAB:+--unlabeled-id-path "$UNLAB"} --no-wandb

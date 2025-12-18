#!/bin/bash

set -euo pipefail

method="$1"
config="$2"
labeled_id_path="$3"
unlabeled_id_path="$4"
save_path="$5"
now="$6"
split="$7"

echo "--- Experiment Metadata ---"
echo "Method: $method | Config: $config"
echo "Log:    $save_path/${now}.log"
echo "---------------------------"

pip install -r requirements.txt

PY_ARGS=(
    --config "$config"
    --labeled-id-path "$labeled_id_path"
    --save-path "$save_path"
    --split "$split"
)

if [[ -n "$unlabeled_id_path" ]]; then
    PY_ARGS+=(--unlabeled-id-path "$unlabeled_id_path")
fi

mkdir -p "$save_path"

SCRIPT_NAME="${method}.py"
if [[ ! -f "$SCRIPT_NAME" ]]; then
    echo "Error: Training script $SCRIPT_NAME not found."
    exit 1
fi

python -u "$SCRIPT_NAME" "${PY_ARGS[@]}" 2>&1 | tee "$save_path/${now}.log"
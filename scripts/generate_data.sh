#!/bin/bash
#SBATCH -p RTXA6000-SDS,A100-SDS,V100-32GB-SDS,H200-SDS,RTX3090,RTXA6000,A100-40GB,A100-80GB
#SBATCH --job-name=clustered-generate
#SBATCH --gpus=0
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --container-image=/enroot/nvcr.io_nvidia_pytorch_25.02-py3.sqsh
#SBATCH --container-workdir=$SLUM_I_ROOT  # set SLUM_I_ROOT before submitting
#SBATCH --container-mounts=/netscratch:/netscratch
#SBATCH --output=logs/data_pipeline/generate-%j.out

set -euo pipefail

CONFIG="$1"
CITY=$(basename "$CONFIG" .yaml)
ROOT="${SLUM_I_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

LOG_FILE="$ROOT/data_pipeline/logs/generate_${CITY}.log"
mkdir -p "$ROOT/data_pipeline/logs"

echo "--- clustered generate ---" | tee "$LOG_FILE"
echo "config:  $CONFIG"           | tee -a "$LOG_FILE"
echo "city:    $CITY"             | tee -a "$LOG_FILE"
echo "host:    $(hostname)"       | tee -a "$LOG_FILE"
echo "date:    $(date -Iseconds)" | tee -a "$LOG_FILE"
echo "--------------------------" | tee -a "$LOG_FILE"

pip install --quiet mapcv pyproj shapely scikit-learn pyyaml 2>&1 | tee -a "$LOG_FILE"

python3 "$ROOT/data_pipeline/generate_clustered.py" "$CONFIG" 2>&1 | tee -a "$LOG_FILE"
python3 "$ROOT/data_pipeline/split_clustered.py" --config "$CONFIG" 2>&1 | tee -a "$LOG_FILE"

echo "Done: $(date -Iseconds)" | tee -a "$LOG_FILE"

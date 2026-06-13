#!/bin/bash
# SLURM array wrapper for a Slum-i sweep.
# Picks one run from a sweep directory by $SLURM_ARRAY_TASK_ID.
#
# Usage (submitted by scripts/sweep.py):
#   sbatch --array=0-N%K scripts/array.sh <sweep_dir>
#
#SBATCH -p RTXA6000-SDS,A100-SDS,H200-SDS,RTXA6000,A100-40GB,A100-80GB,A100-PCI,A100-RP,H100,H100-RP,H100-PCI,H200,H200-PCI,L40S,RTXB6000
#SBATCH --exclude=serv-3310
#SBATCH --job-name=slumi-sweep
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --container-image=/enroot/nvcr.io_nvidia_pytorch_25.02-py3.sqsh
#SBATCH --container-workdir=$SLUM_I_ROOT  # set SLUM_I_ROOT before submitting
#SBATCH --container-mounts=/netscratch:/netscratch
#SBATCH --output=logs/slurm-%A_%a.out

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: sbatch --array=... array.sh <sweep_dir>" >&2
    exit 2
fi

SWEEP_DIR="$1"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
RUN_FILE=$(printf "%s/run_%04d.sh" "$SWEEP_DIR" "$TASK_ID")

if [ ! -f "$RUN_FILE" ]; then
    echo "missing run file for task $TASK_ID: $RUN_FILE" >&2
    exit 3
fi

mkdir -p logs
bash "$RUN_FILE"

#!/bin/bash
# Single training job on DFKI Pegasus.
#
# Usage:
#   sbatch scripts/train.sh <method> <config> <labeled> <unlabeled> <save_path> [split] [seed]
#
# Examples:
#   sbatch scripts/train.sh ours configs/deeplab/benchmark/lahore.yaml \
#       data_pipeline/datasets/lahore/splits/10/labeled.txt \
#       data_pipeline/datasets/lahore/splits/10/unlabeled.txt \
#       exp/lahore/ours/r101/10/seed42 10 42
#
#   sbatch scripts/train.sh supervised configs/deeplab/benchmark/lahore.yaml \
#       data_pipeline/datasets/lahore/splits/train.txt "" \
#       exp/lahore/supervised/r101/full/seed42 full 42
#
#SBATCH -p RTXA6000-SDS,A100-SDS,H200-SDS,RTXA6000,A100-40GB,A100-80GB
#SBATCH --job-name=slum-i
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --container-image=/enroot/nvcr.io_nvidia_pytorch_25.02-py3.sqsh
#SBATCH --container-workdir=$SLUM_I_ROOT  # set SLUM_I_ROOT before submitting
#SBATCH --container-mounts=/netscratch:/netscratch
#SBATCH --output=logs/slurm-%j.out

set -euo pipefail
mkdir -p logs
scripts/exec.sh "$@"

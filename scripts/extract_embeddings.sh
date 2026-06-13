#!/bin/bash
#SBATCH -p RTXA6000-SDS,A100-SDS,H200-SDS,RTXA6000,A100-40GB,A100-80GB,A100-PCI,A100-RP,H100,H100-RP,H100-PCI,H200,H200-PCI,L40S,RTXB6000
#SBATCH --exclude=serv-3310
#SBATCH --job-name=slumi-embeddings
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --container-image=/enroot/nvcr.io_nvidia_pytorch_25.02-py3.sqsh
#SBATCH --container-workdir=$SLUM_I_ROOT  # set SLUM_I_ROOT before submitting
#SBATCH --container-mounts=/netscratch:/netscratch
#SBATCH --output=logs/slurm-%j.out

set -euo pipefail
ROOT="${SLUM_I_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
pip install --quiet --disable-pip-version-check -r "${ROOT}/requirements.txt"
python -u "${ROOT}/scripts/extract_embeddings.py"

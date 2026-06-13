#!/bin/bash
# One-off job runner — submits any command inside the Slum-i container.
#
# Usage:
#   bash scripts/run.sh "python some_script.py --arg val"
#   bash scripts/run.sh --gpu "python training_script.py"  # request 1 GPU
#
# Options (before the command string):
#   --gpu       Request 1 GPU (default: 0)
#   --mem MB    Memory in G (default: 32)
#   --time HH   Time limit hours (default: 1)

set -euo pipefail

GPUS=0
MEM=32
TIME=1

while [[ "$1" == --* ]]; do
    case "$1" in
        --gpu)  GPUS=1; shift ;;
        --mem)  MEM="$2"; shift 2 ;;
        --time) TIME="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

CMD="$*"
ROOT="${SLUM_I_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

sbatch \
    -p RTXA6000-SDS,A100-SDS,H200-SDS,RTXA6000,A100-40GB,A100-80GB \
    --job-name=slumi-run \
    --gpus="${GPUS}" \
    --cpus-per-task=8 \
    --mem="${MEM}G" \
    --time="0${TIME}:00:00" \
    --container-image=/enroot/nvcr.io_nvidia_pytorch_25.02-py3.sqsh \
    --container-workdir="$ROOT" \
    --container-mounts=/netscratch:/netscratch \
    --output="$ROOT/logs/run-%j.out" \
    --wrap="cd $ROOT && pip install -q -r requirements.txt && $CMD"

echo "Submitted. Tail log: tail -f $ROOT/logs/run-\$(squeue -u mukhtar -h -o '%i' | tail -1).out"

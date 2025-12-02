#!/bin/bash
#SBATCH -p B200,H100,H200,H200-SDS,A100-80GB
#SBATCH --job-name=training
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --container-image=/enroot/nvcr.io_nvidia_pytorch_25.02-py3.sqsh
#SBATCH --container-workdir=/netscratch/mukhtar/UniMatch
#SBATCH --container-mounts=/netscratch:/netscratch,/netscratch/mukhtar/UniMatch:/netscratch/mukhtar/UniMatch
#SBATCH --output=logs/slurm-%j.out

method="$1"

if [ "$method" == "supervised" ]; then
    scripts/exec.sh "$1" "$2" "$3" "" "$4" "$5"
else
    scripts/exec.sh "$1" "$2" "$3" "$4" "$5" "$6" "$7"
fi
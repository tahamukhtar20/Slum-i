#!/bin/bash
#SBATCH -p RTXA6000
#SBATCH --job-name=training
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --container-image=/enroot/nvcr.io_nvidia_pytorch_25.02-py3.sqsh
#SBATCH --container-workdir=$SLUM_I_ROOT  # set SLUM_I_ROOT before submitting
#SBATCH --container-mounts=/netscratch:/netscratch
#SBATCH --output=logs/slurm-%j.out

if [ "$#" -lt 6 ]; then
    echo "Error: Expected at least 6 arguments, got $#"
    exit 1
fi

if [ "$#" -eq 6 ]; then
    scripts/exec.sh "$1" "$2" "$3" "" "$4" "$5" "$6"
elif [ "$#" -eq 7 ]; then
    scripts/exec.sh "$1" "$2" "$3" "$4" "$5" "$6" "$7"
else
    echo "Error: Unexpected number of arguments ($#). Expected 6 or 7."
    exit 1
fi

#!/bin/bash

method="$1"
now=$(date +"%Y%m%d_%H%M%S")
exp='r101'
datasets=('lower')

if [ "$method" == "supervised" ]; then
    for dataset in "${datasets[@]}"; do
        config="configs/${dataset}.yaml"
        labeled_id_path="splits/${dataset}/train.txt"
        save_path="exp/${dataset}/${method}/${exp}"
        mkdir -p "$save_path"
        sbatch scripts/train_slurm.sh "$method" "$config" "$labeled_id_path" "$save_path" "$now"
    done
elif [ "$method" == "fixmatch" ] || [ "$method" == "unimatch" ]; then
    for dataset in "${datasets[@]}"; do
        for split in 10 20 30; do
            config="configs/${dataset}.yaml"
            labeled_id_path="splits/${dataset}/${split}/labeled.txt"
            unlabeled_id_path="splits/${dataset}/${split}/unlabeled.txt"
            save_path="exp/${dataset}/${method}/${exp}/${split}"
            mkdir -p "$save_path"
            sbatch scripts/train_slurm.sh "$method" "$config" "$labeled_id_path" "$unlabeled_id_path" "$save_path" "$now" "$split"
        done
    done
elif [ "$method" == "unimatchv2" ]; then
    for dataset in "${datasets[@]}"; do
        for split in 10 20 30; do
            config="configs/${dataset}_dino.yaml"
            labeled_id_path="splits/${dataset}/${split}/labeled.txt"
            unlabeled_id_path="splits/${dataset}/${split}/unlabeled.txt"
            save_path="exp/${dataset}/${method}/${exp}/${split}"
            mkdir -p "$save_path"
            sbatch scripts/train_slurm.sh "$method" "$config" "$labeled_id_path" "$unlabeled_id_path" "$save_path" "$now" "$split"
        done
    done
elif [ "$method" == "ours" ]; then
    for dataset in "${datasets[@]}"; do
        for split in 10; do
            config="configs/${dataset}_ours.yaml"
            labeled_id_path="splits/${dataset}/${split}/labeled.txt"
            unlabeled_id_path="splits/${dataset}/${split}/unlabeled.txt"
            save_path="exp/${dataset}/${method}/${exp}/${split}"
            mkdir -p "$save_path"
            sbatch scripts/train_slurm.sh "$method" "$config" "$labeled_id_path" "$unlabeled_id_path" "$save_path" "$now" "$split"
        done
    done
else
    echo "Unknown method: $method"
    exit 1
fi

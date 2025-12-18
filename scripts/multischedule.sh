#!/bin/bash

method="$1"
run_all_splits="$2"
now=$(date +"%Y%m%d_%H%M%S")
exp='r101'
# datasets=('lahore' 'karachi' 'mumbai' 'lower' 'medellin' 'makoko' 'eldaein' 'elgeneina')
# splits=(10 20 30)
datasets=('lower')
splits=(10)

case "$method" in
    supervised|supervised-dino)
        is_semi=false
        [[ "$method" == *"dino"* ]] && config_prefix="configs/dino/benchmark" || config_prefix="configs/deeplab/benchmark"
        ;;
    fixmatch|unimatch|ours)
        is_semi=true
        [[ "$method" == "ours" ]] && config_prefix="configs/deeplab/ours" || config_prefix="configs/deeplab/benchmark"
        ;;
    unimatchv2|ours-dino)
        is_semi=true
        [[ "$method" == "ours-dino" ]] && config_prefix="configs/dino/ours" || config_prefix="configs/dino/benchmark"
        ;;
    *)
        echo "Error: Unknown method: $method"
        exit 1
        ;;
esac

for dataset in "${datasets[@]}"; do
    config="${config_prefix}/${dataset}.yaml"
    if [[ "$is_semi" == true ]]; then
        for split in "${splits[@]}"; do
            labeled_id_path="splits/${dataset}/${split}/labeled.txt"
            unlabeled_id_path="splits/${dataset}/${split}/unlabeled.txt"
            # save_path="exp/${dataset}/${method}/${exp}/${split}"
            save_path="tmp/exp/${dataset}/${method}/${exp}/${split}"

            [[ ! -f "$labeled_id_path" ]] && echo "Missing $labeled_id_path, skipping." && continue

            mkdir -p "$save_path"
            sbatch scripts/train_slurm.sh "$method" "$config" "$labeled_id_path" "$unlabeled_id_path" "$save_path" "$now" "$split"
        done
    else
        if [[ "$run_all_splits" == "y" ]]; then
            for split in "${splits[@]}"; do
                labeled_id_path="splits/${dataset}/${split}/labeled.txt"
                save_path="exp/${dataset}/${method}/${exp}/${split}"

                [[ ! -f "$labeled_id_path" ]] && echo "Missing $labeled_id_path, skipping." && continue

                mkdir -p "$save_path"
                sbatch scripts/train_slurm.sh "$method" "$config" "$labeled_id_path" "$save_path" "$now" "$split"
            done
        else
            split="full"
            labeled_id_path="splits/${dataset}/train.txt"
            save_path="exp/${dataset}/${method}/${exp}/${split}"

            mkdir -p "$save_path"
            sbatch scripts/train_slurm.sh "$method" "$config" "$labeled_id_path" "$save_path" "$now" "$split"
        fi
    fi
done

import os
import random
import math
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from collections import Counter

DATASET_DIRS = ["Lahore", "Karachi", "Mumbai"]
BASE_INPUT_DIR = "/netscratch/mukhtar/DataExtraction"
BASE_OUTPUT_DIR = "/netscratch/mukhtar/UniMatch/splits"

TEST_RATIO = 0.20
VAL_RATIO_FROM_TRAIN_VAL = 0.10
LABELED_RATIOS = [0.10, 0.20, 0.30]
RANDOM_SEED = 42
SAMPLES = 1000

def classify_mask(mask_path):
    mask = Image.open(mask_path).convert('L')
    mask_np = np.array(mask)
    unique_vals = np.unique(mask_np)
    if len(unique_vals) == 1:
        return 0 if unique_vals[0] == 0 else 1
    return 2

def get_stats(files, lookup):
    if len(files) == 0:
        return {}
    counts = Counter([lookup[f] for f in files])
    total = len(files)
    return {cls: f"{(count/total)*100:.1f}%" for cls, count in counts.items()}

def split_dataset(dataset_name):
    random.seed(RANDOM_SEED)

    input_path = Path(BASE_INPUT_DIR) / dataset_name
    img_dir = input_path / "Images"
    mask_dir = input_path / "Masks"
    output_dir = Path(BASE_OUTPUT_DIR) / dataset_name.lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_files = [f for f in os.listdir(img_dir) if (img_dir / f).is_file()]
    if not all_files:
        return

    file_to_class = {}
    stratified_groups = {0: [], 1: [], 2: []}

    for fname in tqdm(all_files, desc=f"Classifying {dataset_name}"):
        cls = classify_mask(mask_dir / fname)
        stratified_groups[cls].append(fname)
        file_to_class[fname] = cls

    total_available = len(all_files)
    target_total = min(SAMPLES, total_available) if SAMPLES else total_available

    sampled_pool = []
    for cls, files in stratified_groups.items():
        k = int(round((len(files) / total_available) * target_total))
        sampled_pool.extend(random.sample(files, min(k, len(files))))

    random.shuffle(sampled_pool)

    test_size = math.ceil(len(sampled_pool) * TEST_RATIO)
    test_files = sampled_pool[:test_size]
    train_val_pool = sampled_pool[test_size:]

    val_size = math.ceil(len(train_val_pool) * VAL_RATIO_FROM_TRAIN_VAL)
    val_files = train_val_pool[:val_size]
    ssl_train_pool = train_val_pool[val_size:]

    splits = {
        "test.txt": test_files,
        "val.txt": val_files,
        "train.txt": ssl_train_pool
    }

    for name, files in splits.items():
        with open(output_dir / name, 'w') as f:
            f.write('\n'.join(files))
        print(f"{dataset_name} -> {name}: {len(files)} | {get_stats(files, file_to_class)}")

    random.shuffle(ssl_train_pool)

    for ratio in LABELED_RATIOS:
        ratio_dir = output_dir / str(int(ratio * 100))
        ratio_dir.mkdir(parents=True, exist_ok=True)

        l_size = math.ceil(len(ssl_train_pool) * ratio)
        l_files = ssl_train_pool[:l_size]
        u_files = ssl_train_pool[l_size:]

        with open(ratio_dir / "labeled.txt", 'w') as f:
            f.write('\n'.join(l_files))
        with open(ratio_dir / "unlabeled.txt", 'w') as f:
            f.write('\n'.join(u_files))

        print(f"{dataset_name} -> {int(ratio*100)}% labeled: {len(l_files)}, unlabeled: {len(u_files)}")

def main():
    for dataset in DATASET_DIRS:
        split_dataset(dataset)

if __name__ == "__main__":
    main()

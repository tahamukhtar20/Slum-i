#!/usr/bin/env python3
"""Split a clustered-tiling dataset into train/val/test + labeled/unlabeled.

Reads manifest.json + clusters.json from a city's output dir and writes:
    splits/train.txt, val.txt, test.txt
    splits/{10,20,30}/labeled.txt, unlabeled.txt

Two strategies are supported via --strategy:

    cluster_level (default):
        Whole clusters are assigned to one of train/val/test. Zero geographic
        leakage between splits, but ratios are approximate (greedy bin-pack
        to match target ratios as closely as possible).

    patch_level:
        Patches are split with stratified random shuffle, ignoring cluster_id.
        Strict ratios, but neighbouring patches within a cluster can land in
        different splits (mild intra-cluster correlation).

In both strategies, the labeled/unlabeled split within the train pool is
patch-level stratified at the requested ratios.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple


# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------

def _load_manifest(path: Path) -> List[dict]:
    data = json.loads(path.read_text())
    return list(data.get("patches", []))


def _load_clusters(path: Path) -> Dict[str, int]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {str(k): int(v) for k, v in data.get("assignments", {}).items()}


def _classify_entry(entry: dict) -> int:
    """0 = pure background, 1 = pure foreground, 2 = mixed.

    Mirrors mapcv.splitter._classify_entry so the labeled stratification matches.
    """
    counts = entry.get("per_class_pixel_counts") or {}
    if counts:
        has_bg = "0" in counts
        has_fg = any(k != "0" for k in counts)
        if has_bg and not has_fg:
            return 0
        if has_fg and not has_bg:
            return 1
        return 2
    empty = entry.get("empty_ratio", 0.0)
    if empty >= 1.0:
        return 0
    if empty <= 0.0:
        return 1
    return 2


# -----------------------------------------------------------------------------
# Cluster-level split
# -----------------------------------------------------------------------------

def _cluster_level_split(
    cluster_to_patches: Dict[int, List[str]],
    test_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    """Best-fit-decreasing bin-pack: assign whole clusters to splits.

    For each cluster (largest first), pick the pool whose post-assignment size
    has the smallest relative error w.r.t. its target ratio. Zero leakage,
    ratios approximate (gracefully degrades when one cluster is larger than
    a pool's entire target).
    """
    total = sum(len(v) for v in cluster_to_patches.values())
    target_test = total * test_ratio
    target_val = total * (1 - test_ratio) * val_ratio
    target_train = total - target_test - target_val

    pools: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    targets = {"train": target_train, "val": target_val, "test": target_test}

    rng = random.Random(seed)
    items = sorted(cluster_to_patches.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    for _, patches in items:
        size = len(patches)
        best_pool = None
        best_score = float("inf")
        for pool_name, target in targets.items():
            new_size = len(pools[pool_name]) + size
            score = abs(new_size - target) / max(target, 1.0)
            if score < best_score:
                best_score = score
                best_pool = pool_name
        pools[best_pool].extend(patches)

    rng.shuffle(pools["train"])
    rng.shuffle(pools["val"])
    rng.shuffle(pools["test"])
    return pools["train"], pools["val"], pools["test"]


# -----------------------------------------------------------------------------
# Patch-level split (mapcv-style stratified)
# -----------------------------------------------------------------------------

def _patch_level_split(
    entries: List[dict],
    test_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    rng = random.Random(seed)
    groups: Dict[int, List[str]] = {0: [], 1: [], 2: []}
    for e in entries:
        groups[_classify_entry(e)].append(e["filename"])
    pool: List[str] = []
    for files in groups.values():
        rng.shuffle(files)
        pool.extend(files)
    rng.shuffle(pool)

    test_size = math.ceil(len(pool) * test_ratio)
    test_files = pool[:test_size]
    rest = pool[test_size:]
    val_size = math.ceil(len(rest) * val_ratio)
    val_files = rest[:val_size]
    train_files = rest[val_size:]
    return train_files, val_files, test_files


# -----------------------------------------------------------------------------
# Labeled / unlabeled split (always patch-level, stratified)
# -----------------------------------------------------------------------------

def _labeled_unlabeled(
    train_files: List[str],
    classes_by_file: Dict[str, int],
    ratios: List[float],
    seed: int,
    output_dir: Path,
) -> None:
    rng = random.Random(seed)
    groups: Dict[int, List[str]] = {0: [], 1: [], 2: []}
    for f in train_files:
        groups[classes_by_file.get(f, 0)].append(f)
    for k in groups:
        rng.shuffle(groups[k])

    for ratio in ratios:
        labeled: List[str] = []
        unlabeled: List[str] = []
        for cls_files in groups.values():
            k = math.ceil(len(cls_files) * ratio)
            labeled.extend(cls_files[:k])
            unlabeled.extend(cls_files[k:])
        rng.shuffle(labeled)
        rng.shuffle(unlabeled)
        ratio_dir = output_dir / str(int(round(ratio * 100)))
        ratio_dir.mkdir(parents=True, exist_ok=True)
        (ratio_dir / "labeled.txt").write_text("\n".join(labeled))
        (ratio_dir / "unlabeled.txt").write_text("\n".join(unlabeled))


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def run(
    input_dir: Path,
    strategy: Literal["cluster_level", "patch_level"],
    test_ratio: float,
    val_ratio: float,
    labeled_ratios: List[float],
    seed: int,
) -> None:
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {input_dir}")
    entries = _load_manifest(manifest_path)
    if not entries:
        print("[split_clustered] manifest has no patches; nothing to do")
        return

    cluster_map = _load_clusters(input_dir / "clusters.json")
    classes_by_file = {e["filename"]: _classify_entry(e) for e in entries}

    if strategy == "cluster_level":
        if not cluster_map:
            print(
                "[split_clustered] no clusters.json found; falling back to patch_level"
            )
            train, val, test = _patch_level_split(entries, test_ratio, val_ratio, seed)
        else:
            buckets: Dict[int, List[str]] = defaultdict(list)
            for e in entries:
                fname = e["filename"]
                cid = cluster_map.get(fname, -1)
                buckets[cid].append(fname)
            train, val, test = _cluster_level_split(buckets, test_ratio, val_ratio, seed)
    elif strategy == "patch_level":
        train, val, test = _patch_level_split(entries, test_ratio, val_ratio, seed)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    splits_dir = input_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "train.txt").write_text("\n".join(train))
    (splits_dir / "val.txt").write_text("\n".join(val))
    (splits_dir / "test.txt").write_text("\n".join(test))

    _labeled_unlabeled(train, classes_by_file, labeled_ratios, seed, splits_dir)

    total = len(train) + len(val) + len(test)
    print(
        f"[split_clustered] strategy={strategy} total={total} "
        f"train={len(train)} ({100 * len(train) / total:.1f}%) "
        f"val={len(val)} ({100 * len(val) / total:.1f}%) "
        f"test={len(test)} ({100 * len(test) / total:.1f}%)"
    )
    for r in labeled_ratios:
        lpath = splits_dir / str(int(round(r * 100))) / "labeled.txt"
        if lpath.exists():
            n = len(lpath.read_text().splitlines())
            print(f"[split_clustered]   labeled@{int(round(r * 100))}%: {n} patches")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        type=Path,
        nargs="?",
        help="Dataset dir containing manifest.json (+ clusters.json)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional clustered-generation YAML; reads output_dir, seed, and split.* from it",
    )
    parser.add_argument(
        "--strategy",
        choices=["cluster_level", "patch_level"],
        default=None,
    )
    parser.add_argument("--test-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument(
        "--labeled-ratios",
        type=float,
        nargs="+",
        default=None,
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg_strategy = "cluster_level"
    cfg_test = 0.15
    cfg_val = 0.10
    cfg_labeled = [0.10, 0.20, 0.30]
    cfg_seed = 42
    cfg_input_dir: Optional[Path] = None

    if args.config is not None:
        import yaml

        raw = yaml.safe_load(Path(args.config).read_text())
        cfg_input_dir = Path(raw["output_dir"])
        cfg_seed = int(raw.get("seed", cfg_seed))
        split_cfg = raw.get("split", {}) or {}
        cfg_strategy = split_cfg.get("strategy", cfg_strategy)
        cfg_test = float(split_cfg.get("test_ratio", cfg_test))
        cfg_val = float(split_cfg.get("val_ratio", cfg_val))
        cfg_labeled = list(split_cfg.get("labeled_ratios", cfg_labeled))

    input_dir = args.input_dir or cfg_input_dir
    if input_dir is None:
        parser.error("Provide input_dir positionally or --config with output_dir")

    run(
        input_dir,
        args.strategy or cfg_strategy,
        args.test_ratio if args.test_ratio is not None else cfg_test,
        args.val_ratio if args.val_ratio is not None else cfg_val,
        args.labeled_ratios if args.labeled_ratios is not None else cfg_labeled,
        args.seed if args.seed is not None else cfg_seed,
    )


if __name__ == "__main__":
    main()

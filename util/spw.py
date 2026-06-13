"""Slum-prior pseudo-label reliability helpers.

The helpers are intentionally small and stateless so each component can be
run independently from the UniMatch training scripts.
"""

import math
import os

import numpy as np
from PIL import Image
import torch


def estimate_labeled_foreground_prior(data_root, labeled_id_path, device=None):
    """Estimate foreground pixel ratio from the raw labeled masks."""
    fg_pixels = 0
    valid_pixels = 0
    with open(labeled_id_path, "r") as f:
        ids = [line.strip() for line in f if line.strip()]

    for tile_id in ids:
        mask_path = os.path.join(data_root, "Masks", tile_id)
        mask = np.array(Image.open(mask_path))
        if mask.max() > 1:
            mask = mask // 255
        valid = mask != 255
        fg_pixels += int(((mask == 1) & valid).sum())
        valid_pixels += int(valid.sum())

    prior = fg_pixels / max(valid_pixels, 1)
    tensor = torch.tensor(float(prior), dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def _valid_denominator(ignore_mask):
    return (ignore_mask != 255).sum().clamp_min(1).float()


def _hard_weight(conf, ignore_mask, conf_thresh):
    return ((conf >= conf_thresh) & (ignore_mask != 255)).float()

def _warmup_thresh(epoch, conf_thresh, warmup_epochs, warmup_start):
    """Linearly ramp conf_thresh from warmup_start up to conf_thresh over warmup_epochs."""
    if warmup_epochs <= 0 or epoch >= warmup_epochs:
        return conf_thresh
    t = float(epoch) / float(warmup_epochs)
    return warmup_start + (conf_thresh - warmup_start) * t



def _rank_weight(conf, pseudo, ignore_mask, keep_fraction, min_conf):
    valid = (ignore_mask != 255) & (conf >= min_conf)
    weights = torch.zeros_like(conf, dtype=torch.float32)
    keep_fraction = min(max(float(keep_fraction), 0.0), 1.0)

    if keep_fraction <= 0:
        return weights

    nclass = int(pseudo.max().item()) + 1 if pseudo.numel() else 0
    for cls in range(nclass):
        cls_mask = valid & (pseudo == cls)
        values = conf[cls_mask]
        if values.numel() == 0:
            continue
        k = max(1, int(math.ceil(values.numel() * keep_fraction)))
        cutoff = torch.topk(values, k, largest=True).values.min()
        selected = cls_mask & (conf >= cutoff)
        weights[selected] = conf[selected]
    return weights


def _prior_weight(conf, pseudo, ignore_mask, conf_thresh, foreground_prior, strength, clip):
    base = _hard_weight(conf, ignore_mask, conf_thresh)
    valid = ignore_mask != 255
    valid_count = _valid_denominator(ignore_mask)

    pseudo_fg = ((pseudo == 1) & valid).float().sum() / valid_count
    target_fg = foreground_prior.to(conf.device).clamp(1e-4, 1.0 - 1e-4)
    strength = float(strength)
    clip = float(clip)

    fg_scale = (target_fg / (pseudo_fg + 1e-6)).pow(strength).clamp(1.0 / clip, clip)
    bg_scale = ((1.0 - target_fg) / (1.0 - pseudo_fg + 1e-6)).pow(strength).clamp(1.0 / clip, clip)
    scale = torch.where(pseudo == 1, fg_scale, bg_scale)
    return base * scale.float()


def _boundary_weight(probs, conf, ignore_mask, conf_thresh, gamma):
    base = _hard_weight(conf, ignore_mask, conf_thresh)
    nclass = probs.shape[1]
    entropy = -(probs.clamp_min(1e-6) * probs.clamp_min(1e-6).log()).sum(dim=1)
    entropy = entropy / math.log(max(nclass, 2))
    confidence_structure = (1.0 - entropy).clamp(0.0, 1.0).pow(float(gamma))
    return base * confidence_structure


def spw_reliability_weight(
    probs,
    pseudo,
    ignore_mask,
    mode="none",
    conf_thresh=0.95,
    rank_keep=0.50,
    rank_min_conf=0.60,
    foreground_prior=None,
    prior_strength=0.50,
    prior_clip=3.0,
    boundary_gamma=1.0,
    epoch=0,
    warmup_epochs=0,
    warmup_start=0.5,
):
    """Return a BxHxW reliability map for pseudo-label CE.

    Modes are disjoint by design. `combined` is only for testing surviving
    components after individual 9-cell validation runs.
    """
    conf = probs.max(dim=1).values
    conf_thresh = _warmup_thresh(epoch, conf_thresh, warmup_epochs, warmup_start)
    in_warmup = (warmup_epochs > 0) and (epoch < warmup_epochs)
    effective_gamma = 0.0 if in_warmup else boundary_gamma
    mode = (mode or "none").lower()

    if mode == "none":
        return _hard_weight(conf, ignore_mask, conf_thresh)
    if mode == "rank":
        return _rank_weight(conf, pseudo, ignore_mask, rank_keep, rank_min_conf)
    if mode == "prior":
        if foreground_prior is None:
            raise ValueError("foreground_prior is required for SPW prior mode")
        return _prior_weight(conf, pseudo, ignore_mask, conf_thresh, foreground_prior, prior_strength, prior_clip)
    if mode == "boundary":
        return _boundary_weight(probs, conf, ignore_mask, conf_thresh, effective_gamma)
    if mode == "combined":
        if foreground_prior is None:
            raise ValueError("foreground_prior is required for SPW combined mode")
        rank = _rank_weight(conf, pseudo, ignore_mask, rank_keep, rank_min_conf)
        prior = _prior_weight(conf, pseudo, ignore_mask, conf_thresh, foreground_prior, prior_strength, prior_clip)
        boundary = _boundary_weight(probs, conf, ignore_mask, conf_thresh, effective_gamma)
        prior_scale = prior / (_hard_weight(conf, ignore_mask, conf_thresh) + 1e-6)
        boundary_scale = boundary / (_hard_weight(conf, ignore_mask, conf_thresh) + 1e-6)
        return rank * prior_scale.clamp(0.0, prior_clip) * boundary_scale.clamp(0.0, 1.0)

    raise ValueError(f"unknown SPW mode: {mode}")


def compute_class_weights_effective_num(data_root, labeled_id_path, nclass,
                                        beta=0.9999, device=None):
    """Class-balanced weights via effective number of samples (Cui et al. 2019).

    Returns a [nclass] tensor normalised so sum == nclass.
    """
    counts = np.zeros(nclass, dtype=np.int64)
    with open(labeled_id_path) as f:
        ids = [line.strip() for line in f if line.strip()]
    for tile_id in ids:
        mask = np.array(Image.open(os.path.join(data_root, 'Masks', tile_id)))
        if mask.max() > 1:
            mask = mask // 255
        for c in range(nclass):
            counts[c] += int(((mask == c) & (mask != 255)).sum())
    weights = np.zeros(nclass, dtype=np.float32)
    for c in range(nclass):
        if counts[c] > 0:
            weights[c] = (1.0 - beta) / (1.0 - beta ** counts[c])
    total = weights.sum()
    if total > 0:
        weights = weights / total * nclass
    tensor = torch.tensor(weights, dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def focal_modulation(probs, pseudo, gamma=2.0, ignore_mask=None):
    """Per-pixel focal modulation weight for the unlabeled branch.

    Returns a BxHxW detached float tensor.
    """
    p_t = probs.detach().gather(1, pseudo.unsqueeze(1).long()).squeeze(1)
    mod = (1.0 - p_t).clamp(min=1e-6) ** gamma
    if ignore_mask is not None:
        mod = mod * (ignore_mask != 255).float()
    return mod.detach()

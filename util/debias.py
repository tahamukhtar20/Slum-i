"""Imbalance-aware pseudo-labelling for semi-supervised segmentation.

LogitDebias: the teacher's predicted class distribution on unlabelled pixels is tracked as an
EMA (p_hat). Pseudo-labels are taken from debiased log-probabilities log p - lam * log p_hat,
which removes the majority (background) class's prior advantage (adaptive logit adjustment,
as in DebiasPL / DBMatch).

rank_weights: instead of a fixed confidence threshold, every pseudo-labelled pixel is weighted
by its confidence percentile among the pixels assigned to the same class in the batch, so the
minority class contributes as many well-ranked pixels as the majority (within-class rank
weighting, as in DWL). Pixels below `floor` percentile get weight 0.
"""
import torch


class LogitDebias:
    """With target=None: subtract lam*log(p_hat) (push towards a uniform prior, DebiasPL).
    With a target prior (e.g. the labelled-set class frequencies): distribution alignment,
    log p - lam*(log p_hat - log target) (ReMixMatch), which is zero once the teacher's
    predicted distribution matches the target, so it cannot over-correct."""

    def __init__(self, nclass, lam=0.5, momentum=0.999, device="cuda", target=None):
        self.lam = lam
        self.m = momentum
        self.p_hat = torch.full((nclass,), 1.0 / nclass, device=device)
        self.target = None if target is None else torch.as_tensor(target, dtype=torch.float32, device=device)

    @torch.no_grad()
    def update(self, probs, valid):
        """probs (B, C, H, W) teacher probabilities; valid (B, H, W) bool."""
        v = valid.unsqueeze(1).float()
        mean = (probs * v).sum((0, 2, 3)) / v.sum().clamp(min=1)
        self.p_hat = self.m * self.p_hat + (1 - self.m) * mean

    @torch.no_grad()
    def apply(self, probs):
        shift = self.p_hat.clamp(min=1e-8).log()
        if self.target is not None:
            shift = shift - self.target.clamp(min=1e-8).log()
        logits = probs.clamp(min=1e-8).log() - self.lam * shift.view(1, -1, 1, 1)
        return logits.softmax(1)


def labeled_prior(root, id_path, nclass, ignore=255):
    """Pixel class frequencies over the labelled tiles (ignore pixels excluded)."""
    import os
    import numpy as np
    from PIL import Image
    counts = np.zeros(nclass, dtype=np.float64)
    for name in open(id_path).read().split():
        m = np.asarray(Image.open(os.path.join(root, 'Masks', name)))
        counts += np.bincount(m[m != ignore].ravel(), minlength=256)[:nclass]
    return (counts / max(counts.sum(), 1)).tolist()


@torch.no_grad()
def _ranks(vals, ties):
    """1-based ranks of vals. ties: 'scan' (argsort order, the original behaviour), 'average'
    (tied values share their mean rank) or 'random' (ties broken by a random permutation)."""
    n = vals.numel()
    ar = torch.arange(1, n + 1, device=vals.device, dtype=vals.dtype)
    ranks = torch.empty_like(vals)
    if ties == "scan":
        ranks[vals.argsort()] = ar
    elif ties == "average":
        sv, order = vals.sort()
        _, inv, counts = torch.unique_consecutive(sv, return_inverse=True, return_counts=True)
        start = torch.cumsum(counts, 0) - counts  # 0-based position of each tie group
        ranks[order] = (start.to(vals.dtype) + (counts.to(vals.dtype) + 1) / 2)[inv]
    elif ties == "random":
        perm = torch.randperm(n, device=vals.device)
        order = vals[perm].sort(stable=True)[1]
        ranks[perm[order]] = ar
    else:
        raise ValueError(ties)
    return ranks


@torch.no_grad()
def rank_weights(conf, label, valid, nclass, floor=0.2, ties="scan"):
    """Per-pixel weight = within-class confidence percentile in the batch (0 if invalid or below floor)."""
    w = torch.zeros_like(conf)
    for c in range(nclass):
        sel = valid & (label == c)
        n = int(sel.sum())
        if n == 0:
            continue
        vals = conf[sel]
        ranks = _ranks(vals, ties)
        pct = ranks / n
        w[sel] = torch.where(pct >= floor, pct, torch.zeros_like(pct))
    return w


class SoftClassWeights:
    """Class-wise SoftMatch weighting (Chen et al., ICLR 2023), without uniform alignment.

    For each predicted class c the mean and variance of the teacher's confidence over unlabelled
    pixels are tracked as EMAs. A pixel predicted as c gets weight 1 if its confidence is at least
    the class mean, and exp(-(conf - mu_c)^2 / (2 var_c)) otherwise, so confident pixels keep full
    weight and the down-weighting of the rest tightens as the class becomes confident.
    """

    def __init__(self, nclass, momentum=0.999, device="cuda"):
        self.m = momentum
        self.mu = torch.full((nclass,), 1.0 / nclass, device=device)
        self.var = torch.ones(nclass, device=device)

    @torch.no_grad()
    def update(self, conf, cls, valid):
        for c in range(self.mu.numel()):
            v = conf[valid & (cls == c)]
            if v.numel() < 2:
                continue
            self.mu[c] = self.m * self.mu[c] + (1 - self.m) * v.mean()
            self.var[c] = self.m * self.var[c] + (1 - self.m) * v.var(unbiased=True)

    @torch.no_grad()
    def __call__(self, conf, cls, valid):
        mu, var = self.mu[cls], self.var[cls].clamp(min=1e-8)
        w = torch.where(conf >= mu, torch.ones_like(conf), torch.exp(-(conf - mu) ** 2 / (2 * var)))
        return w * valid.float()


class SelfAdaptiveThreshold:
    """FreeMatch self-adaptive thresholds (Wang et al., ICLR 2023), without the fairness loss.

    Global threshold tau = EMA of the mean teacher confidence; class threshold
    tau_c = tau * p_c / max(p). With local='prob' (FreeMatch), p = EMA of the mean class
    probability on unlabelled pixels, which also scales with the class prior; with local='conf',
    p = EMA of the mean confidence of the pixels predicted as each class (learning status only).
    A pixel is kept (weight 1) if its confidence exceeds the threshold of its predicted class.
    """

    def __init__(self, nclass, momentum=0.999, device="cuda", local="prob"):
        self.m = momentum
        self.local = local
        self.tau = torch.tensor(1.0 / nclass, device=device)
        self.p = torch.full((nclass,), 1.0 / nclass, device=device)

    @torch.no_grad()
    def update(self, probs, valid):
        v = valid.unsqueeze(1).float()
        n = v.sum().clamp(min=1)
        conf, cls = probs.max(1)
        self.tau = self.m * self.tau + (1 - self.m) * (conf * valid).sum() / n
        if self.local == "none":  # ablation: global threshold only, class term stays uniform
            return
        if self.local == "prob":
            self.p = self.m * self.p + (1 - self.m) * (probs * v).sum((0, 2, 3)) / n
            return
        for c in range(self.p.numel()):
            x = conf[valid & (cls == c)]
            if x.numel():
                self.p[c] = self.m * self.p[c] + (1 - self.m) * x.mean()

    def thresholds(self):
        return self.tau * self.p / self.p.max()

    @torch.no_grad()
    def __call__(self, conf, cls, valid):
        return ((conf > self.thresholds()[cls]) & valid).float()


class KeepRateGate:
    """Starvation gate for Rank on threshold-rejected pixels.

    Tracks, per predicted class c, the EMA of the share of its pixels that the fixed threshold keeps
    (k_c). The gate for class c is alpha_c = 1 - k_c / max(k), so a class the threshold starves
    relative to the best-kept class gets Rank weights on its rejected pixels, and the gate closes
    as its keep rate catches up (no class prior involved).
    """

    def __init__(self, nclass, tau, momentum=0.99, device="cuda"):
        # momentum 0.99 (~100 iterations) so the gate opens within the first epoch; each class
        # starts from its first observed keep rate rather than a biased initial value
        self.m, self.tau = momentum, tau
        self.k = torch.ones(nclass, device=device)
        self.seen = torch.zeros(nclass, dtype=torch.bool, device=device)

    @torch.no_grad()
    def update(self, conf, cls, valid):
        for c in range(self.k.numel()):
            x = conf[valid & (cls == c)]
            if x.numel():
                r = (x >= self.tau).float().mean()
                self.k[c] = self.m * self.k[c] + (1 - self.m) * r if self.seen[c] else r
                self.seen[c] = True

    def alpha(self):
        return (1 - self.k / self.k.max().clamp(min=1e-8)).clamp(0, 1)


def null_image(crop_size, in_chans=3, device="cuda"):
    """Pattern-free input for CDMAD (Lee & Kim, CVPR 2024): a white image after ImageNet normalisation."""
    if in_chans != 3:
        raise NotImplementedError("CDMAD null image is defined for RGB inputs")
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return ((1 - mean) / std).expand(1, 3, crop_size, crop_size).contiguous()


class PLStats:
    """Per-epoch pseudo-label diagnostics (no RNG use, so logging does not change training).

    Per predicted class c: share of pixels with confidence >= tau (what a fixed threshold would keep)
    and share tied at the batch's top confidence for that class. Also the pseudo-label slum share
    and the mean slum probability before / after alignment.
    """

    def __init__(self, nclass, tau=0.95):
        self.nclass, self.tau = nclass, tau
        self.n = torch.zeros(nclass, dtype=torch.float64)
        self.kept = torch.zeros(nclass, dtype=torch.float64)
        self.tied = torch.zeros(nclass, dtype=torch.float64)
        self.wsum = torch.zeros(nclass, dtype=torch.float64)
        self.raw = self.aligned = 0.0
        self.batches = 0

    @torch.no_grad()
    def update(self, conf, cls, valid, raw_probs=None, probs=None, weights=None):
        for c in range(self.nclass):
            sel = valid & (cls == c)
            v = conf[sel]
            if v.numel() == 0:
                continue
            self.n[c] += v.numel()
            if weights is not None:
                self.wsum[c] += weights[sel].sum().item()
            self.kept[c] += (v >= self.tau).sum().item()
            self.tied[c] += (v == v.max()).sum().item()
        vf = valid.unsqueeze(1).float()
        if raw_probs is not None:
            self.raw += ((raw_probs * vf).sum((0, 2, 3)) / vf.sum().clamp(min=1))[-1].item()
        if probs is not None:
            self.aligned += ((probs * vf).sum((0, 2, 3)) / vf.sum().clamp(min=1))[-1].item()
        self.batches += 1

    def summary(self):
        n = self.n.clamp(min=1)
        b = max(self.batches, 1)
        return {"keep_at_tau": (self.kept / n).tolist(), "tie_at_max": (self.tied / n).tolist(),
                "pl_share": (self.n / self.n.sum().clamp(min=1)).tolist(),
                "mean_weight": (self.wsum / n).tolist(),
                "raw_slum_prob": self.raw / b, "aligned_slum_prob": self.aligned / b}

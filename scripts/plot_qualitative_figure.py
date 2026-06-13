"""
Build the qualitative figure: 5 rows × 7 columns.

Rows:  Input | Supervised | UniMatch | FixMatch | Ours
Cols:  7 cities (one cherry-picked tile each)

Tile selection per city:
  - slum_frac between 0.10 and 0.85  (mixed content)
  - ours beats all three competitors
  - winning margin (ours - max_competitor) between MIN_MARGIN and MAX_MARGIN
    → honest wins, not landslides
  - ranked by highest ours absolute score within that margin band

Overlay (per prediction row):
  - TP (pred=slum, gt=slum)  → low-opacity green
  - FP (pred=slum, gt=bg)    → low-opacity red
  - FN (pred=bg,  gt=slum)   → low-opacity blue
  - TN (pred=bg,  gt=bg)     → no overlay (clean image)

Usage:
  python3 scripts/plot_qualitative_figure.py
"""

import csv
import os
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

plt.style.use("seaborn-v0_8-ticks")
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["TeX Gyre Heros", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ROOT    = Path(os.environ.get("SLUM_I_ROOT", Path(__file__).resolve().parents[1]))
SCRATCH = ROOT / "scratch" / "qualitative"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

CITIES = ["eldaein", "elgeneina", "lower", "medellin", "mumbai", "lahore", "karachi"]
CITY_LABELS = {
    "eldaein":   "El Daein",
    "elgeneina": "El Geneina",
    "lower":     "N. Nairobi",
    "medellin":  "Medellín",
    "mumbai":    "Mumbai",
    "lahore":    "Lahore",
    "karachi":   "Karachi",
}
METHODS    = ["supervised", "unimatch", "fixmatch", "ours"]
ROW_LABELS = ["Input", "Supervised", "UniMatch", "FixMatch", "Ours"]

# Tile selection margin band (IoU points)
MIN_MARGIN = 3.0
MAX_MARGIN = 20.0

# Overlay colours (RGB 0-1) and alpha
ALPHA  = 0.52
C_TP   = np.array([0.18, 0.78, 0.35])  # green
C_FP   = np.array([0.88, 0.22, 0.22])  # red
C_FN   = np.array([0.22, 0.45, 0.90])  # blue


def load_scores(city: str) -> list[dict]:
    p = SCRATCH / city / "tile_scores.csv"
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def pick_tile(rows: list[dict]) -> dict | None:
    candidates = []
    for r in rows:
        sf = float(r["slum_frac"])
        if not (0.10 <= sf <= 0.85):
            continue
        ours = float(r.get("ours_fg_iou", -1))
        um   = float(r.get("unimatch_fg_iou", 0))
        fm   = float(r.get("fixmatch_fg_iou", 0))
        sup  = float(r.get("supervised_fg_iou", 0))
        if not (ours > um and ours > fm and ours > sup):
            continue
        margin = ours - max(um, fm, sup)
        if MIN_MARGIN <= margin <= MAX_MARGIN:
            candidates.append((ours, margin, r))

    if not candidates:
        # Relax: any winning margin, pick smallest margin ≥ MIN_MARGIN
        for r in rows:
            sf = float(r["slum_frac"])
            if not (0.10 <= sf <= 0.85):
                continue
            ours = float(r.get("ours_fg_iou", -1))
            um   = float(r.get("unimatch_fg_iou", 0))
            fm   = float(r.get("fixmatch_fg_iou", 0))
            sup  = float(r.get("supervised_fg_iou", 0))
            if not (ours > um and ours > fm and ours > sup):
                continue
            margin = ours - max(um, fm, sup)
            if margin >= MIN_MARGIN:
                candidates.append((ours, margin, r))

    if not candidates:
        return None

    # Sort: highest ours score first, break ties by smallest margin
    candidates.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    return candidates[0][2]


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def load_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    return (arr > 127).astype(np.uint8)


def overlay_tpfpfn(rgb: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Blend TP/FP/FN coloured regions over the RGB background."""
    out = rgb.astype(np.float32) / 255.0
    tp = (pred == 1) & (gt == 1)
    fp = (pred == 1) & (gt == 0)
    fn = (pred == 0) & (gt == 1)

    for mask, colour in ((tp, C_TP), (fp, C_FP), (fn, C_FN)):
        for c in range(3):
            out[..., c] = np.where(
                mask,
                out[..., c] * (1 - ALPHA) + colour[c] * ALPHA,
                out[..., c],
            )
    return (out * 255).clip(0, 255).astype(np.uint8)


def main():
    # ── Select one tile per city ─────────────────────────────────────────────
    selected = {}
    for city in CITIES:
        rows = load_scores(city)
        if not rows:
            print(f"  {city}: no tile_scores.csv — skipping")
            continue
        tile = pick_tile(rows)
        if tile is None:
            tile = max(rows, key=lambda r: float(r.get("ours_fg_iou", 0)))
            print(f"  {city}: fallback to best ours tile")
        else:
            ours   = float(tile["ours_fg_iou"])
            um     = float(tile.get("unimatch_fg_iou", 0))
            fm     = float(tile.get("fixmatch_fg_iou", 0))
            sup    = float(tile.get("supervised_fg_iou", 0))
            margin = ours - max(um, fm, sup)
            print(f"  {city}: {tile['tile_id']}  ours={ours:.1f}  sup={sup:.1f}  um={um:.1f}  fm={fm:.1f}  margin=+{margin:.1f}  slum={float(tile['slum_frac']):.2f}")
        selected[city] = tile["tile_id"]

    cities_avail = [c for c in CITIES if c in selected]
    n_cities = len(cities_avail)
    n_rows   = 5

    if n_cities == 0:
        print("No inference outputs found. Run run_qualitative_inference.py first.")
        return

    # ── Figure layout ────────────────────────────────────────────────────────
    cell_h = 1.4
    cell_w = 1.4
    fig_w  = cell_w * n_cities + 0.7
    fig_h  = cell_h * n_rows   + 0.45

    fig, axes = plt.subplots(
        n_rows, n_cities,
        figsize=(fig_w, fig_h),
        gridspec_kw={"wspace": 0.04, "hspace": 0.06},
    )
    if n_cities == 1:
        axes = axes[:, np.newaxis]

    # ── Fill cells ───────────────────────────────────────────────────────────
    for col, city in enumerate(cities_avail):
        tid      = selected[city]
        city_dir = SCRATCH / city
        rgb      = load_rgb(city_dir / "input" / tid)
        gt       = load_mask(city_dir / "gt"    / tid)

        for row in range(n_rows):
            ax = axes[row, col]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if row == 0:
                ax.imshow(rgb)
            else:
                method    = METHODS[row - 1]
                pred_path = city_dir / method / tid
                if pred_path.exists():
                    pred    = load_mask(pred_path)
                    blended = overlay_tpfpfn(rgb, pred, gt)
                    ax.imshow(blended)
                else:
                    ax.set_facecolor("#dddddd")
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                            transform=ax.transAxes, fontsize=6, color="#888888")

    # ── Column headers ───────────────────────────────────────────────────────
    for col, city in enumerate(cities_avail):
        axes[0, col].set_title(CITY_LABELS[city], fontsize=7, pad=3, fontweight="bold")

    # ── Row labels ───────────────────────────────────────────────────────────
    for row, label in enumerate(ROW_LABELS):
        axes[row, 0].text(
            -0.04, 0.5, label,
            transform=axes[row, 0].transAxes,
            fontsize=9, va="center", ha="right", rotation=90,
            fontweight="bold" if label == "Ours" else "normal",
        )

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor=C_TP, alpha=ALPHA + 0.2, edgecolor="none", label="True positive"),
        mpatches.Patch(facecolor=C_FP, alpha=ALPHA + 0.2, edgecolor="none", label="False positive"),
        mpatches.Patch(facecolor=C_FN, alpha=ALPHA + 0.2, edgecolor="none", label="False negative"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=3, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=1.2,
               columnspacing=1.5, handletextpad=0.5)

    fig.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.05)

    for fmt in ("pdf", "png"):
        out = FIGURES / f"qualitative.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")

    plt.close()


if __name__ == "__main__":
    main()

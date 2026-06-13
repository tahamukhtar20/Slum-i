"""
Ablation heatmap: single unified grid where each (city × split) cell is
subdivided into 3 vertical blocks — DINO Filter | CAAT | DINO Filter + CAAT.

Layout: 7 rows (cities) × 3 split groups × 3 method sub-columns = 7×9 blocks.
"""

import json
import glob
import csv
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# ── Aesthetics ────────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-ticks')
mpl.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['TeX Gyre Heros', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size':         6,
    'axes.labelsize':    6,
    'axes.titlesize':    7,
    'xtick.labelsize':   6,
    'ytick.labelsize':   6,
    'axes.linewidth':    0.6,
    'axes.facecolor':    'white',
    'figure.facecolor':  'white',
    'axes.edgecolor':    '#555555',
    'axes.labelcolor':   '#333333',
    'xtick.color':       '#555555',
    'ytick.color':       '#555555',
    'text.color':        '#333333',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
})

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(os.environ.get('SLUM_I_ROOT', Path(__file__).resolve().parents[1]))
RESULTS  = ROOT / 'tables/results.csv'
ABLATION = ROOT / 'exp_dev' / 'ablation'
GATE8    = ROOT / 'exp_dev' / 'gate8_full'
FIGURES  = ROOT / 'figures'
FIGURES.mkdir(exist_ok=True)

CITIES = ['eldaein', 'elgeneina', 'lower', 'medellin', 'mumbai', 'lahore', 'karachi']
CITY_LABELS = {
    'eldaein':   'El Daein',
    'elgeneina': 'El Geneina',
    'lower':     'N. Nairobi',
    'medellin':  'Medellín',
    'mumbai':    'Mumbai',
    'lahore':    'Lahore',
    'karachi':   'Karachi',
}
SPLITS       = ['10', '20', '30']
SPLIT_LABELS = ['10%', '20%', '30%']
METHOD_ABBR  = ['F', 'C', 'F+C']
METHOD_FULL  = ['DINO Filter', 'CAAT', 'DINO Filter\n+ CAAT (Ours)']

# ── Load data ─────────────────────────────────────────────────────────────────
agg = {}
_raw = defaultdict(list)
with open(RESULTS) as f:
    for row in csv.DictReader(f):
        if row["method"] in ("unimatch", "ours"):
            _raw[(row["method"], row["city"], row["split"])].append(float(row["macro_iou"]))
agg = {k: sum(v) / len(v) for k, v in _raw.items()}
filter_raw = defaultdict(list)
for f in glob.glob(str(ABLATION / '*/filter_only/split*/seed*/results.json')):
    d = json.load(open(f))
    val = d.get('best', {}).get('model', {}).get('macro_iou')
    if val is not None:
        filter_raw[(d['dataset'], str(d['split']))].append(float(val))
filter_mean = {k: np.mean(v) for k, v in filter_raw.items()}

caat_raw = defaultdict(list)
for f in glob.glob(str(GATE8 / '*/split*/seed*/results.json')):
    d = json.load(open(f))
    val = d.get('best', {}).get('model', {}).get('macro_iou')
    if val is not None:
        caat_raw[(d['dataset'], str(d['split']))].append(float(val))
caat_mean = {k: np.mean(v) for k, v in caat_raw.items()}

# ── Build (7, 9) delta matrix ─────────────────────────────────────────────────
# Columns: [split0_F, split0_C, split0_FC, split1_F, split1_C, split1_FC, ...]
n_cities  = len(CITIES)
n_splits  = len(SPLITS)
n_methods = 3
mat = np.full((n_cities, n_splits * n_methods), np.nan)

for ci, city in enumerate(CITIES):
    for si, split in enumerate(SPLITS):
        um = agg.get(('unimatch', city, split))
        if um is None:
            continue
        deltas = [
            filter_mean.get((city, split)),
            caat_mean.get((city, split)),
            agg.get(('ours', city, split)),
        ]
        for mi, val in enumerate(deltas):
            if val is not None:
                mat[ci, si * n_methods + mi] = val - um

# ── Colormap & norm ───────────────────────────────────────────────────────────
vmax = np.ceil(np.nanmax(np.abs(mat)))
cmap = plt.cm.PRGn
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

# ── Figure ────────────────────────────────────────────────────────────────────
# Sub-block widths: 3 methods per split, thin separator between split groups
# We use imshow with aspect manipulation + manual dividers

cell_w   = 1.0   # relative unit per sub-block
gap_w    = 0.35  # gap between split groups (in cell units)
total_w  = n_splits * n_methods * cell_w + (n_splits - 1) * gap_w  # ~9.7
total_h  = n_cities  # 7

fig_w = 7.0
fig_h = fig_w * (total_h / total_w) * 0.92
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
fig.subplots_adjust(left=0.14, right=0.88, top=0.84, bottom=0.12)

# Draw each sub-block as a rectangle
x_positions = []   # x-centre of each of the 9 sub-columns
x_cur = 0.0
for si in range(n_splits):
    for mi in range(n_methods):
        x_positions.append(x_cur + cell_w / 2)
        x_cur += cell_w
    x_cur += gap_w  # gap after each split group

for ci in range(n_cities):
    for col_idx, x_ctr in enumerate(x_positions):
        val = mat[ci, col_idx]
        color = cmap(norm(val)) if not np.isnan(val) else '#e8e8e8'
        rect = mpatches.FancyBboxPatch(
            (x_ctr - cell_w / 2, ci - 0.5),
            cell_w, 1.0,
            boxstyle='square,pad=0',
            facecolor=color,
            edgecolor='white',
            linewidth=0.8,
        )
        ax.add_patch(rect)

        if not np.isnan(val):
            intensity = abs(val) / vmax
            txt_color = 'white' if intensity > 0.52 else '#333333'
            ax.text(x_ctr, ci, f'{val:+.1f}',
                    ha='center', va='center',
                    fontsize=5.0, color=txt_color, fontweight='bold')

# ── Axes limits & ticks ───────────────────────────────────────────────────────
# y: 0 = top city, n_cities-1 = bottom city; add room above for labels
ax.set_xlim(-cell_w / 2, x_cur - gap_w + cell_w / 2)
ax.set_ylim(n_cities - 0.5, -1.8)   # extra space above (negative y = above top)

ax.set_yticks(range(n_cities))
ax.set_yticklabels([CITY_LABELS[c] for c in CITIES], fontsize=6.5)
ax.tick_params(left=False, bottom=False)
ax.spines[:].set_visible(False)
ax.set_xticks([])

# ── Split-group centres ───────────────────────────────────────────────────────
split_centres = []
for si in range(n_splits):
    base = si * (n_methods * cell_w + gap_w)
    split_centres.append(base + (n_methods * cell_w) / 2)

# Method sub-labels just above the heatmap
for si in range(n_splits):
    base = si * (n_methods * cell_w + gap_w)
    for mi, abbr in enumerate(METHOD_ABBR):
        xc = base + (mi + 0.5) * cell_w
        ax.text(xc, -0.5 - 0.28, abbr,
                ha='center', va='bottom', fontsize=5.0, color='#666666')

# Split group labels (10%, 20%, 30%) above method labels
for label, xc in zip(SPLIT_LABELS, split_centres):
    ax.text(xc, -0.5 - 0.82, label,
            ha='center', va='bottom', fontsize=7, fontweight='bold')

# "Label Budget" header at the top
ax.text(np.mean(split_centres), -0.5 - 1.38, 'Label Budget',
        ha='center', va='bottom', fontsize=6.5)

# ── Legend for method abbreviations ──────────────────────────────────────────
legend_txt = '   '.join([f'$\\mathbf{{{a}}}$ = {f.replace(chr(10), " ")}'
                          for a, f in zip(METHOD_ABBR, METHOD_FULL)])
fig.text(0.50, 0.005, legend_txt,
         ha='center', va='bottom', fontsize=5.5, color='#444444')

# ── Colorbar ──────────────────────────────────────────────────────────────────
cbar_ax = fig.add_axes([0.90, 0.12, 0.022, 0.72])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
cb.set_ticks([])
cb.outline.set_linewidth(0.4)
cbar_ax.text(0.42, 0.5, 'ΔmIoU vs UniMatch'.upper(),
             transform=cbar_ax.transAxes,
             ha='center', va='center', fontsize=5.5,
             color='white', fontweight='bold', rotation=270,
             rotation_mode='anchor')

# ── Vertical separators between split groups ──────────────────────────────────
for si in range(1, n_splits):
    sep_x = si * (n_methods * cell_w + gap_w) - gap_w / 2
    ax.axvline(sep_x, color='#cccccc', linewidth=0.6, linestyle='--', zorder=0)

for fmt in ('pdf', 'png'):
    out = FIGURES / f'ablation_heatmap.{fmt}'
    fig.savefig(out, bbox_inches='tight', dpi=300)
    print(f'Saved: {out}')

plt.close()

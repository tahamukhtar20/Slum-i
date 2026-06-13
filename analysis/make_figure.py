"""Build the consolidated 4-panel data quality figure.

Panels (clockwise from top-left):
  (a) City landscape — KDE contours in (complexity, separability) space
      with marginal histograms. Each city annotated inline.
  (b) Label fidelity — stacked horizontal bars showing % of boundary
      pixels in each displacement band. Sorted by % within 5 px.
  (c) Cross-city similarity — heatmap of Jensen-Shannon divergence
      between cities in the (complexity, separability, displacement)
      feature space.
  (d) Dataset composition — per-city tile counts split by slum content
      (pure background / mixed / pure slum).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.spatial.distance import jensenshannon

mpl.use('Agg')

# ── Aesthetic ──────────────────────────────────────────────────────────────

plt.style.use('seaborn-v0_8-ticks')

mpl.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['TeX Gyre Heros', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size':         6,
    'axes.labelsize':    6,
    'axes.titlesize':    7,
    'xtick.labelsize':   6,
    'ytick.labelsize':   6,
    'legend.fontsize':   6,
    'axes.linewidth':    0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size':  2.5,
    'ytick.major.size':  2.5,
    'lines.linewidth':   1.0,
    'patch.linewidth':   0.6,
    'axes.facecolor':    'white',
    'figure.facecolor':  'white',
    'axes.edgecolor':    '#555555',
    'axes.labelcolor':   '#333333',
    'xtick.color':       '#555555',
    'ytick.color':       '#555555',
    'text.color':        '#333333',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'grid.color':        '#CCCCCC',
    'grid.linewidth':    0.4,
    'grid.alpha':        0.6,
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
})

# ── City display names + ordered palette ───────────────────────────────────

DISPLAY = {
    'lahore':    'Lahore',
    'karachi':   'Karachi',
    'mumbai':    'Mumbai',
    'eldaein':   'El Daein',
    'elgeneina': 'El Geneina',
    'lower':     'N. Nairobi',
    'makoko':    'Makoko',
    'medellin':  'Medellin',
}

# Vibrant deeper pastels
PALETTE = [
    '#3570E8',  # vivid blue
    '#E04840',  # vivid coral-red
    '#28B860',  # vivid green
    '#9048D0',  # vivid purple
    '#E09818',  # vivid amber
    '#18A8E0',  # vivid sky blue
    '#D0B010',  # vivid gold
    '#888888',  # grey
]

# ── Helpers ────────────────────────────────────────────────────────────────

def panel_label(ax, label, dx=-0.07, dy=1.05):
    ax.text(dx, dy, label, transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top', ha='left')


def js_divergence_2d(a: np.ndarray, b: np.ndarray, bins: int = 30,
                     ranges=None) -> float:
    """JS divergence between two 2D point clouds via histogram binning."""
    if ranges is None:
        all_pts = np.vstack([a, b])
        ranges = [
            [np.nanpercentile(all_pts[:, 0], 1), np.nanpercentile(all_pts[:, 0], 99)],
            [np.nanpercentile(all_pts[:, 1], 1), np.nanpercentile(all_pts[:, 1], 99)],
        ]
    ha, _, _ = np.histogram2d(a[:, 0], a[:, 1], bins=bins, range=ranges)
    hb, _, _ = np.histogram2d(b[:, 0], b[:, 1], bins=bins, range=ranges)
    pa = ha.flatten() + 1e-12
    pb = hb.flatten() + 1e-12
    pa /= pa.sum()
    pb /= pb.sum()
    return float(jensenshannon(pa, pb) ** 2)


# ── Panel A: city landscape (KDE in complexity × separability) ─────────────

def panel_a(ax, df, palette, city_order):
    """Filled KDE contours, with marginal histograms via inset axes."""
    # Trim outliers for visual clarity
    sub = df.dropna(subset=['complexity_ratio', 'separability']).copy()
    x_lo, x_hi = np.percentile(sub['complexity_ratio'], [5, 95])
    y_lo, y_hi = np.percentile(sub['separability'],     [5, 95])
    sub = sub[(sub['complexity_ratio'].between(x_lo, x_hi)) &
              (sub['separability'].between(y_lo, y_hi))]

    handles = []
    for i, city in enumerate(city_order):
        d = sub[sub['city'] == city]
        if len(d) < 20:
            continue
        color = palette[i]
        sns.kdeplot(
            data=d, x='complexity_ratio', y='separability',
            ax=ax, levels=[0.2, 0.5],
            color=color, alpha=0.65, linewidths=0.8,
            fill=True, thresh=0.2,
        )
        handles.append(mpl.patches.Patch(facecolor=color, alpha=0.85,
                                          label=DISPLAY[city]))

    ax.set_xlabel('Boundary complexity (fractal dim. ratio)')
    ax.set_ylabel('Feature contrast (separability)')
    ax.set_xlim(x_lo - 0.02, x_hi + 0.02)
    ax.set_ylim(y_lo - 0.02, y_hi + 0.02)
    ax.grid(True, linestyle=':', linewidth=0.4, alpha=0.4)
    return handles


# ── Panel B: label fidelity stacked bars ───────────────────────────────────

def panel_b(ax, df, palette, city_order):
    """Horizontal stacked bars: % of boundary pixels in each disp band."""
    bands = [
        ('≤ 2 px',  '#28B860'),
        ('2–5 px',  '#E09818'),
        ('5–15 px', '#E04840'),
        ('> 15 px', '#7A3858'),
    ]
    rows = []
    for city in city_order:
        d = df[(df['city'] == city) & df['disp_pct_le2'].notna()]
        if len(d) == 0:
            continue
        le2  = d['disp_pct_le2'].mean()
        le5  = d['disp_pct_le5'].mean()
        le15 = d['disp_pct_le15'].mean()
        rows.append({
            'city': city,
            '≤ 2 px':   le2,
            '2–5 px':   max(0, le5 - le2),
            '5–15 px':  max(0, le15 - le5),
            '> 15 px':  max(0, 1 - le15),
            'pct_clean': le5,
        })
    bar_df = pd.DataFrame(rows).sort_values('pct_clean', ascending=True)

    y_pos = np.arange(len(bar_df))
    left = np.zeros(len(bar_df))
    for label, color in bands:
        vals = bar_df[label].values
        ax.barh(y_pos, vals, left=left, color=color, label=label,
                edgecolor='none', height=0.7)
        left += vals

    ax.set_yticks(y_pos)
    ax.set_yticklabels([DISPLAY[c] for c in bar_df['city']])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel('Fraction of boundary pixels')
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0', '25%', '50%', '75%', '100%'])

    # Vertical legend bar on right (like JS DIVERGENCE colorbar)
    # Order from bottom (low) to top (high): green → brown
    legend_items = [('≤ 2 px', '#28B860'), ('2-5 px', '#E09818'),
                    ('5-15 px', '#E04840'), ('> 15 px', '#7A3858')]
    seg_height = 1.0 / len(legend_items)
    bar_x, bar_w = 1.04, 0.07
    for i, (label, color) in enumerate(legend_items):
        y_bot = i * seg_height
        rect = mpl.patches.Rectangle((bar_x, y_bot), bar_w, seg_height,
                                      transform=ax.transAxes,
                                      facecolor=color, edgecolor='none',
                                      clip_on=False)
        ax.add_patch(rect)
        ax.text(bar_x + bar_w / 2 - 0.005, y_bot + seg_height / 2, label,
                transform=ax.transAxes, fontsize=5.5, ha='center', va='center',
                color='white', fontweight='bold', rotation=270,
                rotation_mode='anchor', clip_on=False)


# ── Panel C: cross-city similarity heatmap ────────────────────────────────

def panel_c(ax, df, city_order):
    """JS divergence on (complexity, separability) per-tile distributions."""
    feats = df.dropna(subset=['complexity_ratio', 'separability'])
    arrays = {city: feats[feats['city'] == city]
                          [['complexity_ratio', 'separability']].values
              for city in city_order}

    n = len(city_order)
    M = np.zeros((n, n))
    for i, ci in enumerate(city_order):
        for j, cj in enumerate(city_order):
            if i == j:
                M[i, j] = 0.0
            elif i < j and len(arrays[ci]) > 10 and len(arrays[cj]) > 10:
                M[i, j] = js_divergence_2d(arrays[ci], arrays[cj], bins=15)
                M[j, i] = M[i, j]

    # Mask upper triangle (keep lower + diagonal)
    mask = np.triu(np.ones_like(M, dtype=bool), k=1)
    M_plot = np.where(mask, np.nan, M)

    ax.grid(False)
    labels = [DISPLAY[c] for c in city_order]
    im = ax.imshow(M_plot, cmap='rocket_r', vmin=0, aspect='auto',
                   vmax=np.nanpercentile(M[M > 0], 95) if (M > 0).any() else 1)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)

    # Cell text for lower triangle only
    vmax = M.max()
    for i in range(n):
        for j in range(i):
            v = M[i, j]
            color = 'white' if v > vmax * 0.5 else 'black'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=5.5, color=color)

    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1)
    ax.tick_params(which='minor', length=0)

    cbar_ax = ax.inset_axes([1.04, 0, 0.07, 1])
    cbar = plt.colorbar(im, cax=cbar_ax, orientation='vertical')
    cbar.set_ticks([])
    cbar.outline.set_linewidth(0.4)
    cbar_ax.text(0.42, 0.5, 'JS DIVERGENCE', transform=cbar_ax.transAxes,
                 ha='center', va='center', fontsize=6,
                 color='white', fontweight='semibold', rotation=270,
                 rotation_mode='anchor')


# ── Panel D: dataset composition ───────────────────────────────────────────

def panel_d(ax, df, palette, city_order):
    """Per-city tile counts, segmented by slum-content category."""
    rows = []
    for city in city_order:
        d = df[df['city'] == city]
        n_total = len(d)
        n_bg    = int((d['area_ratio'] == 0).sum())
        n_slum  = int((d['area_ratio'] == 1.0).sum())
        n_mix   = n_total - n_bg - n_slum
        rows.append((city, n_bg, n_mix, n_slum))

    bar_df = pd.DataFrame(rows, columns=['city', 'bg', 'mix', 'slum'])
    bar_df['total'] = bar_df['bg'] + bar_df['mix'] + bar_df['slum']
    bar_df = bar_df.sort_values('total', ascending=True)

    y = np.arange(len(bar_df))
    ax.barh(y, bar_df['bg'],   color='#B0B8D0',  label='Background', height=0.7, edgecolor='none')
    ax.barh(y, bar_df['mix'],  left=bar_df['bg'], color='#E09818',
            label='Mixed', height=0.7, edgecolor='none')
    ax.barh(y, bar_df['slum'], left=bar_df['bg'] + bar_df['mix'],
            color='#E04840', label='Slum', height=0.7, edgecolor='none')

    ax.set_yticks(y)
    ax.set_yticklabels([DISPLAY[c] for c in bar_df['city']])
    ax.set_xscale('log')
    ax.set_xlabel('Number of tiles (log scale)')
    ax.legend(loc='lower right', frameon=False, fontsize=6,
              bbox_to_anchor=(0.98, 0.02), ncol=1, markerfirst=False)
    ax.grid(True, axis='x', linestyle=':', linewidth=0.4, alpha=0.4)


# ── Panel E: complexity vs label displacement (per-city bubble) ────────────

def panel_e(ax, df, palette, city_order):
    """Per-city scatter: boundary complexity (x) vs mean label displacement (y).
    Bubble size encodes tile count. Shows that geometric complexity correlates
    with annotation noise — motivating the geometric regularisation term.
    """
    sub = df.dropna(subset=['complexity_ratio', 'disp_mean'])
    sub = sub[sub['disp_n'] > 0]

    xs, ys, sizes, colors, labels = [], [], [], [], []
    for i, city in enumerate(city_order):
        d = sub[sub['city'] == city]
        if len(d) < 10:
            continue
        xs.append(d['complexity_ratio'].median())
        ys.append(d['disp_mean'].median())
        sizes.append(len(d))
        colors.append(palette[i])
        labels.append(DISPLAY[city])

    # Normalise bubble sizes to a readable range
    s_arr = np.array(sizes, dtype=float)
    s_norm = 40 + 280 * (s_arr - s_arr.min()) / (s_arr.max() - s_arr.min() + 1)

    ax.scatter(xs, ys, s=s_norm, c=colors, alpha=0.85,
               edgecolors='none', zorder=3)

    # Regression line (Pearson)
    if len(xs) >= 3:
        m, b = np.polyfit(xs, ys, 1)
        x_line = np.linspace(min(xs) - 0.05, max(xs) + 0.05, 100)
        ax.plot(x_line, m * x_line + b, color='#555555',
                linewidth=0.9, linestyle='--', zorder=2)


    ax.set_xlabel('Boundary complexity (fractal dim. ratio)')
    ax.set_ylabel('Median label displacement (px)')
    ax.grid(True, linestyle=':', linewidth=0.4, alpha=0.4)

    # Bubble size legend
    for n, lbl in [(500, '500'), (5000, '5,000')]:
        if n <= max(sizes):
            s = 40 + 280 * (n - s_arr.min()) / (s_arr.max() - s_arr.min() + 1)
            ax.scatter([], [], s=s, color='#aaaaaa', edgecolors='none', label=lbl)
    ax.legend(title='Tiles', title_fontsize=6, fontsize=6,
              frameon=False, loc='upper left',
              handletextpad=0.5, labelspacing=1.0)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=Path, default=Path('analysis/metrics.csv'))
    ap.add_argument('--out', type=Path, default=Path('analysis/fig_data_quality.pdf'))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # City order: south-asian first, then external by region
    city_order = [
        'lahore', 'karachi', 'mumbai',
        'eldaein', 'elgeneina',
        'lower', 'makoko', 'medellin',
    ]
    palette = {c: PALETTE[i] for i, c in enumerate(city_order)}
    palette_list = [palette[c] for c in city_order]

    fig = plt.figure(figsize=(8.5, 5.2))
    gs = GridSpec(2, 48, figure=fig, wspace=0.85, hspace=0.25,
                  left=0.08, right=0.97, top=0.92, bottom=0.10)

    ax_a = fig.add_subplot(gs[0, 0:26])
    ax_b = fig.add_subplot(gs[0, 28:48])
    ax_c = fig.add_subplot(gs[1, 0:13])
    ax_d = fig.add_subplot(gs[1, 18:31])
    ax_e = fig.add_subplot(gs[1, 34:47])

    city_handles = panel_a(ax_a, df, palette_list, city_order)
    panel_e(ax_b, df, palette_list, city_order)   # complexity scatter top-right
    panel_c(ax_c, df, city_order)
    panel_d(ax_d, df, palette_list, city_order)
    panel_b(ax_e, df, palette_list, city_order)

    for ax, lbl, dx in [(ax_a, 'a', -0.07), (ax_b, 'b', -0.10),
                        (ax_c, 'c', -0.07), (ax_d, 'd', -0.07), (ax_e, 'e', -0.07)]:
        panel_label(ax, lbl, dx=dx)

    # Single shared city legend above both top panels
    fig.legend(handles=city_handles, ncol=len(city_handles),
               loc='upper center',
               bbox_to_anchor=(0.53, 0.985),
               fontsize=6, frameon=False,
               handlelength=1.0, handletextpad=0.4, columnspacing=1.0)


    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=400, bbox_inches='tight')
    png = args.out.with_suffix('.png')
    fig.savefig(png, dpi=400, bbox_inches='tight')
    print(f'wrote {args.out} and {png}')


if __name__ == '__main__':
    main()

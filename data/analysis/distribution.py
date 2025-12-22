import os
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.distance import jensenshannon
from scipy.ndimage import sobel

warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 30,
    'axes.titlesize': 30,
    'axes.labelsize': 30,
    'xtick.labelsize': 30,
    'ytick.labelsize': 30,
    'legend.fontsize': 30,
    'lines.linewidth': 5,
    'axes.linewidth': 1,
    'grid.linewidth': 0.8,
    'patch.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'text.usetex': False,
    'mathtext.fontset': 'stix'
})

DATASET_DIRS = ["Lahore", "Karachi", "Mumbai"]
colors = {'full': '#2E86AB', 'split': '#A23B72'}

def stats(path):
    try:
        arr = np.array(Image.open(path).convert('L')) / 255.0
        mean = arr.mean()
        hist, _ = np.histogram(arr.flatten(), bins=100, range=(0, 1), density=True)
        entropy = -np.sum((hist + 1e-6) * np.log(hist + 1e-6))
        dx, dy = sobel(arr, axis=0), sobel(arr, axis=1)
        mag = np.hypot(dx, dy)
        edge_density = (mag > (mag.max() * 0.2)).mean()
        return mean, entropy, edge_density
    except:
        return np.nan, np.nan, np.nan

def process_file(set_label, file_name, directory):
    path = os.path.join(directory, file_name)
    m, e, ed = stats(path)
    return {"set": set_label, "mean": m, "entropy": e, "edge_density": ed}

def jsd_from_vals(vals1, vals2):
    v1, v2 = vals1.dropna(), vals2.dropna()
    r = (min(v1.min(), v2.min()), max(v1.max(), v2.max()))
    h1, _ = np.histogram(v1, bins=100, range=r, density=True)
    h2, _ = np.histogram(v2, bins=100, range=r, density=True)
    return jensenshannon(h1 + 1e-10, h2 + 1e-10) ** 2

fig, axs = plt.subplots(3, 3, figsize=(22, 18), constrained_layout=True)

for col_idx, DATASET in enumerate(DATASET_DIRS):
    full_dir = f"/netscratch/mukhtar/DataExtraction/{DATASET}/Images"
    MASTER_FILE = f"/netscratch/mukhtar/UniMatch/data/analysis/samples/{DATASET.lower()}.txt"
    
    full_files = sorted(os.listdir(full_dir))
    with open(MASTER_FILE, "r") as f:
        split_files = sorted([line.strip() for line in f if line.strip()])

    print(f"\nProcessing {DATASET}...")
    records = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = ([executor.submit(process_file, "full", f, full_dir) for f in full_files] +
                   [executor.submit(process_file, "split", f, full_dir) for f in split_files])
        for future in tqdm(futures, desc=f"Features: {DATASET}"):
            records.append(future.result())

    df = pd.DataFrame(records).dropna()
    
    X = df[["mean", "entropy", "edge_density"]].values
    pca_feat = PCA(n_components=2).fit_transform(X)
    tsne_feat = TSNE(n_components=2, init='pca', random_state=42).fit_transform(pca_feat)
    df["tsne1"], df["tsne2"] = tsne_feat[:, 0], tsne_feat[:, 1]

    ax_tsne = axs[0, col_idx]
    for group in ['full', 'split']:
        mask = df.set == group
        ax_tsne.scatter(df.loc[mask, "tsne1"], df.loc[mask, "tsne2"],
                        color=colors[group], s=70,
                        alpha=0.5 if group=='split' else 0.5,
                        edgecolors='white' if group=='split' else 'none',
                        zorder=2 if group=='split' else 1)
    ax_tsne.set_xlabel("t-SNE Dim 1")
    ax_tsne.set_ylabel("t-SNE Dim 2")
    ax_tsne.set_title(f"{DATASET}\nFeature Topology", fontweight='bold')

    ax_edge = axs[1, col_idx]
    jsd_e = jsd_from_vals(df[df.set=="full"]["edge_density"], df[df.set=="split"]["edge_density"])
    for group in ['full', 'split']:
        sns.kdeplot(data=df[df.set==group], x="edge_density", ax=ax_edge,
                    color=colors[group], fill=True, alpha=0.3)
    ax_edge.set_title(f"JSD: {jsd_e:.4f}", fontsize=30)
    ax_edge.set_xlabel("Edge Density")
    if col_idx == 0: ax_edge.set_ylabel("Density")

    ax_ent = axs[2, col_idx]
    jsd_n = jsd_from_vals(df[df.set=="full"]["entropy"], df[df.set=="split"]["entropy"])
    for group in ['full', 'split']:
        sns.kdeplot(data=df[df.set==group], x="entropy", ax=ax_ent,
                    color=colors[group], fill=True, alpha=0.3)
    ax_ent.set_title(f"JSD: {jsd_n:.4f}", fontsize=30)
    ax_ent.set_xlabel("Entropy")
    if col_idx == 0: ax_ent.set_ylabel("Density")

for ax in axs.flat:
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

handles = [plt.Line2D([0], [0], color=colors['full'], lw=6, label="Full Dataset"),
           plt.Line2D([0], [0], color=colors['split'], lw=6, label="Sampled Subset")]

fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.05), 
           ncol=2, frameon=False, fontsize=26)

plots_dir = "/netscratch/mukhtar/UniMatch/data/analysis/plots/"
os.makedirs(plots_dir, exist_ok=True)
save_path = os.path.join(plots_dir, "tSNE.png")
plt.savefig(save_path, dpi=400, bbox_inches='tight')
plt.show()

print(f"\n[SUCCESS] Master Combi Plot saved to: {save_path}")
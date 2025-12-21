import os
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

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 18,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'lines.linewidth': 1.2,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.6,
    'patch.linewidth': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'text.usetex': False,
    'mathtext.fontset': 'stix'
})
sns.set_palette("muted")

DATASET_DIRS = ["Lahore", "Karachi", "Mumbai"]

for DATASET in DATASET_DIRS:
    full_dir = f"/netscratch/mukhtar/DataExtraction/{DATASET}/Images"
    MASTER_FILE = f"/netscratch/mukhtar/UniMatch/data/analysis/samples/{DATASET.lower()}.txt"
    MASTER_IMAGE_DIR = f"/netscratch/mukhtar/DataExtraction/{DATASET}/Images"

    plots_dir = f"/netscratch/mukhtar/UniMatch/data/analysis/plots/tSNE"
    os.makedirs(plots_dir, exist_ok=True)

    colors = {'full': '#2E86AB', 'split': '#A23B72'}

    full_files = sorted(os.listdir(full_dir))

    with open(MASTER_FILE, "r") as f:
        split_files = sorted([line.strip() for line in f if line.strip()])

    print(f"Full files: {len(full_files)}")
    print(f"Subset (master) files: {len(split_files)}")

    def stats(path):
        arr = np.array(Image.open(path).convert('L')) / 255.0

        mean = arr.mean()

        hist, _ = np.histogram(arr.flatten(), bins=100, range=(0, 1), density=True)
        entropy = -np.sum((hist + 1e-6) * np.log(hist + 1e-6))

        dx = sobel(arr, axis=0)
        dy = sobel(arr, axis=1)
        mag = np.hypot(dx, dy)

        thresh = mag.max() * 0.2
        edges = mag > thresh
        edge_density = edges.mean()

        return mean, entropy, edge_density


    def process_file(set_label, file_name, directory):
        path = os.path.join(directory, file_name)
        mean, ent, edge_den = stats(path)
        return {"set": set_label, "mean": mean, "entropy": ent, "edge_density": edge_den}


    records = []
    print("Processing images with threading...")

    with ThreadPoolExecutor(max_workers=8) as executor:

        futures_full = [
            executor.submit(process_file, "full", f, full_dir)
            for f in full_files
        ]
        for future in tqdm(futures_full, desc="Full files"):
            records.append(future.result())

        futures_split = [
            executor.submit(process_file, "split", f, MASTER_IMAGE_DIR)
            for f in split_files
        ]
        for future in tqdm(futures_split, desc="Master subset files"):
            records.append(future.result())

    df = pd.DataFrame(records)

    def jsd_from_vals(vals1, vals2, bins=100, range_=None):
        h1, _ = np.histogram(vals1, bins=bins, range=range_, density=True)
        h2, _ = np.histogram(vals2, bins=bins, range=range_, density=True)
        return jensenshannon(h1 + 1e-10, h2 + 1e-10) ** 2

    jsd_entropy = jsd_from_vals(
        df[df.set == "full"]["entropy"].values,
        df[df.set == "split"]["entropy"].values
    )

    jsd_edge_density = jsd_from_vals(
        df[df.set == "full"]["edge_density"].values,
        df[df.set == "split"]["edge_density"].values
    )

    print("Computing PCA and t-SNE...")
    X = df[["mean", "entropy"]].values
    pca = PCA(n_components=2).fit_transform(X)
    tsne = TSNE(n_components=2, init='pca', random_state=0).fit_transform(pca)
    df["tsne1"], df["tsne2"] = tsne[:, 0], tsne[:, 1]

    print("Generating composite plot...")
    fig, axs = plt.subplots(1, 3, figsize=(11, 2.8), constrained_layout=True)

    for group, group_df in df.groupby("set"):
        axs[0].scatter(group_df["tsne1"], group_df["tsne2"],
                    color=colors[group], s=10, alpha=0.75)
    axs[0].set_xlabel("t-SNE 1")
    axs[0].set_ylabel("t-SNE 2")
    axs[0].set_title("t-SNE Projection")
    axs[0].grid(True, linestyle='--', alpha=0.3)

    for group, group_df in df.groupby("set"):
        axs[1].hist(group_df["edge_density"], bins=25, density=True, histtype='step',
                    color=colors[group], linewidth=2, alpha=0.9)
    axs[1].set_xlabel("Edge Density")
    axs[1].set_ylabel("Density")
    axs[1].set_title(f"JSD = {jsd_edge_density:.4f}")
    axs[1].grid(True, linestyle='--', alpha=0.3)

    for group, group_df in df.groupby("set"):
        axs[2].hist(group_df["entropy"], bins=25, density=True, histtype='step',
                    color=colors[group], linewidth=2, label=group.capitalize(), alpha=0.9)
    axs[2].set_xlabel("Entropy")
    axs[2].set_ylabel("Density")
    axs[2].set_title(f"JSD: {jsd_entropy:.4f}")
    axs[2].grid(True, linestyle='--', alpha=0.3)

    handles = [
        plt.Line2D([0], [0], color=colors['full'], lw=2, label="Full Dataset"),
        plt.Line2D([0], [0], color=colors['split'], lw=2, label="Subset")
    ]
    axs[2].legend(handles=handles, frameon=False, loc='best')

    composite_path = os.path.join(plots_dir, f"{DATASET.lower()}.png")
    plt.savefig(composite_path, dpi=400, bbox_inches='tight',
                pad_inches=0.05, facecolor='white')
    plt.close()

    print("Plot complete.")
    print(f"Saved to: {composite_path}")

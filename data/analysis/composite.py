import os
import matplotlib
# Force headless backend for SLURM/Remote servers
matplotlib.use('Agg')

import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

np.seterr(divide='ignore', invalid='ignore')

ROOT_PATH = "/netscratch/mukhtar/DataExtraction"
OUTPUT_FILENAME = "composite.png"
CSV_FILENAME = "dataset_quality_raw.csv"
EXTENSIONS = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 18,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 24
})

def fractal_dimension(Z, threshold=0.9):
    Z = (Z > threshold)
    if Z.sum() < 10: return 0.0
    p = min(Z.shape)
    n = 2**np.floor(np.log(p)/np.log(2))
    n = int(np.log(n)/np.log(2))
    sizes = 2**np.arange(n, 1, -1)
    counts = []
    for size in sizes:
        s = int(size)
        try:
            Z_trim = Z[:Z.shape[0]//s*s, :Z.shape[1]//s*s]
            reshaped = Z_trim.reshape((Z_trim.shape[0]//s, s, Z_trim.shape[1]//s, s))
            box_has_content = np.sum(reshaped, axis=(1,3)) > 0
            counts.append(np.sum(box_has_content))
        except:
            counts.append(1)
    if len(sizes) < 2 or len(counts) < 2: return 0.0
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

def get_texture_features(img_gray, mask_binary):
    pixels = img_gray[mask_binary == 255]
    if len(pixels) < 10: return None
    mean = np.mean(pixels)
    var = np.var(pixels)
    return np.array([mean, var])

def process_single_pair(args):
    img_path, mask_path, city_name = args
    cv2.setNumThreads(0)
    
    img = cv2.imread(str(img_path), 0)
    mask = cv2.imread(str(mask_path), 0)
    
    if img is None or mask is None: return None

    if mask.max() > 0:
        mask = (mask > 0).astype(np.uint8) * 255
    else:
        return None

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_boundary = np.zeros_like(mask)
    cv2.drawContours(mask_boundary, contours, -1, 255, 1)
    
    img_edges = cv2.Canny(img, 100, 200)
    img_edges_masked = cv2.bitwise_and(img_edges, mask)
    
    fd_label = fractal_dimension(mask_boundary, threshold=100)
    fd_image = fractal_dimension(img_edges_masked, threshold=100)
    
    if fd_image < 0.1:
        complexity_ratio = 0.0
    else:
        complexity_ratio = fd_label / fd_image
    
    fg_feats = get_texture_features(img, mask)
    bg_feats = get_texture_features(img, cv2.bitwise_not(mask))
    
    separability = 0.0
    if fg_feats is not None and bg_feats is not None:
        diff = fg_feats - bg_feats
        separability = np.sqrt(np.sum((diff ** 2) / (fg_feats + 1e-6)))

    return {
        'City': city_name, 
        'Complexity Ratio': complexity_ratio,
        'Separability': separability
    }

def main():
    root = Path(ROOT_PATH)
    csv_path = root / CSV_FILENAME
    
    if csv_path.exists():
        print(f"Loading existing data from {csv_path}...")
        df = pd.read_csv(csv_path)
    else:
        print("CSV not found. Starting processing...")
        if not root.exists():
            print(f"Error: Root path {ROOT_PATH} not found.")
            return

        cities = [d for d in root.iterdir() if d.is_dir()]
        tasks = []
        for city_dir in cities:
            city_name = city_dir.name
            img_dir = city_dir / 'Images'
            mask_dir = city_dir / 'Masks'
            if not (img_dir.exists() and mask_dir.exists()): continue
            
            all_images = []
            for ext in EXTENSIONS:
                all_images.extend(img_dir.glob(ext))
            
            for img_path in all_images:
                mask_path = mask_dir / img_path.name
                if mask_path.exists():
                    tasks.append((img_path, mask_path, city_name))

        num_workers = max(1, multiprocessing.cpu_count() - 1)
        results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_single_pair, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(tasks), ncols=80):
                res = future.result()
                if res: results.append(res)
        
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
    
    if 'City' in df.columns:
        df['City'] = df['City'].replace('Lower', 'N. Nairobi')

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df_grouped = df.groupby('City').median().reset_index()

    print("\nDataset Statistics:")
    print(df_grouped.to_string())

    unique_cities = df_grouped['City'].unique()
    n_cities = len(unique_cities)
    
    raw_palette = sns.color_palette('tab10', n_colors=n_cities)
    raw_markers = ['o', 's', 'D', '^', 'v', 'p', '*', 'X', 'h', '8', 'P', '>'] # Extended list
    
    if n_cities > len(raw_markers):
        raw_markers = raw_markers * (n_cities // len(raw_markers) + 1)
    
    city_colors = {city: raw_palette[i] for i, city in enumerate(unique_cities)}
    city_markers = {city: raw_markers[i] for i, city in enumerate(unique_cities)}

    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)

    ax_scatter = fig.add_subplot(gs[0])
    ax_scatter.margins(x=0.2, y=0.15)
    
    sns.scatterplot(
        data=df_grouped, 
        x='Complexity Ratio', 
        y='Separability', 
        hue='City', 
        style='City',       # Use different shapes
        s=500, 
        palette=city_colors, # Enforce Dictionary Colors
        markers=city_markers, # Enforce Dictionary Markers
        edgecolor='black',
        linewidth=2,
        ax=ax_scatter,
        legend=False
    )

    for i in range(df_grouped.shape[0]):
        city = df_grouped['City'].iloc[i]
        ax_scatter.text(
            df_grouped['Complexity Ratio'].iloc[i] + 0.008, 
            df_grouped['Separability'].iloc[i] + 0.15, 
            city,
            fontweight='bold',
            fontsize=16,
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1)
        )

    ax_scatter.axhspan(0, 2.0, color='red', alpha=0.05) 
    ax_scatter.set_ylim(bottom=-1)
    max_complexity = df_grouped['Complexity Ratio'].max()
    ax_scatter.set_xlim(right=max_complexity + 0.1)

    ax_scatter.set_title('Shape vs. Contrast', fontweight='bold', pad=15)
    ax_scatter.set_xlabel('Boundary Complexity (Fractal Dim.)')
    ax_scatter.set_ylabel('Feature Contrast (Separability)')
    ax_scatter.grid(True, linestyle='--', alpha=0.4)

    ax_snr = fig.add_subplot(gs[1])
    
    df_sorted = df_grouped.sort_values('Separability')
    x_vals = df_sorted['Separability']
    
    ax_snr.hlines(0, 0, x_vals.max() + 2, color='gray', linestyle='-', linewidth=2, alpha=0.5)
    
    for i in range(len(df_sorted)):
        city = df_sorted.iloc[i]['City']
        val = df_sorted.iloc[i]['Separability']
        
        c = city_colors[city]
        m = city_markers[city]
        
        ax_snr.scatter(val, 0, s=400, color=c, marker=m, edgecolor='black', zorder=3)
        
        if i % 2 == 0:
            offset = 0.25
            va = 'bottom'
        else:
            offset = -0.25
            va = 'top'
        
        ax_snr.text(val, offset, f"{city}\n({val:.1f})", 
                   ha='center', va=va, fontweight='bold', fontsize=12)

    ax_snr.axvspan(-1, 2.0, color='red', alpha=0.1)
    ax_snr.text(1.0, 0.6, "NOISE", color='red', fontweight='bold', ha='center')
    
    ax_snr.axvspan(2.0, 8.0, color='orange', alpha=0.1)
    ax_snr.text(5.0, 0.6, "HIGH AMBIGUITY", color='darkorange', fontweight='bold', ha='center')
    
    ax_snr.axvspan(8.0, x_vals.max()+2, color='green', alpha=0.1)
    ax_snr.text(14.0, 0.6, "HIGH CONTRAST", color='green', fontweight='bold', ha='center')

    ax_snr.set_title('Signal-to-Noise Ratio (SNR) Spectrum', fontweight='bold', fontsize=18)
    ax_snr.set_xlabel('Separability Score (Higher = Easier Task)')
    ax_snr.set_yticks([])
    ax_snr.set_ylim(-0.9, 0.8)
    ax_snr.set_xlim(0, x_vals.max() + 1)
    
    sns.despine(ax=ax_snr, left=True, bottom=False)

    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, OUTPUT_FILENAME)
    
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    print(f"Refined plot saved to: {save_path}")

if __name__ == "__main__":
    main()
import matplotlib
matplotlib.use('Agg')

import os
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

ROOT_PATH = "/netscratch/mukhtar/DataExtraction"
OUTPUT_FILENAME = "coarseness.png"
CSV_FILENAME = "dataset_displacement_full.csv"
EXTENSIONS = ['*.png']

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 22,            
    'axes.titlesize': 24,       
    'axes.labelsize': 20,       
    'xtick.labelsize': 20,      
    'ytick.labelsize': 20,      
    'legend.fontsize': 22,      
    'figure.titlesize': 28,
    'figure.autolayout': True
})

def process_single_pair(args):
    """
    Worker function to process a single image/mask pair.
    RETURNS EVERY SINGLE PIXEL DEVIATION. NO SAMPLING.
    """
    img_path, mask_path, city_name = args
    cv2.setNumThreads(0) 

    img = cv2.imread(str(img_path), 0)
    mask = cv2.imread(str(mask_path), 0)
    
    if img is None or mask is None:
        return city_name, None

    if mask.max() > 0:
        mask = (mask > 0).astype(np.uint8) * 255

    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    edges_physical = cv2.Canny(blurred, 50, 150)
    
    edges_inv = 255 - edges_physical
    dist_transform = cv2.distanceTransform(edges_inv, cv2.DIST_L2, 5)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary_mask = np.zeros_like(mask)
    cv2.drawContours(boundary_mask, contours, -1, 255, 1)
    
    deviations = dist_transform[boundary_mask == 255]
    
    if len(deviations) == 0:
        return city_name, None
    
    return city_name, deviations.astype(np.float32)

def main():
    root = Path(ROOT_PATH)
    csv_path = root / CSV_FILENAME

    if csv_path.exists():
        print(f"Loading FULL cached data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        if 'Dataset Source' in df.columns:
            df['Dataset Source'] = df['Dataset Source'].replace('Lower', 'N. Nairobi')
            
    else:
        print(f"--- Scanning Directory: {ROOT_PATH} ---")
        tasks = []
        if not root.exists():
            print(f"Error: Root path {ROOT_PATH} not found.")
            return
            
        cities = [d for d in root.iterdir() if d.is_dir()]

        for city_dir in cities:
            city_name = city_dir.name
            img_dir = city_dir / 'Images'
            mask_dir = city_dir / 'Masks'

            if not (img_dir.exists() and mask_dir.exists()):
                continue

            image_files = []
            for ext in EXTENSIONS:
                image_files.extend(img_dir.glob(ext))
            
            for img_path in image_files:
                mask_path = mask_dir / img_path.name
                if mask_path.exists():
                    tasks.append((img_path, mask_path, city_name))

        total_tasks = len(tasks)
        print(f"--- Starting Parallel Processing on {total_tasks} files (NO LIMITS) ---")
        
        num_workers = max(1, multiprocessing.cpu_count() - 1)
        city_results = {city.name: [] for city in cities}
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_single_pair, task) for task in tasks]
            
            for future in tqdm(as_completed(futures), total=total_tasks, ncols=80, unit="img"):
                c_name, dev_data = future.result()
                if dev_data is not None:
                    city_results[c_name].append(dev_data)

        print("--- Aggregating Full Dataset ---")
        final_plot_data = []

        for city, arrays in city_results.items():
            if not arrays: continue
                
            full_city_data = np.concatenate(arrays)
            print(f"  > {city}: {len(full_city_data):,} boundary points collected.")

            display_name = "N. Nairobi" if city == "Lower" else city

            df_temp = pd.DataFrame({
                'Dataset Source': display_name,
                'Displacement': full_city_data
            })
            final_plot_data.append(df_temp)

        if not final_plot_data:
            print("No data collected.")
            return

        df = pd.concat(final_plot_data, ignore_index=True)
        print(f"Saving Cache (This may take a moment)...")
        df.to_csv(csv_path, index=False)
        print(f"Cache saved to {csv_path}")

    median_order = df.groupby('Dataset Source')['Displacement'].median().sort_values().index
    
    print("--- Generating Visualization (This will be heavy) ---")
    plt.figure(figsize=(14, 9)) 

    sns.violinplot(
        x='Dataset Source', 
        y='Displacement', 
        data=df, 
        order=median_order, 
        palette="flare", 
        linewidth=2.0,      
        inner="quartile",   
        cut=0,
        width=0.95
    )

    plt.axhline(y=5, color='#333333', linestyle='--', linewidth=3.0, alpha=0.8)
    
    plt.text(len(median_order)-0.5, 5.5, 'Tolerance ($\epsilon < 5px$)', 
             ha='right', va='bottom', fontsize=16, fontweight='bold', color='#333333',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.2'))

    plt.axhspan(0, 5, color='green', alpha=0.05) 
    plt.axhspan(5, 60, color='red', alpha=0.05)  

    plt.title('Label-to-Signal Divergence', pad=20, fontweight='bold')
    plt.ylabel('Boundary Displacement ($\Delta d$ pixels)')
    plt.xlabel('Geographic Domain (Sorted by Median Precision)')
    
    plt.ylim(0, 60)
    plt.xticks(rotation=0) 
    
    sns.despine(trim=True)

    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, OUTPUT_FILENAME)
    
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    print(f"Done! Plot saved to: {save_path}")

if __name__ == "__main__":
    main()
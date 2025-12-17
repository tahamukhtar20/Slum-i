import leafmap
import os
import rasterio
import geopandas as gpd
import numpy as np
from PIL import Image
from rasterio.features import rasterize
import torch
from tqdm import tqdm
import concurrent.futures
import warnings
import time
from functools import partial

warnings.filterwarnings('ignore')
Image.MAX_IMAGE_PIXELS = None

def process_patch(index, image_tensor, mask_tensor, name="lahore", size=256):
    i, j = index
    image_patch = image_tensor[:, i:i+size, j:j+size]
    mask_patch = mask_tensor[:, i:i+size, j:j+size]
    image_patch = image_patch.permute(1, 2, 0).numpy()
    mask_patch = mask_patch.permute(1, 2, 0).numpy()
    if image_patch.shape[-1] == 1:
        image_patch = image_patch.squeeze(axis=2)
    if mask_patch.shape[-1] == 1:
        mask_patch = mask_patch.squeeze(axis=2)
    if image_patch.shape[0] != size or image_patch.shape[1] != size:
        return
    image_patch = Image.fromarray((image_patch).astype(np.uint8))
    mask_patch = Image.fromarray((mask_patch * 255).astype(np.uint8))
    image_patch.save(f"{name}/Images/{i}_{j}_0.png")
    mask_patch.save(f"{name}/Masks/{i}_{j}_0.png")

def create_data(bbox, kml, name, size=256, processes=4):
    """
    Create image and mask data from a bounding box and KML file.
    
    Args:
        bbox: Bounding box coordinates [min_lon, min_lat, max_lon, max_lat]
        kml: Path to KML file
        name: Base name for output files
        size: Size of image patches
        processes: Number of processes for parallel processing
    """
    def create_folders():
        os.makedirs(name, exist_ok=True)
        os.makedirs(f"{name}/Images", exist_ok=True)
        os.makedirs(f"{name}/Masks", exist_ok=True)
        os.makedirs("/netscratch/mukhtar/tmp", exist_ok=True)
    
    def generate_and_process_data():
        start_time = time.time()
        print(f"[+] Starting process for {bbox}")
        print("[+] Fetching satellite imagery...")
        tmp_tiff = f"/netscratch/mukhtar/tmp/{name}.tif"
        try:
            leafmap.map_tiles_to_geotiff(
                output=tmp_tiff, 
                bbox=bbox, 
                zoom=19, 
                source="SATELLITE", 
                overwrite=True,
                options={"BIGTIFF": "YES"}
            )
            print(f"[+] Satellite imagery fetched in {time.time() - start_time:.2f} seconds")
            
            with rasterio.open(tmp_tiff) as src:
                image_data = src.read()
                transform = src.transform
                crs = src.crs
                shape = (src.height, src.width)
            
            mask_start = time.time()
            print(f"[+] Generating mask from KML ({shape[0]}x{shape[1]} pixels)...")
            
            gdf = gpd.read_file(kml, driver="KML").to_crs(crs)
            slum_polygons = [(geom, 1) for geom in gdf.geometry if geom is not None]
            
            mask_data = rasterize(
                slum_polygons,
                out_shape=shape,
                transform=transform,
                fill=0,
                dtype=rasterio.uint8,
                all_touched=False
            )
            
            mask_tensor = torch.tensor(mask_data).unsqueeze(0)
            image_tensor = torch.tensor(image_data)
            
            print(f"[+] Mask generated in {time.time() - mask_start:.2f} seconds")
            
            indexes = []
            for i in range(0, shape[0], size):
                for j in range(0, shape[1], size):
                    indexes.append((i, j))
            
            process_fn = partial(process_patch,
                                 image_tensor=image_tensor,
                                 mask_tensor=mask_tensor,
                                 name=name,
                                 size=size)
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=processes) as executor:
                list(tqdm(executor.map(process_fn, indexes), total=len(indexes)))
            
            total_time = time.time() - start_time
            print(f"[+] Processing completed in {total_time:.2f} seconds")
            
        except Exception as e:
            print(f"[-] Error: {str(e)}")
    
    create_folders()
    generate_and_process_data()

if __name__ == "__main__":
    create_data([66.8912802, 24.7816344, 67.2775371, 24.9432908],
                "./Lahore.kml",
                "Lahore",
                size=512,
                processes=32)

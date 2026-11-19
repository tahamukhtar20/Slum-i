"""Cut a 64x64 Sentinel-2 window (640 m) around every Esri tile, so the S2 dataset has the same tiles,
splits and seeds as the Esri one; only the sensor differs.

  imagery : the merged 10-band GeoTIFF (data/mapcv/<city>/s2/<city>_s2.tif)
  labels  : mapcv's S2 masks (same labels config as Esri), stitched back to the full raster
  window  : centred on the Esri tile's centroid; pixels outside the tile footprint are 255 (only the
            centre ~26x26 px is labelled), and imagery that falls in a tile of a *different* split is
            zeroed so no train window sees val/test ground (and vice versa).
Output data/tiles/<city>_s2/{Images/<stem>.tif (10x64x64 uint16), Masks/<stem>.png, splits/ (copied),
s2_meta.json (band names, acquisition, per-band train mean/std)}.
"""
import json, os, shutil, sys

import numpy as np
import pyproj
import rasterio
import rasterio.features
from PIL import Image
from rasterio.windows import Window
from shapely.geometry import shape
from shapely.ops import transform

city = sys.argv[1]
from pathlib import Path
M = str(Path(__file__).resolve().parents[2])  # repo root
ESRI = {"lahore": "lahore_esri_z18", "karachi": "karachi_esri_z18_f", "mumbai": "mumbai_esri_z18"}[city]
esri, raw = f"{M}/data/tiles/{ESRI}", f"{M}/data/tiles/{city}_s2_raw"
out = f"{M}/data/tiles/{city}_s2"
WIN, HALF = 64, 32
CODE = {"train": 1, "val": 2, "test": 3}

src = rasterio.open(f"{M}/data/mapcv/{city}/s2/{city}_s2.tif")
data, T, H, W = src.read(), src.transform, src.height, src.width
fwd = pyproj.Transformer.from_crs(4326, src.crs, always_xy=True).transform

# mapcv's S2 masks stitched back onto the full raster (255 where no patch was written)
full = np.full((H, W), 255, np.uint8)
for f in json.load(open(f"{raw}/patches.geojson"))["features"]:
    p = f["properties"]
    with rasterio.open(f"{raw}/Masks/{p['filename']}") as m:
        a = m.read(1)
    r, c = p["row"], p["col"]
    h, w = min(a.shape[0], H - r), min(a.shape[1], W - c)
    full[r:r + h, c:c + w] = a[:h, :w]

# which split each S2 pixel's ground belongs to (0 = no Esri tile)
split_of = {}
for s in ("train", "val", "test"):
    for n in open(f"{esri}/splits/{s}.txt").read().split():
        split_of[n] = s
feats = json.load(open(f"{esri}/patches.geojson"))["features"]
geoms = {f["properties"]["filename"]: transform(fwd, shape(f["geometry"])) for f in feats}
code = rasterio.features.rasterize([(g, CODE[split_of[n]]) for n, g in geoms.items() if n in split_of],
                                   out_shape=(H, W), transform=T, fill=0, dtype="uint8")

if os.path.exists(out):
    shutil.rmtree(out)
os.makedirs(f"{out}/Images"); os.makedirs(f"{out}/Masks")
shutil.copytree(f"{esri}/splits", f"{out}/splits")
inv = ~T
n_blank, n_out, cen = [], 0, []
sums, sqs, cnt = np.zeros(data.shape[0]), np.zeros(data.shape[0]), 0
for n, g in geoms.items():
    if n not in split_of:
        continue
    col, row = inv * (g.centroid.x, g.centroid.y)
    r0, c0 = int(round(row)) - HALF, int(round(col)) - HALF
    img = np.zeros((data.shape[0], WIN, WIN), np.uint16)
    msk = np.full((WIN, WIN), 255, np.uint8)
    cod = np.zeros((WIN, WIN), np.uint8)
    rs, re, cs, ce = max(r0, 0), min(r0 + WIN, H), max(c0, 0), min(c0 + WIN, W)
    if rs >= re or cs >= ce:
        n_out += 1
        continue
    img[:, rs - r0:re - r0, cs - c0:ce - c0] = data[:, rs:re, cs:ce]
    msk[rs - r0:re - r0, cs - c0:ce - c0] = full[rs:re, cs:ce]
    cod[rs - r0:re - r0, cs - c0:ce - c0] = code[rs:re, cs:ce]
    wT = T * rasterio.Affine.translation(c0, r0)
    inside = rasterio.features.geometry_mask([g], (WIN, WIN), wT, invert=True)
    msk[~inside] = 255                                   # label only the Esri tile's footprint
    other = (cod != 0) & (cod != CODE[split_of[n]])
    img[:, other] = 0                                    # no imagery from another split's ground
    n_blank.append(other.mean()); cen.append(inside.sum())
    stem = os.path.splitext(n)[0]
    with rasterio.open(f"{out}/Images/{stem}.tif", "w", driver="GTiff", dtype="uint16", count=img.shape[0],
                       width=WIN, height=WIN, crs=src.crs, transform=wT, nodata=0, compress="deflate") as d:
        d.write(img)
    np.save(f"{out}/Images/{stem}.npy", img)
    Image.fromarray(msk).save(f"{out}/Masks/{stem}.png")
    if split_of[n] == "train":
        v = img.reshape(img.shape[0], -1)[:, (img > 0).all(0).ravel()].astype(np.float64)
        sums += v.sum(1); sqs += (v ** 2).sum(1); cnt += v.shape[1]

mean = sums / max(cnt, 1); std = np.sqrt(np.maximum(sqs / max(cnt, 1) - mean ** 2, 1e-6))
meta = {"bands": [src.descriptions[i] or f"b{i+1}" for i in range(src.count)], "tags": src.tags(),
        "window": WIN, "mean": mean.tolist(), "std": std.tolist(), "windows": len(cen), "outside_raster": n_out,
        "centre_px_mean": float(np.mean(cen)), "blanked_frac_mean": float(np.mean(n_blank)),
        "blanked_frac_max": float(np.max(n_blank))}
json.dump(meta, open(f"{out}/s2_meta.json", "w"), indent=1)
lab = np.concatenate([np.array(Image.open(f"{out}/Masks/{os.path.splitext(n)[0]}.png")).ravel() for n in list(geoms)[:400] if n in split_of])
print(json.dumps({k: v for k, v in meta.items() if k not in ("mean", "std")}),
      "slum share of labelled centre px (first 400): %.3f" % ((lab == 1).sum() / max((lab != 255).sum(), 1)), flush=True)

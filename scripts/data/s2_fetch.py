"""Fetch one cloud-free Sentinel-2 L2A acquisition over a city region from Earth Search and write it as a
single 10-band GeoTIFF on the 10 m UTM grid (uint16, 0 = no data). A city may straddle two MGRS granules;
granules of the same acquisition (same date, same UTM zone) are merged. 20 m bands and SCL are repeated
2x (nearest), as mapcv's stac_cog source does; SCL classes 3, 8, 9, 10 (shadow, cloud, cirrus) become 0.
Usage: python s2_fetch.py <city> <datetime range> <out.tif>
"""
import json, math, sys
from collections import defaultdict

import numpy as np
import pyproj
import rasterio
import rasterio.features
from rasterio.windows import from_bounds
from pystac_client import Client
from shapely.geometry import shape, box
from shapely.ops import unary_union, transform

city, dt, out = sys.argv[1], sys.argv[2], sys.argv[3]
from pathlib import Path
M = str(Path(__file__).resolve().parents[2])  # repo root
BANDS = ["blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "swir16", "swir22"]
region = unary_union([shape(f["geometry"]) for f in json.load(open(f"{M}/data/mapcv/{city}/region_s2.geojson"))["features"]])

items = list(Client.open("https://earth-search.aws.element84.com/v1").search(
    collections=["sentinel-2-l2a"], intersects=region.__geo_interface__, datetime=dt,
    query={"eo:cloud_cover": {"lt": 20}}).items())
groups = defaultdict(list)
for it in items:
    groups[(it.datetime.date(), it.properties.get("proj:epsg") or it.properties.get("proj:code"))].append(it)

def covers(group):
    return unary_union([shape(i.geometry) for i in group]).buffer(1e-6).contains(region)

cands = [(max(i.properties["eo:cloud_cover"] for i in g), k, g) for k, g in groups.items() if covers(g)]
if not cands:
    sys.exit(f"no acquisition covers the region ({len(items)} items, {len(groups)} acquisitions)")
cloud, (date, epsg), group = sorted(cands, key=lambda c: (c[0], c[1][0]))[0]
epsg = int(str(epsg).split(":")[-1])
print("chosen", date, "EPSG", epsg, "max cloud %.2f%%" % cloud, [i.id for i in group], flush=True)

# region bounds in UTM, snapped outward to the 20 m grid so both resolutions share pixel corners
fwd = pyproj.Transformer.from_crs(4326, epsg, always_xy=True).transform
x0, y0, x1, y1 = transform(fwd, region).bounds
x0, y0 = math.floor(x0 / 20) * 20, math.floor(y0 / 20) * 20
x1, y1 = math.ceil(x1 / 20) * 20, math.ceil(y1 / 20) * 20
W, H = int((x1 - x0) / 10), int((y1 - y0) / 10)
stack = np.zeros((len(BANDS), H, W), np.uint16)
filled = np.zeros((H, W), bool)

def read10(href):
    """Read the region from one asset onto the 10 m grid (20 m pixels repeated 2x)."""
    with rasterio.open(href) as src:
        res = src.res[0]
        win = from_bounds(x0, y0, x1, y1, src.transform).round_offsets().round_lengths()
        a = src.read(1, window=win, boundless=True, fill_value=0)
    f = int(round(res / 10))
    if f > 1:
        a = a.repeat(f, 0).repeat(f, 1)
    return a[:H, :W]

for it in sorted(group, key=lambda i: i.properties["eo:cloud_cover"]):
    scl = read10(it.assets["scl"].href)
    bands = np.stack([read10(it.assets[b].href) for b in BANDS])
    valid = (bands > 0).all(0) & ~np.isin(scl, [3, 8, 9, 10]) & ~filled
    stack[:, valid] = bands[:, valid]
    filled |= valid
    print(it.id, "filled %.4f" % filled.mean(), flush=True)

prof = {"driver": "GTiff", "dtype": "uint16", "count": len(BANDS), "width": W, "height": H, "crs": f"EPSG:{epsg}",
        "transform": rasterio.transform.from_origin(x0, y1, 10, 10), "nodata": 0, "compress": "deflate",
        "tiled": True, "blockxsize": 512, "blockysize": 512, "predictor": 2}
with rasterio.open(out, "w", **prof) as dst:
    dst.write(stack)
    for i, b in enumerate(BANDS, 1):
        dst.set_band_description(i, b)
    dst.update_tags(acquisition=str(date), items=",".join(i.id for i in group), max_cloud=str(cloud))
inside = rasterio.features.geometry_mask([transform(fwd, region)], (H, W), prof["transform"], invert=True)
print("written", out, (H, W), "valid inside region %.4f" % filled[inside].mean(), flush=True)

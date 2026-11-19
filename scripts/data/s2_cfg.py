"""Write mapcv Sentinel-2 configs: same labels and region as the Esri datasets, region widened by 400 m
so 64-px (640 m) windows centred on edge tiles still have imagery around them."""
import json, sys
import pyproj
from shapely.geometry import shape, mapping
from shapely.ops import transform, unary_union

from pathlib import Path
M = str(Path(__file__).resolve().parents[2])  # repo root
DATES = {"lahore": "2024-03-01/2024-05-15", "karachi": "2024-01-01/2024-04-30", "mumbai": "2024-01-15/2024-04-30"}
BANDS = ["blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "swir16", "swir22"]
for city in sys.argv[1:]:
    d = f"{M}/data/mapcv/{city}"
    g = json.load(open(f"{d}/region_buf250.geojson"))
    geom = unary_union([shape(f["geometry"]) for f in g["features"]])
    lon, lat = geom.centroid.x, geom.centroid.y
    utm = pyproj.CRS.from_dict({"proj": "utm", "zone": int((lon + 180) // 6) + 1, "south": lat < 0})
    fwd = pyproj.Transformer.from_crs(4326, utm, always_xy=True).transform
    inv = pyproj.Transformer.from_crs(utm, 4326, always_xy=True).transform
    big = transform(inv, transform(fwd, geom).buffer(400))
    json.dump({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": mapping(big)}]},
              open(f"{d}/region_s2.geojson", "w"))
    esri = open(f"{d}/esri_z18.yaml").read()
    labels = esri[esri.index("labels:"):esri.index("sampler:")]
    split = esri[esri.index("split:"):]
    cfg = f'''region:
  path: "region_s2.geojson"
imagery:
  type: "geotiff"
  path: "s2/{city}_s2.tif"
  nodata: 0
{labels}sampler:
  patch_size: 256
  edge_strategy: "pad"
  pad_mode: "zero"
  max_empty_ratio: 1.0
writer:
  staging_dir: "{M}/data/tiles/{city}_s2_raw"
  image_format: "tif"
  mask_format: "tif"
{split}'''
    open(f"{d}/s2.yaml", "w").write(cfg)
    print(city, "bbox", [round(v, 4) for v in big.bounds], "utm", utm.to_epsg() or utm.name)

"""Turn a city's settlement polygons (KML) into the mapcv inputs in data/mapcv/<city>/.

Writes labels.geojson (one feature per polygon, class "slum"), region_buf250.geojson (the polygons
buffered by 250 m and dissolved) and, when a polygon is larger than 500 ha, annotated_area.geojson
(the region without those polygons, so their pixels get the ignore value). Only Karachi has such
polygons (four).

Usage: python scripts/data/make_labels.py <city> <polygons.kml>
"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from pyproj import Transformer
from shapely.geometry import Polygon, mapping
from shapely.ops import transform, unary_union

ROOT = Path(__file__).resolve().parents[2]
BUFFER_M = 250.0
GIANT_HA = 500.0


def parse_kml(path):
    """Shapely polygons (WGS84) in document order, one per <Polygon>."""
    def ring(el):
        txt = next((c.text for c in el.iter() if c.tag.endswith("coordinates")), None)
        if not txt:
            return None
        pts = [(float(t.split(",")[0]), float(t.split(",")[1])) for t in txt.split() if len(t.split(",")) >= 2]
        return pts if len(pts) >= 3 else None

    polys = []
    for el in ET.parse(path).getroot().iter():
        if not el.tag.endswith("Polygon"):
            continue
        outer, inners = None, []
        for c in el:
            if c.tag.endswith("outerBoundaryIs"):
                outer = ring(c)
            elif c.tag.endswith("innerBoundaryIs"):
                r = ring(c)
                if r:
                    inners.append(r)
        if outer:
            p = Polygon(outer, inners)
            if not p.is_valid:
                p = p.buffer(0)
            if not p.is_empty:
                polys.append(p)
    return polys


def write_geojson(path, geoms_props):
    fc = {"type": "FeatureCollection",
          "features": [{"type": "Feature", "properties": p, "geometry": mapping(g)} for g, p in geoms_props]}
    path.write_text(json.dumps(fc))


def main(city, kml):
    polys = parse_kml(Path(kml))
    c = unary_union(polys).centroid
    epsg = (32600 if c.y >= 0 else 32700) + int((c.x + 180) // 6) + 1
    fwd = Transformer.from_crs(4326, epsg, always_xy=True).transform
    inv = Transformer.from_crs(epsg, 4326, always_xy=True).transform
    pm = [transform(fwd, p) for p in polys]
    areas = np.array([p.area for p in pm]) / 1e4
    giant = [i for i, a in enumerate(areas) if a > GIANT_HA]
    keep = [i for i in range(len(pm)) if i not in giant]
    region_m = unary_union([pm[i].buffer(BUFFER_M) for i in keep])
    parts = list(region_m.geoms) if hasattr(region_m, "geoms") else [region_m]
    d = ROOT / "data" / "mapcv" / city
    d.mkdir(parents=True, exist_ok=True)
    write_geojson(d / "labels.geojson", [(polys[i], {"polygon_id": i, "class": "slum"}) for i in keep])
    write_geojson(d / f"region_buf{int(BUFFER_M)}.geojson",
                  [(transform(inv, g), {"region": i + 1}) for i, g in enumerate(parts)])
    if giant:
        ann = region_m.difference(unary_union([pm[i] for i in giant]))
        ann_parts = list(ann.geoms) if hasattr(ann, "geoms") else [ann]
        write_geojson(d / "annotated_area.geojson",
                      [(transform(inv, g), {"part": i + 1}) for i, g in enumerate(ann_parts)])
    print(f"{city}: {len(polys)} polygons, {len(giant)} over {GIANT_HA:.0f} ha ignored, "
          f"region {region_m.area / 1e6:.1f} km2 in {len(parts)} parts, UTM EPSG {epsg}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

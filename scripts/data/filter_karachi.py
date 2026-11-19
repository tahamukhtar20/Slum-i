"""Karachi: drop the tiles that are more than half unlabelled (inside the four polygons over 500 ha).

Reads data/tiles/karachi_esri_z18 (written by mapcv) and writes data/tiles/karachi_esri_z18_f, whose
Images/ and Masks/ link to the unfiltered folders and whose split lists keep only the remaining tiles.
The shipped data/tiles/karachi_esri_z18_f/splits are the result (792 of 1,850 tiles).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
src = ROOT / "data" / "tiles" / "karachi_esri_z18"
dst = ROOT / "data" / "tiles" / "karachi_esri_z18_f"
dst.mkdir(parents=True, exist_ok=True)
for d in ("Images", "Masks"):
    if not (dst / d).exists():
        (dst / d).symlink_to(src / d)
for f in ("patches.geojson", "manifest.json"):
    if not (dst / f).exists():
        (dst / f).symlink_to(src / f)

ignored = {}
for f in json.loads((src / "patches.geojson").read_text())["features"]:
    cp = f["properties"].get("class_pixels") or {}
    ignored[f["properties"]["filename"]] = cp.get("255", 0) / (sum(cp.values()) or 1)
keep = {k for k, v in ignored.items() if k and v <= 0.5}
for p in sorted((src / "splits").rglob("*.txt")):
    rel = p.relative_to(src / "splits")
    names = [l for l in p.read_text().splitlines() if l]
    out = dst / "splits" / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(n for n in names if n in keep) + "\n")
    print(f"{rel}: kept {sum(n in keep for n in names)} of {len(names)}")

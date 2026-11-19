#!/bin/bash
# Rebuild the Esri and Sentinel-2 datasets of one city under data/tiles/.
# Usage: bash scripts/data/build_city.sh <lahore|karachi|mumbai> [polygons.kml]
# Lahore ships its polygons (data/lahore.kml, already converted in data/mapcv/lahore/).
# For Karachi and Mumbai pass the KML of the source map (see README, Data).
# Needs the packages in requirements-data.txt (mapcv 0.3.0, rasterio, pystac-client, pyproj, shapely).
set -eo pipefail
CITY="$1"; KML="$2"
cd "$(dirname "$0")/../.."
declare -A DATES=([lahore]="2024-03-01/2024-05-15" [karachi]="2024-01-01/2024-04-30" [mumbai]="2024-01-15/2024-04-30")
ESRI=${CITY}_esri_z18; [ "$CITY" = karachi ] && ESRI=karachi_esri_z18_f

# Move the shipped split lists aside (mapcv writes into empty folders) to check the rebuild against them.
mkdir -p data/tiles/.ref
for d in "$ESRI" "${CITY}_s2" ${CITY}_esri_z18; do
  [ -d "data/tiles/$d" ] && [ ! -e "data/tiles/.ref/$d" ] && mv "data/tiles/$d" "data/tiles/.ref/$d"
done

# 1. Labels and region (KML -> GeoJSON).
if [ -n "$KML" ]; then python scripts/data/make_labels.py "$CITY" "$KML"; fi

# 2. Esri World Imagery tiles, masks and spatial splits.
mapcv generate "data/mapcv/$CITY/esri_z18.yaml" --yes
[ "$CITY" = karachi ] && python scripts/data/filter_karachi.py

# 3. Sentinel-2: config, one L2A acquisition, mapcv masks, then 64x64 windows around the Esri tiles.
python scripts/data/s2_cfg.py "$CITY"
mkdir -p "data/mapcv/$CITY/s2"
python scripts/data/s2_fetch.py "$CITY" "${DATES[$CITY]}" "data/mapcv/$CITY/s2/${CITY}_s2.tif"
mapcv generate "data/mapcv/$CITY/s2.yaml" --yes
python scripts/data/s2_windows.py "$CITY"

# 4. The rebuilt splits must equal the shipped ones.
for d in "$ESRI" "${CITY}_s2"; do
  diff -r -x split.json "data/tiles/.ref/$d/splits" "data/tiles/$d/splits" && echo "splits match: $d"
done

#!/usr/bin/env python3
"""Clustered tiling pipeline for KML cities.

Workflow:
    1. Parse KML annotations into WGS-84 polygons
    2. Quadtree decompose the full bbox in WGS-84 space (no UTM needed)
    3. Buffer each leaf cell, union all, grid-decompose, beam-search merge
    4. Drop rectangles that contain no annotation polygon
    5. For each final rectangle: fetch tiles, rasterize its polygons, sample + write patches
    6. Save manifest.json (mapcv format) + clusters.json sidecar
    7. Run split_clustered.py separately to produce split files

Output layout is identical to mapcv's standard staging dir.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from shapely.geometry import Point, box
from shapely.ops import unary_union
from shapely.strtree import STRtree

from mapcv._mapcv_rs import snap_bbox, stitch_tiles as stitch_tiles_rs
from mapcv._mapcv_rs import tile_transform as tile_transform_rs
from mapcv.downloader import download_region_strips, iter_tile_strips, resolve_url_template
from mapcv.labels import parse_kml, transform_to_mercator
from mapcv.rasterizer import rasterize
from mapcv.sampler import SamplerConfig, sample_patches
from mapcv.writer import Manifest, WriterConfig, load_or_create_manifest, write_patches


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class QuadtreeConfig:
    min_size_m: float = 500.0


@dataclass
class TilesCfg:
    source: str = "esri_satellite"
    url_template: Optional[str] = None
    policy: str = "strict"
    max_connections: int = 16
    max_failed_ratio: float = 0.05
    strip_rows: int = 4


@dataclass
class SamplerCfg:
    max_empty_ratio: float = 0.95
    min_label_ratio: float = 0.0
    edge_strategy: str = "drop"
    pad_mode: str = "zero"
    mode: str = "grid"


@dataclass
class GenerateConfig:
    city: str
    kml_path: Path
    output_dir: Path
    zoom: int = 19
    patch_size: int = 512
    buffer_m: float = 200.0
    seed: int = 42
    label_field: Optional[str] = None
    all_touched: bool = False
    beam_width: int = 20
    quadtree: QuadtreeConfig = field(default_factory=QuadtreeConfig)
    tiles: TilesCfg = field(default_factory=TilesCfg)
    sampler: SamplerCfg = field(default_factory=SamplerCfg)

    @classmethod
    def from_yaml(cls, path: Path) -> "GenerateConfig":
        data = yaml.safe_load(Path(path).read_text())
        quadtree = QuadtreeConfig(**data.pop("quadtree", {}))
        tiles = TilesCfg(**data.pop("tiles", {}))
        sampler = SamplerCfg(**data.pop("sampler", {}))
        data.pop("split", None)
        data.pop("dbscan", None)
        data["kml_path"] = Path(data["kml_path"])
        data["output_dir"] = Path(data["output_dir"])
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(quadtree=quadtree, tiles=tiles, sampler=sampler,
                   **{k: v for k, v in data.items() if k in valid})


# ---------------------------------------------------------------------------
# Metres ↔ degrees (approximate, at a given latitude)
# ---------------------------------------------------------------------------

def _metres_to_deg(metres: float, lat: float) -> Tuple[float, float]:
    lat_deg = metres / 111_320.0
    lon_deg = metres / (111_320.0 * math.cos(math.radians(lat)))
    return lon_deg, lat_deg


# ---------------------------------------------------------------------------
# WGS-84 quadtree
# ---------------------------------------------------------------------------

def _quadtree_leaves(
    w: float, s: float, e: float, n: float,
    min_lon: float, min_lat: float,
    tree: STRtree, polys: list,
) -> List[Tuple[float, float, float, float]]:
    cell = box(w, s, e, n)
    hits = tree.query(cell)
    if not any(polys[i].intersects(cell) for i in hits):
        return []
    if (e - w) <= min_lon or (n - s) <= min_lat:
        return [(w, s, e, n)]
    cx, cy = (w + e) / 2, (s + n) / 2
    out: List[Tuple[float, float, float, float]] = []
    for quad in [(w, cy, cx, n), (cx, cy, e, n), (w, s, cx, cy), (cx, s, e, cy)]:
        out.extend(_quadtree_leaves(*quad, min_lon, min_lat, tree, polys))
    return out


# ---------------------------------------------------------------------------
# Grid decomposition + merging
# ---------------------------------------------------------------------------

def _grid_from_union(buffered_boxes, union):
    xs = sorted(set(c for g in buffered_boxes for c in [g.bounds[0], g.bounds[2]]))
    ys = sorted(set(c for g in buffered_boxes for c in [g.bounds[1], g.bounds[3]]))
    grid: List[Tuple[float, float, float, float]] = []
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            cx, cy = (xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2
            if union.contains(Point(cx, cy)):
                grid.append((xs[i], ys[j], xs[i + 1], ys[j + 1]))
    return grid


def _deterministic_merge(
    grid: List[Tuple[float, float, float, float]], h_first: bool
) -> List[Tuple[float, float, float, float]]:
    TOL = 1e-12
    if h_first:
        by_y: Dict = defaultdict(list)
        for (w, s, e, n) in grid:
            by_y[(s, n)].append((w, e))
        strips = []
        for (s, n), xe in by_y.items():
            xe.sort(); mw, me = xe[0]
            for (w, e) in xe[1:]:
                if abs(w - me) < TOL: me = e
                else: strips.append((mw, s, me, n)); mw, me = w, e
            strips.append((mw, s, me, n))
        by_x: Dict = defaultdict(list)
        for (w, s, e, n) in strips: by_x[(w, e)].append((s, n))
        result = []
        for (w, e), sn in by_x.items():
            sn.sort(); ms, mn = sn[0]
            for (s, n) in sn[1:]:
                if abs(s - mn) < TOL: mn = n
                else: result.append((w, ms, e, mn)); ms, mn = s, n
            result.append((w, ms, e, mn))
        return result
    else:
        by_x: Dict = defaultdict(list)
        for (w, s, e, n) in grid: by_x[(w, e)].append((s, n))
        strips = []
        for (w, e), sn in by_x.items():
            sn.sort(); ms, mn = sn[0]
            for (s, n) in sn[1:]:
                if abs(s - mn) < TOL: mn = n
                else: strips.append((w, ms, e, mn)); ms, mn = s, n
            strips.append((w, ms, e, mn))
        by_y: Dict = defaultdict(list)
        for (w, s, e, n) in strips: by_y[(s, n)].append((w, e))
        result = []
        for (s, n), xe in by_y.items():
            xe.sort(); mw, me = xe[0]
            for (w, e) in xe[1:]:
                if abs(w - me) < TOL: me = e
                else: result.append((mw, s, me, n)); mw, me = w, e
            result.append((mw, s, me, n))
        return result


def _merge_candidates(state: tuple, tol: float = 1e-12):
    rects = state
    candidates = []
    for i in range(len(rects)):
        wi, si, ei, ni = rects[i]
        for j in range(i + 1, len(rects)):
            wj, sj, ej, nj = rects[j]
            if abs(si - sj) < tol and abs(ni - nj) < tol:
                if abs(ei - wj) < tol: candidates.append((i, j, (wi, si, ej, ni)))
                elif abs(ej - wi) < tol: candidates.append((i, j, (wj, si, ei, ni)))
            elif abs(wi - wj) < tol and abs(ei - ej) < tol:
                if abs(ni - sj) < tol: candidates.append((i, j, (wi, si, ei, nj)))
                elif abs(nj - si) < tol: candidates.append((i, j, (wi, sj, ei, ni)))
    return candidates


def _beam_search_merge(
    initial: List[Tuple[float, float, float, float]], beam_width: int = 20
) -> List[Tuple[float, float, float, float]]:
    def canonical(rects): return tuple(sorted(rects))
    best = canonical(initial)
    beam = {best}
    improved = True
    while improved:
        improved = False
        next_beam: Dict[tuple, int] = {}
        for state in beam:
            rlist = list(state)
            for (i, j, merged) in _merge_candidates(state):
                new_state = canonical(
                    [r for k, r in enumerate(rlist) if k not in (i, j)] + [merged]
                )
                score = len(new_state)
                if new_state not in next_beam or next_beam[new_state] > score:
                    next_beam[new_state] = score
            if state not in next_beam:
                next_beam[state] = len(state)
        sorted_states = sorted(next_beam, key=lambda s: next_beam[s])
        beam = set(sorted_states[:beam_width])
        if len(sorted_states[0]) < len(best):
            best = sorted_states[0]
            improved = True
    return list(best)


def _decompose_and_merge(
    leaves: List[Tuple[float, float, float, float]],
    buf_lon: float, buf_lat: float,
    beam_width: int = 20,
) -> Tuple[List[Tuple[float, float, float, float]], object]:
    buffered_boxes = [box(w - buf_lon, s - buf_lat, e + buf_lon, n + buf_lat)
                      for (w, s, e, n) in leaves]
    union = unary_union(buffered_boxes)
    grid = _grid_from_union(buffered_boxes, union)
    h = _deterministic_merge(grid, h_first=True)
    v = _deterministic_merge(grid, h_first=False)
    seed = h if len(h) <= len(v) else v
    print(f"[generate_clustered]   H-first {len(h)} | V-first {len(v)} → seed {len(seed)}")
    result = _beam_search_merge(seed, beam_width=beam_width)
    return result, union


# ---------------------------------------------------------------------------
# Per-cluster tile fetch + rasterize + sample + write
# ---------------------------------------------------------------------------

def _patches_touching_missing_tiles(
    meta: list,
    missing_xy: set,
    min_x: int,
    min_y: int,
    patch_size: int,
    tile_px: int = 256,
) -> List[str]:
    """Return indices into meta for patches that overlap at least one missing tile region."""
    affected = []
    for idx, m in enumerate(meta):
        pr, pc = m["row"], m["col"]  # patch top-left in strip pixel coords
        for (tx, ty) in missing_xy:
            tile_col_start = (tx - min_x) * tile_px
            tile_row_start = (ty - min_y) * tile_px
            tile_col_end = tile_col_start + tile_px
            tile_row_end = tile_row_start + tile_px
            # Check overlap between patch rect and tile rect
            if (pc < tile_col_end and pc + patch_size > tile_col_start and
                    pr < tile_row_end and pr + patch_size > tile_row_start):
                affected.append(idx)
                break
    return affected


def _process_cluster(
    cluster_id: int,
    west: float, south: float, east: float, north: float,
    poly_indices: List[int],
    polygons_mercator: list,
    classes: List[int],
    cfg: GenerateConfig,
    manifest: Manifest,
    writer_cfg: WriterConfig,
    sampler_cfg: SamplerConfig,
    url_template: str,
    cluster_assignments: Dict[str, int],
    cluster_bboxes: Dict[int, dict],
    failed_tile_patches: set,  # mutated in-place: filenames of patches with missing tiles
) -> int:
    cluster_bboxes[cluster_id] = {
        "west": west, "south": south, "east": east, "north": north,
        "polygons": list(poly_indices),
    }
    cluster_geoms = [(polygons_mercator[i], classes[i]) for i in poly_indices]

    # Get the snapped bbox (same as what mapcv uses internally)
    snapped = snap_bbox(west, south, east, north, cfg.zoom)
    sw, ss, se, sn = snapped.west, snapped.south, snapped.east, snapped.north

    # Expected tiles per strip (before fetching)
    expected_strips = iter_tile_strips(sw, ss, se, sn, cfg.zoom, cfg.tiles.strip_rows)
    expected_by_strip: List[set] = [
        {(t.x, t.y) for t in strip} for strip in expected_strips
    ]

    all_strips = download_region_strips(
        west, south, east, north,
        cfg.zoom, cfg.tiles.strip_rows,
        url_template=url_template,
        max_connections=cfg.tiles.max_connections,
        policy=cfg.tiles.policy,
        max_failed_ratio=cfg.tiles.max_failed_ratio,
    )

    written = 0
    base_strip = cluster_id * 10_000
    for si, strip_data in enumerate(all_strips):
        if not strip_data:
            continue
        image_arr, min_x, min_y = stitch_tiles_rs(strip_data)
        transform = tile_transform_rs(min_x, min_y, cfg.zoom)
        h, w = image_arr.shape[:2]
        if cluster_geoms:
            mask_arr = rasterize(cluster_geoms, (h, w), transform, cfg.all_touched)
        else:
            mask_arr = np.zeros((h, w), dtype="uint8")

        imgs, msks, meta = sample_patches(image_arr, mask_arr, sampler_cfg)
        before = len(manifest.patches)
        write_patches(imgs, msks, meta, writer_cfg, manifest, strip_index=base_strip + si)
        new_entries = manifest.patches[before:]

        for entry in new_entries:
            cluster_assignments[entry["filename"]] = cluster_id

        # Track patches overlapping any failed tile in this strip
        if si < len(expected_by_strip):
            fetched_xy = {(t.x, t.y) for t, _ in strip_data}
            missing_xy = expected_by_strip[si] - fetched_xy
            if missing_xy:
                affected_idxs = _patches_touching_missing_tiles(
                    meta, missing_xy, min_x, min_y, cfg.patch_size
                )
                for idx in affected_idxs:
                    failed_tile_patches.add(new_entries[idx]["filename"])
                if affected_idxs:
                    print(
                        f"[generate_clustered]   ⚠ {len(missing_xy)} missing tile(s) in strip {si}, "
                        f"{len(affected_idxs)} patch(es) flagged"
                    )

        written += len(new_entries)
    return written


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(cfg: GenerateConfig) -> None:
    print(f"[generate_clustered] city={cfg.city}  zoom={cfg.zoom}  buffer={cfg.buffer_m}m")
    print(f"[generate_clustered] quadtree min_size={cfg.quadtree.min_size_m}m  beam_width={cfg.beam_width}")
    print(f"[generate_clustered] tiles source={cfg.tiles.source}  policy={cfg.tiles.policy}")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[generate_clustered] parsing {cfg.kml_path}")
    kml_bytes = Path(cfg.kml_path).read_bytes()
    raw, class_map = parse_kml(kml_bytes, cfg.label_field)
    if not raw:
        print("[generate_clustered] no polygons; aborting")
        return
    polys_wgs = [g for g, _ in raw]
    classes = [c for _, c in raw]
    polys_mercator = [transform_to_mercator(g) for g in polys_wgs]
    print(f"[generate_clustered] {len(polys_wgs)} polygons, {len(class_map) or 1} class(es)")

    lat_mean = sum(float(g.centroid.y) for g in polys_wgs) / len(polys_wgs)
    min_lon, min_lat_deg = _metres_to_deg(cfg.quadtree.min_size_m, lat_mean)
    buf_lon, buf_lat = _metres_to_deg(cfg.buffer_m, lat_mean)

    tree = STRtree(polys_wgs)
    all_w = min(g.bounds[0] for g in polys_wgs)
    all_s = min(g.bounds[1] for g in polys_wgs)
    all_e = max(g.bounds[2] for g in polys_wgs)
    all_n = max(g.bounds[3] for g in polys_wgs)

    print(f"[generate_clustered] running quadtree...")
    leaves = _quadtree_leaves(all_w, all_s, all_e, all_n, min_lon, min_lat_deg, tree, polys_wgs)
    print(f"[generate_clustered] {len(leaves)} leaf cells")

    print(f"[generate_clustered] decomposing and beam-search merging...")
    rects_wgs, _ = _decompose_and_merge(leaves, buf_lon, buf_lat, cfg.beam_width)
    print(f"[generate_clustered] {len(rects_wgs)} rectangles after merge")

    # Multi-pass filter: drop bboxes with < 10m x 10m of polygon intersection area
    poly_tree = STRtree(polys_wgs)
    min_m = 10.0
    lat_deg = min_m / 111_320.0
    lon_deg = min_m / (111_320.0 * math.cos(math.radians(lat_mean)))
    min_inter_area = lat_deg * lon_deg

    def _has_content(r):
        cell = box(*r)
        for i in poly_tree.query(cell):
            try:
                poly = polys_wgs[i] if polys_wgs[i].is_valid else polys_wgs[i].buffer(0)
                inter = poly.intersection(cell)
                if not inter.is_empty and inter.area >= min_inter_area:
                    return True
            except Exception:
                if polys_wgs[i].intersects(cell):
                    return True
        return False

    prev = len(rects_wgs) + 1
    passes = 0
    while len(rects_wgs) < prev:
        prev = len(rects_wgs)
        passes += 1
        rects_wgs = [r for r in rects_wgs if _has_content(r)]
    print(f"[generate_clustered] {len(rects_wgs)} rectangles after {passes}-pass empty filter")

    # Assign each polygon to its nearest rectangle centre
    rect_centres = [((w + e) / 2, (s + n) / 2) for (w, s, e, n) in rects_wgs]
    poly_to_rect: Dict[int, int] = {}
    for pi, poly in enumerate(polys_wgs):
        cx, cy = float(poly.centroid.x), float(poly.centroid.y)
        best = min(
            range(len(rect_centres)),
            key=lambda ri: (cx - rect_centres[ri][0])**2 + (cy - rect_centres[ri][1])**2
        )
        poly_to_rect[pi] = best
    rect_to_polys: Dict[int, List[int]] = {i: [] for i in range(len(rects_wgs))}
    for pi, ri in poly_to_rect.items():
        rect_to_polys[ri].append(pi)

    # Sort largest area first
    rects_sorted = sorted(
        range(len(rects_wgs)),
        key=lambda ri: -(rects_wgs[ri][2] - rects_wgs[ri][0]) * (rects_wgs[ri][3] - rects_wgs[ri][1])
    )

    manifest_path = cfg.output_dir / "manifest.json"
    clusters_path = cfg.output_dir / "clusters.json"
    for p in (manifest_path, clusters_path):
        if p.exists(): p.unlink()

    manifest = load_or_create_manifest(manifest_path, class_map or {})
    writer_cfg = WriterConfig(staging_dir=cfg.output_dir, image_format="png")
    sampler_cfg = SamplerConfig(
        patch_size=cfg.patch_size, stride=0,
        mode=cfg.sampler.mode, edge_strategy=cfg.sampler.edge_strategy,
        pad_mode=cfg.sampler.pad_mode, max_empty_ratio=cfg.sampler.max_empty_ratio,
        min_label_ratio=cfg.sampler.min_label_ratio, random_seed=cfg.seed,
    )
    url_template = resolve_url_template(cfg.tiles.url_template, cfg.tiles.source)

    cluster_assignments: Dict[str, int] = {}
    cluster_bboxes: Dict[int, dict] = {}
    failed_tile_patches: set = set()

    for new_id, orig_ri in enumerate(rects_sorted):
        w, s, e, n = rects_wgs[orig_ri]
        area_km2 = (e - w) * 111.32 * math.cos(math.radians((s + n) / 2)) * (n - s) * 111.32
        poly_idxs = rect_to_polys[orig_ri]
        print(
            f"[generate_clustered] cluster {new_id:>3}: "
            f"{len(poly_idxs):>3} polys, {area_km2:.2f} km²"
        )
        n_patches = _process_cluster(
            new_id, w, s, e, n, poly_idxs,
            polys_mercator, classes, cfg,
            manifest, writer_cfg, sampler_cfg, url_template,
            cluster_assignments, cluster_bboxes,
            failed_tile_patches,
        )
        print(f"[generate_clustered]   wrote {n_patches} patches")

    manifest.save(manifest_path)

    # Save list of patches that overlapped at least one failed tile
    failed_path = cfg.output_dir / "failed_tile_patches.txt"
    if failed_tile_patches:
        failed_path.write_text("\n".join(sorted(failed_tile_patches)))
        print(
            f"[generate_clustered] ⚠ {len(failed_tile_patches)} patches flagged for failed tiles "
            f"→ {failed_path}"
        )
    else:
        print("[generate_clustered] ✓ no failed tile patches detected")

    clusters_path.write_text(json.dumps({
        "assignments": cluster_assignments,
        "bboxes_wgs84": {str(k): v for k, v in cluster_bboxes.items()},
        "config": {
            "zoom": cfg.zoom, "patch_size": cfg.patch_size,
            "buffer_m": cfg.buffer_m,
            "quadtree": {"min_size_m": cfg.quadtree.min_size_m},
            "beam_width": cfg.beam_width,
            "seed": cfg.seed, "tiles_source": cfg.tiles.source,
        },
    }, indent=2))

    print(
        f"[generate_clustered] done. patches={len(manifest.patches)}, "
        f"clusters={len(cluster_bboxes)}, output={cfg.output_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    run(GenerateConfig.from_yaml(args.config))


if __name__ == "__main__":
    main()

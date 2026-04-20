"""
make_release_raster.py — CLI wrapper to generate AvaFrame release input for Val.

Generates:
  release_depth_jan18_<label>.tif   float32 GeoTIFF, slab depth (m)
  release_depth_jan18_<label>.asc   ASCII grid (AvaFrame legacy)
  release_info_jan18.json           summary stats

All geometry and depth logic lives in release_geometry.py.

Usage:
  python make_release_raster.py
  python make_release_raster.py --depth-source uniform --depth-m 0.6
  python make_release_raster.py --depth-source snowpack --snapshot 2026-01-17
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr

PROJECT_DIR = Path("/home/ron/snowpack_model_feeder")
ZARR_PATH   = Path("/home/ron/snowpack/little_prof/output/slope_snowpack.zarr")
OUT_DIR     = PROJECT_DIR / "outputs/scenarios/jan18_release"

sys.path.insert(0, str(PROJECT_DIR / "src/snowpack-model-feeder"))
from config import ProjectConfig
from snowpack_io import load_dataset, kml_to_mask
from release_geometry import depth_from_snowpack, rasterize_release_polygon
from snowpack_analysis import geojson_to_mask


def write_asc(array: np.ndarray, transform, nodata: float, path: Path):
    nrows, ncols = array.shape
    xllcorner    = transform.c
    yllcorner    = transform.f + nrows * transform.e
    cellsize     = transform.a
    with open(str(path), 'w') as f:
        f.write(f"ncols         {ncols}\n")
        f.write(f"nrows         {nrows}\n")
        f.write(f"xllcorner     {xllcorner:.6f}\n")
        f.write(f"yllcorner     {yllcorner:.6f}\n")
        f.write(f"cellsize      {cellsize:.6f}\n")
        f.write(f"NODATA_value  {nodata}\n")
        for row in array:
            f.write(' '.join(
                f"{nodata}" if np.isnan(v) else f"{v:.4f}"
                for v in row) + '\n')
    print(f"  ASC: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth-source', choices=['snowpack', 'uniform'],
                        default='snowpack')
    parser.add_argument('--depth-m',   type=float, default=0.6)
    parser.add_argument('--snapshot',  default='2026-01-17')
    parser.add_argument('--density',   type=float, default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = ProjectConfig(project_dir=PROJECT_DIR)

    # Load DEM
    with rasterio.open(str(cfg.resampled_dir / "dem_1m.tif")) as src:
        dem       = src.read(1).astype(np.float32)
        dem[dem == src.nodata] = np.nan
        transform = src.transform
        crs       = src.crs
        profile   = src.profile.copy()

    # Rasterize observed release area as the release polygon (Jan 18 validation)
    gj_path      = cfg.project_dir / "data/boundaries/avalanche_release_area.geojson"
    release_mask = geojson_to_mask(gj_path, dem.shape, transform)
    print(f"Release area: {release_mask.sum()} pixels")

    # Build depth grid
    if args.depth_source == 'uniform':
        depth_grid = np.full(dem.shape, args.depth_m, dtype=np.float32)
        label      = f"uniform_{args.depth_m:.2f}m"
    else:
        cluster_map = np.load(str(cfg.analysis_dir / "cluster_map.npy"))
        ds          = load_dataset(cfg.pro_dir, zarr_path=ZARR_PATH)
        depth_grid  = depth_from_snowpack(cluster_map, ds, args.snapshot)
        label       = f"snowpack_{args.snapshot}"

    # Apply mask
    release_depth = rasterize_release_polygon(
        None, depth_grid, dem.shape, transform)
    # Use mask directly since we have the observed boundary
    release_depth = np.where(release_mask & ~np.isnan(depth_grid),
                              depth_grid, np.nan).astype(np.float32)

    valid       = release_depth[~np.isnan(release_depth)]
    mean_depth  = float(np.nanmean(valid))
    total_vol   = float(np.nansum(valid))
    area_m2     = int(release_mask.sum())

    print(f"Release raster: area={area_m2}m²  "
          f"mean_depth={mean_depth:.2f}m  vol={total_vol:.0f}m³")

    # Resolve density from features CSV if not provided
    density = args.density
    if density is None:
        feat_csv = cfg.analysis_dir / f"release_zone_features_{args.snapshot}.csv"
        if feat_csv.exists():
            import pandas as pd
            df  = pd.read_csv(str(feat_csv))
            rel = df[df.get('group', df.columns[0]) == 'release'] \
                  if 'group' in df.columns else df
            if not rel.empty and 'slab_density' in rel.columns:
                density = float(rel['slab_density'].median())
                print(f"Density from SNOWPACK features: {density:.1f} kg/m³")
        if density is None:
            density = 300.0
            print(f"Density fallback: {density} kg/m³")

    # Write GeoTIFF
    tif_path = OUT_DIR / f"release_depth_jan18_{label}.tif"
    profile.update(dtype='float32', count=1, nodata=np.nan,
                   compress='lzw', predictor=2)
    with rasterio.open(str(tif_path), 'w', **profile) as dst:
        dst.write(release_depth[np.newaxis, ...])
    print(f"GeoTIFF: {tif_path}")

    # Write ASC
    asc_path = OUT_DIR / f"release_depth_jan18_{label}.asc"
    write_asc(release_depth, transform, nodata=-9999.0, path=asc_path)

    # Write metadata
    info = {
        "event":             "Little Professor Jan 18 2026 D2",
        "crs":               str(crs),
        "pixel_size_m":      float(transform.a),
        "release_area_m2":   area_m2,
        "mean_depth_m":      round(mean_depth, 3),
        "total_volume_m3":   round(total_vol, 0),
        "depth_source":      args.depth_source,
        "snapshot_date":     args.snapshot if args.depth_source == "snowpack" else None,
        "slab_density_kgm3": round(density, 1),
        "tif_file":          tif_path.name,
        "asc_file":          asc_path.name,
        "notes": (
            "Slab depth = 80% of SNOWPACK HS (WL ~20%). "
            "Release polygon = observed Jan 18 release area GeoJSON. "
            "CRS: EPSG:32613 (UTM 13N). "
            "Feed tif or asc to com1DFA as releaseScenario raster."
        ),
    }
    info_path = OUT_DIR / "release_info_jan18.json"
    info_path.write_text(json.dumps(info, indent=2))
    print(f"Metadata: {info_path}")
    print(f"\nReady for Val → {OUT_DIR}")


if __name__ == '__main__':
    main()
    
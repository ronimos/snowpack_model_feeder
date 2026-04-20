"""
scenario_writer.py — Write AvaFrame com1DFA scenario input files.

Provides:
    write_trigger_locations()   GeoJSON of candidate trigger points
    write_scenario()            Single scenario directory with all AvaFrame inputs
    write_scenario_weights()    JSON of per-scenario probabilities
    write_summary_csv()         One-row-per-scenario summary table
    write_metadata()            Run metadata JSON
    write_asc()                 ASCII grid (AvaFrame legacy format)

The output directory structure follows the design in step_scenarios_design.md:

    outputs/scenarios/YYYY-MM-DD/
        metadata.json
        trigger_locations.geojson
        scenario_weights.json
        summary.csv
        scenarios/
            scenario_001/
                release.geojson   release polygon + properties
                depth.tif         float32 GeoTIFF, depth (m), NaN outside
                depth.asc         ASCII grid (AvaFrame legacy)
                density.json      slab density mean + std
                params.json       com1DFA run parameters

No CLI. Called by analysis_pipeline.py step_scenarios().
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import rasterio


# -----------------------------------------------------------------------
# Utility: ASCII grid writer
# -----------------------------------------------------------------------

def write_asc(array: np.ndarray,
              transform,
              path: Path,
              nodata: float = -9999.0) -> None:
    """
    Write a float32 array as an Arc/Info ASCII grid (.asc).
    AvaFrame legacy format — used when raster depth input is unavailable.
    """
    nrows, ncols  = array.shape
    xllcorner     = transform.c
    yllcorner     = transform.f + nrows * transform.e   # e is negative
    cellsize      = transform.a

    with open(str(path), 'w') as f:
        f.write(f"ncols         {ncols}\n")
        f.write(f"nrows         {nrows}\n")
        f.write(f"xllcorner     {xllcorner:.6f}\n")
        f.write(f"yllcorner     {yllcorner:.6f}\n")
        f.write(f"cellsize      {cellsize:.6f}\n")
        f.write(f"NODATA_value  {nodata:.0f}\n")
        for row in array:
            f.write(' '.join(
                f"{nodata:.0f}" if np.isnan(v) else f"{v:.4f}"
                for v in row) + '\n')


# -----------------------------------------------------------------------
# Trigger locations GeoJSON
# -----------------------------------------------------------------------

def write_trigger_locations(triggers: pd.DataFrame,
                             cluster_map: np.ndarray,
                             transform,
                             out_path: Path) -> None:
    """
    Write candidate trigger locations as a GeoJSON FeatureCollection.

    Parameters
    ----------
    triggers    : DataFrame indexed by cluster_id with columns:
                    min_sk38, wl_shear_strength (or tau_p), sk38_rank
    cluster_map : 2D array of cluster IDs
    transform   : rasterio Affine transform (UTM)
    out_path    : output .geojson path
    """
    features = []
    for rank, (cid, row) in enumerate(triggers.iterrows(), start=1):
        pxs = np.argwhere(cluster_map == cid)
        if len(pxs) == 0:
            continue
        r, c = pxs.mean(axis=0)
        # pixel centre → UTM
        x = transform.c + c * transform.a
        y = transform.f + r * transform.e

        props = {
            'cluster_id': int(cid),
            'sk38_rank':  rank,
            'min_sk38':   float(row.get('min_sk38', np.nan)),
            'tau_p_pa':   float(row.get('wl_shear_strength',
                                         row.get('tau_p', np.nan))),
        }
        features.append({
            'type': 'Feature',
            'geometry': {'type': 'Point', 'coordinates': [x, y]},
            'properties': props,
        })

    fc = {'type': 'FeatureCollection', 'features': features}
    out_path.write_text(json.dumps(fc, indent=2))


# -----------------------------------------------------------------------
# Single scenario
# -----------------------------------------------------------------------

def write_scenario(scenario_dir: Path,
                   scenario_id: str,
                   release_polygon,
                   depth_raster: np.ndarray,
                   dem_shape: tuple,
                   transform,
                   crs_wkt: str,
                   density_mean: float,
                   density_std: float,
                   trigger_cluster_id: int,
                   A_ca: float,
                   size_factor: float,
                   depth_percentile: int,
                   weight: float,
                   mu: float = 0.155,
                   xi: float = 1500.0,
                   profile: Optional[dict] = None) -> dict:
    """
    Write one scenario directory with all AvaFrame com1DFA inputs.

    Returns a dict of scalar stats for the summary CSV row.

    Parameters
    ----------
    scenario_dir        : parent directory (scenarios/); subdir created here
    scenario_id         : e.g. 'scenario_001'
    release_polygon     : shapely Polygon (UTM) or None
    depth_raster        : float32 array (m), NaN outside release area
    dem_shape           : (nrows, ncols)
    transform           : rasterio Affine
    crs_wkt             : CRS as WKT string
    density_mean/std    : slab density (kg/m³)
    trigger_cluster_id  : cluster ID of trigger point
    A_ca                : Meloche crack arrest length (m)
    size_factor         : release size multiplier (1.0 = median)
    depth_percentile    : 10 / 50 / 90
    weight              : scenario probability weight
    mu, xi              : Voellmy friction parameters
    profile             : rasterio profile dict (if None, built from args)
    """
    out = scenario_dir / scenario_id
    out.mkdir(parents=True, exist_ok=True)

    # --- release.geojson ---
    area_m2    = 0.0
    mean_depth = 0.0
    total_vol  = 0.0

    if release_polygon is not None and not release_polygon.is_empty:
        area_m2 = float(release_polygon.area)
        props = {
            'scenario_id':       scenario_id,
            'trigger_cluster':   int(trigger_cluster_id),
            'A_ca_m':            round(float(A_ca), 2),
            'size_factor':       round(float(size_factor), 3),
            'depth_percentile':  int(depth_percentile),
            'weight':            round(float(weight), 6),
            'release_area_m2':   round(area_m2, 1),
        }
        feature = {
            'type': 'Feature',
            'geometry': release_polygon.__geo_interface__,
            'properties': props,
        }
        fc = {'type': 'FeatureCollection', 'features': [feature]}
        (out / 'release.geojson').write_text(json.dumps(fc, indent=2))

    # --- depth.tif + depth.asc ---
    valid = depth_raster[~np.isnan(depth_raster)]
    if len(valid):
        mean_depth = float(np.mean(valid))
        total_vol  = float(np.sum(valid))  # 1m² pixels

    raster_profile = profile.copy() if profile else {
        'driver': 'GTiff', 'dtype': 'float32', 'count': 1,
        'width': dem_shape[1], 'height': dem_shape[0],
        'transform': transform, 'nodata': float('nan'),
        'crs': crs_wkt,
        'compress': 'lzw', 'predictor': 2,
    }
    raster_profile.update(dtype='float32', count=1,
                           nodata=float('nan'),
                           compress='lzw', predictor=2)

    with rasterio.open(str(out / 'depth.tif'), 'w', **raster_profile) as dst:
        dst.write(depth_raster[np.newaxis, ...])

    write_asc(depth_raster, transform, out / 'depth.asc')

    # --- density.json ---
    (out / 'density.json').write_text(json.dumps({
        'mean_kgm3': round(density_mean, 1),
        'std_kgm3':  round(density_std, 1),
        'note':      'median slab_density from release zone clusters at snapshot',
    }, indent=2))

    # --- params.json ---
    (out / 'params.json').write_text(json.dumps({
        'mu':             mu,
        'xi':             xi,
        'rho_kgm3':       round(density_mean, 1),
        'release_area_m2':round(area_m2, 1),
        'mean_depth_m':   round(mean_depth, 3),
        'total_volume_m3':round(total_vol, 1),
        'A_ca_m':         round(float(A_ca), 2),
        'size_factor':    round(float(size_factor), 3),
        'depth_percentile': int(depth_percentile),
        'trigger_cluster':  int(trigger_cluster_id),
        'note': ('Voellmy parameters: defaults. '
                 'Calibrate mu/xi against Jan 18 observation before '
                 'using for operational hazard mapping.'),
    }, indent=2))

    return {
        'scenario_id':      scenario_id,
        'trigger_cluster':  int(trigger_cluster_id),
        'A_ca_m':           round(float(A_ca), 2),
        'size_factor':      round(float(size_factor), 3),
        'depth_percentile': int(depth_percentile),
        'release_area_m2':  round(area_m2, 1),
        'mean_depth_m':     round(mean_depth, 3),
        'total_volume_m3':  round(total_vol, 1),
        'weight':           round(float(weight), 6),
    }


# -----------------------------------------------------------------------
# Ensemble-level outputs
# -----------------------------------------------------------------------

def write_scenario_weights(weights: dict, out_path: Path) -> None:
    """
    Write scenario probability weights as JSON.

    Parameters
    ----------
    weights  : {scenario_id: float} summing to 1.0
    out_path : output path
    """
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6 and total > 0:
        weights = {k: v / total for k, v in weights.items()}
    out_path.write_text(json.dumps(weights, indent=2))


def write_summary_csv(rows: list[dict], out_path: Path) -> None:
    """
    Write one-row-per-scenario summary CSV.

    Parameters
    ----------
    rows     : list of dicts from write_scenario() return values
    out_path : output .csv path
    """
    df = pd.DataFrame(rows)
    col_order = [
        'scenario_id', 'trigger_cluster', 'A_ca_m', 'size_factor',
        'depth_percentile', 'release_area_m2', 'mean_depth_m',
        'total_volume_m3', 'weight',
    ]
    for col in col_order:
        if col not in df.columns:
            df[col] = np.nan
    df[col_order].to_csv(str(out_path), index=False)


def write_metadata(out_path: Path,
                   snapshot_date: str,
                   n_scenarios: int,
                   n_triggers: int,
                   size_factors: list,
                   depth_percentiles: list,
                   mu: float,
                   xi: float,
                   a_ca_stats: Optional[dict] = None) -> None:
    """
    Write run metadata JSON for the scenario ensemble.
    """
    import subprocess, datetime
    try:
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git_hash = 'unknown'

    meta = {
        'snapshot_date':    snapshot_date,
        'generated_at':     datetime.datetime.utcnow().isoformat() + 'Z',
        'git_hash':         git_hash,
        'n_scenarios':      n_scenarios,
        'n_trigger_locations': n_triggers,
        'size_factors':     size_factors,
        'depth_percentiles':depth_percentiles,
        'voellmy_mu':       mu,
        'voellmy_xi':       xi,
        'a_ca_stats':       a_ca_stats or {},
        'references': {
            'upslope_boundary':   'Meloche et al. (2025) doi:10.1029/2025JF008470',
            'downslope_boundary': 'Perzl (2007); Veitinger et al. (2016); ~28deg threshold',
            'cross_slope_flanks': 'Gaume et al. (2015) doi:10.5194/tc-9-795-2015',
        },
    }
    out_path.write_text(json.dumps(meta, indent=2))


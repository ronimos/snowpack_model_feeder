"""
cluster_update.py — Mid-season cluster splitting step.

Called after a new UAV survey lands to check whether any existing clusters
have grown too heterogeneous in their seasonal HS development and should be
split into two children.

Splitting rules:
  - Split when max-per-survey HS std > adaptive_split_threshold(median_hs)
  - Adaptive threshold = clip(0.10 × median_hs, 0.03, 0.12) metres
  - Larger child (by pixel count) inherits the parent cluster ID
  - Smaller child gets max_existing_id + 1
  - Child SMET = parent met forcing + child HS series from hourly grids
  - Child input .sno = copy of parent restart .sno (or parent input .sno)
  - Audit log written to outputs/analysis/cluster_splits.json
"""

import json
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ProjectConfig


# ---------------------------------------------------------------------------
# SMET helpers
# ---------------------------------------------------------------------------

def _read_smet(path: Path) -> tuple:
    """
    Parse a SMET 1.1 ASCII file.

    Returns
    -------
    (header_dict, met_df) where met_df columns are the SMET field names
    (HS in cm as written, TA in °C as written, etc.).
    """
    header = {}
    fields = []
    in_data = False
    data_rows = []

    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if not in_data:
                if '=' in stripped and not stripped.startswith('['):
                    key, val = stripped.split('=', 1)
                    header[key.strip()] = val.strip()
                if stripped == '[DATA]':
                    # fields line must have been parsed by now
                    fields = header.get('fields', '').split()
                    in_data = True
            else:
                parts = stripped.split()
                if parts and len(parts) == len(fields):
                    data_rows.append(parts)

    if not fields or not data_rows:
        raise ValueError(f"Could not parse SMET: {path}")

    df = pd.DataFrame(data_rows, columns=fields)
    df = df.set_index('timestamp')
    df.index = pd.to_datetime(df.index)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return header, df


def _write_child_smet(parent_smet_path: Path,
                       child_cid: int,
                       child_hs_m: np.ndarray,
                       timestamps: list,
                       smet_dir: Path) -> Path:
    """
    Write a child SMET: parent met forcing + child HS series.

    Parameters
    ----------
    parent_smet_path : path to parent SMET file
    child_cid        : new cluster ID
    child_hs_m       : child HS series in metres (len = len(timestamps))
    timestamps       : list of pd.Timestamps matching child_hs_m
    smet_dir         : output directory for SMETs

    Returns
    -------
    Path to the written child SMET file.
    """
    from smet_writer import write_smet, StationConfig

    header, met_df = _read_smet(parent_smet_path)

    child_hs_series = pd.Series(child_hs_m * 100.0, index=timestamps)
    met_df['HS'] = child_hs_series.reindex(met_df.index).interpolate(
        method='time').bfill().ffill()

    child_config = StationConfig(
        station_id=f"cluster_{child_cid:04d}",
        station_name=f"cluster_{child_cid}_split",
        latitude=float(header.get('latitude', 0.0)),
        longitude=float(header.get('longitude', 0.0)),
        altitude_m=float(header.get('altitude', 0.0)),
        slope_angle=float(header.get('slope_angle', 0.0)),
        slope_azi=float(header.get('slope_azi', 0.0)),
        tz_offset=float(header.get('tz', 0.0)),
    )

    out_path = smet_dir / f"cluster_{child_cid:04d}.smet"
    write_smet(met_df, str(out_path), child_config)
    return out_path


def _compute_child_hs_series(child_mask: np.ndarray,
                              grid_stack: np.ndarray,
                              timestamps: list) -> np.ndarray:
    """
    Compute mean HS (metres) for child pixels from the hourly grid stack.

    Parameters
    ----------
    child_mask  : (nrows, ncols) boolean mask of child pixels
    grid_stack  : (n_times, nrows, ncols) hourly HS grids in metres
    timestamps  : list of pd.Timestamps matching grid_stack axis 0 (unused here,
                  kept for caller clarity)

    Returns
    -------
    1D array of shape (n_times,), HS in metres.
    """
    rows, cols = np.where(child_mask)
    cell_hs = grid_stack[:, rows, cols]  # (n_times, n_child_cells)
    valid_cells = ~np.all(np.isnan(cell_hs), axis=0)
    if not valid_cells.any():
        raise ValueError("Child cluster has no valid cells in the hourly grids")
    cell_hs = cell_hs[:, valid_cells]
    hs_m = np.nanmean(cell_hs, axis=1)
    nan_ts = np.isnan(hs_m)
    if nan_ts.any():
        hs_m = (pd.Series(hs_m)
                .interpolate(method='linear', limit_direction='both')
                .fillna(0.0)
                .to_numpy())
    return hs_m


# ---------------------------------------------------------------------------
# Main step
# ---------------------------------------------------------------------------

def step_cluster_update(cfg: ProjectConfig, args) -> None:
    """
    Assess cluster homogeneity and split heterogeneous clusters.

    For each cluster where max-per-survey HS std exceeds its adaptive threshold:
      - Split into two children using 2-means on HS PCA.
      - Larger child inherits parent cluster ID (no re-run needed for parent).
      - Smaller child gets max_existing_id + 1.
      - Write child SMET (parent met forcing + child HS series from hourly grids).
      - Copy parent restart .sno → child input .sno (warm start for SNOWPACK).
      - Save updated cluster_map.npy + cluster_map.tif.
      - Append to cluster_splits.json audit log.

    After this step, run SNOWPACK for the new child cluster IDs only
    (use the CLUSTERS_FILE mechanism in run_snowpack.sh).
    """
    from clustering import (
        build_survey_hs_matrix,
        assess_cluster_quality,
        bisect_cluster,
    )

    cluster_map_path = cfg.cluster_map_path
    if not cluster_map_path.exists():
        print("ERROR: No cluster map found. Run 'cluster' step first.")
        return

    cluster_map = np.load(str(cluster_map_path)).copy()
    n_orig = len(np.unique(cluster_map[cluster_map > 0]))
    print(f"Loaded cluster map: {n_orig} clusters from {cluster_map_path}")

    # Load all survey grids (cumulative HS vectors for quality assessment)
    survey_grids = {}
    for npy in sorted(cfg.resampled_dir.glob("hs_*.npy")):
        date_str = npy.stem.replace("hs_", "")
        survey_grids[date_str] = np.load(str(npy))

    if not survey_grids:
        print("ERROR: No survey grids found. Run the gap_fill step first.")
        return

    print(f"Surveys: {sorted(survey_grids.keys())}")

    # Build HS matrix for all clustered cells
    valid_mask = cluster_map > 0
    hs_matrix, cell_idx, survey_dates = build_survey_hs_matrix(survey_grids, valid_mask)
    print(f"HS matrix: {hs_matrix.shape[0]} cells × {hs_matrix.shape[1]} surveys")

    # Threshold parameters (configurable via args or defaults)
    rel_frac = getattr(args, 'split_rel_frac', 0.10)
    min_abs  = getattr(args, 'split_min_abs',  0.03)
    max_abs  = getattr(args, 'split_max_abs',  0.12)
    min_size = cfg.min_cluster_size

    quality = assess_cluster_quality(
        cluster_map, hs_matrix, cell_idx,
        rel_frac=rel_frac, min_abs=min_abs, max_abs=max_abs,
        min_cluster_size=min_size,
    )

    to_split = [q for q in quality if q['should_split']]
    print(f"\n{len(quality)} clusters assessed, {len(to_split)} exceed threshold")

    if not to_split:
        print("No clusters need splitting.")
        return

    # Load hourly grids for child HS series (same logic as step_smet)
    print("\nLoading hourly grids for child HS computation...")
    grid_files = sorted(cfg.grids_dir.glob("hourly_*.npz"))
    if not grid_files:
        print("ERROR: No hourly grid files. Run 'gap_fill' first.")
        return

    all_timestamps = []
    all_grids = []
    seen: set = set()
    for gf in grid_files:
        data = np.load(str(gf))
        ts_strs = [str(ts) for ts in data['timestamps']]
        grids = data['grids']
        for t_idx, ts_str in enumerate(ts_strs):
            if ts_str in seen:
                continue
            seen.add(ts_str)
            all_timestamps.append(pd.Timestamp(ts_str))
            all_grids.append(grids[t_idx])

    sort_idx = np.argsort(all_timestamps)
    all_timestamps = [all_timestamps[i] for i in sort_idx]
    grid_stack = np.stack([all_grids[i] for i in sort_idx])
    del all_grids
    print(f"  {len(all_timestamps)} timesteps")

    # SNOWPACK file directories (must match run_snowpack.sh)
    snow_in_dir  = cfg.slope_dir / "input" / "snow"
    snow_out_dir = cfg.slope_dir / "output"
    smet_dir     = cfg.smet_dir

    # --- Perform splits (worst first by max_std) ---
    splits_log = []
    n_split = 0

    for q in sorted(to_split, key=lambda x: -x['max_std']):
        parent_cid = q['cid']
        print(f"\n  Splitting cluster {parent_cid:04d}  "
              f"n={q['n_cells']}  "
              f"std={q['max_std']:.3f} m > thresh={q['threshold']:.3f} m")

        cluster_map, child_cid = bisect_cluster(
            cluster_map, parent_cid, hs_matrix, cell_idx, min_size)

        if child_cid is None:
            print(f"    → bisect failed (cluster too small to split cleanly)")
            continue

        child_n = int(np.sum(cluster_map == child_cid))
        parent_n = int(np.sum(cluster_map == parent_cid))
        print(f"    → parent {parent_cid:04d} ({parent_n} cells) "
              f"+ child {child_cid:04d} ({child_n} cells)")

        # --- Child SMET ---
        parent_smet = smet_dir / f"cluster_{parent_cid:04d}.smet"
        if not parent_smet.exists():
            print(f"    WARNING: parent SMET not found: {parent_smet}. "
                  f"Skipping SMET/SNO for child {child_cid:04d}.")
        else:
            try:
                child_mask = cluster_map == child_cid
                child_hs_m = _compute_child_hs_series(
                    child_mask, grid_stack, all_timestamps)
                child_smet = _write_child_smet(
                    parent_smet, child_cid, child_hs_m, all_timestamps, smet_dir)
                print(f"    → child SMET: {child_smet.name}")
            except Exception as exc:
                print(f"    ERROR writing child SMET: {exc}")

        # --- Child SNO (warm start from parent restart, else parent input) ---
        parent_str = f"cluster_{parent_cid:04d}"
        child_str  = f"cluster_{child_cid:04d}"
        parent_res_sno = snow_out_dir / f"{parent_str}_{parent_str}.sno"
        parent_in_sno  = snow_in_dir  / f"{parent_str}.sno"
        child_in_sno   = snow_in_dir  / f"{child_str}.sno"

        if parent_res_sno.exists():
            shutil.copy2(str(parent_res_sno), str(child_in_sno))
            print(f"    → child SNO from restart: {child_in_sno.name}")
        elif parent_in_sno.exists():
            shutil.copy2(str(parent_in_sno), str(child_in_sno))
            print(f"    → child SNO from parent input: {child_in_sno.name}")
        else:
            print(f"    WARNING: no parent .sno found — "
                  f"SNOWPACK will use template.sno for {child_str}")

        splits_log.append({
            'parent_cid':    parent_cid,
            'child_cid':     child_cid,
            'parent_n_cells': parent_n,
            'child_n_cells':  child_n,
            'max_std':       q['max_std'],
            'threshold':     q['threshold'],
            'timestamp':     datetime.utcnow().isoformat(),
        })
        n_split += 1

    if n_split == 0:
        print("\nNo clusters were successfully split.")
        return

    # --- Save updated cluster map ---
    np.save(str(cluster_map_path), cluster_map)

    dem_path = cfg.dem_1m_path
    if dem_path.exists():
        with rasterio.open(str(dem_path)) as src:
            profile = src.profile.copy()
        profile.update(dtype='int32', count=1, compress='lzw', nodata=0)
        with rasterio.open(str(cfg.cluster_map_tif_path), 'w', **profile) as dst:
            dst.write(cluster_map, 1)

    # --- Append to splits audit log ---
    splits_json = cfg.analysis_dir / "cluster_splits.json"
    existing = []
    if splits_json.exists():
        with open(splits_json) as f:
            existing = json.load(f)
    existing.extend(splits_log)
    with open(splits_json, 'w') as f:
        json.dump(existing, f, indent=2)

    n_final = len(np.unique(cluster_map[cluster_map > 0]))
    print(f"\n{n_split} split(s) performed: {n_orig} → {n_final} clusters")
    print(f"Cluster map: {cluster_map_path}")
    print(f"Splits log:  {splits_json}")
    print(f"\nRun SNOWPACK for new child clusters:")
    for rec in splits_log:
        print(f"  cluster_{rec['child_cid']:04d}")

#!/usr/bin/env python3
"""
Distributed SNOWPACK forcing generation pipeline.

Workflow:
  1. resample   - Resample all survey GeoTIFFs to 1m reference grid
  2. transport  - Compute transport fields for all consecutive survey pairs
  3. features   - Compute Sx and terrain features for each snow surface
  4. train      - Fit Random Forest transport model with leave-one-out CV
  5. gap_fill   - Generate hourly HS grids between all surveys
  6. smet       - Write per-cell SMET forcing files

Usage:
  python batch_process.py resample
  python batch_process.py transport
  python batch_process.py features
  python batch_process.py train
  python batch_process.py gap_fill
  python batch_process.py smet
  python batch_process.py all          # run everything
  python batch_process.py validate     # LOO-CV only (after features)

Each step saves intermediate results to outputs/, so you can
restart from any step without recomputing previous ones.
"""

import sys
import argparse
import json
import re
import pickle
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import ProjectConfig
from spatial_model import (
    create_reference_dem, load_and_resample,
    compute_sx_multi, compute_wind_weighted_sx,
    compute_terrain_features, compute_transport_field,
    smooth_field, build_feature_array, compute_wind_stats,
    train_transport_model, predict_transport,
    gap_fill_period, validate_prediction,
)
from smet_writer import load_and_convert, write_smet, StationConfig


# =====================================================================
# Helpers
# =====================================================================

def parse_survey_date(filename: str) -> datetime.date:
    """Extract date from survey filename like '260114_Professor_PTC_snowHeight.tif'."""
    match = re.match(r'^(\d{6})', filename)
    if not match:
        raise ValueError(f"Cannot parse date from: {filename}")
    ds = match.group(1)
    year = 2000 + int(ds[:2])
    month = int(ds[2:4])
    day = int(ds[4:6])
    return datetime.date(year, month, day)


def discover_surveys(cfg: ProjectConfig) -> list:
    """Find all survey files and return sorted (date, path) list."""
    survey_files = sorted(cfg.survey_dir.glob(cfg.survey_glob))
    surveys = []
    for f in survey_files:
        d = parse_survey_date(f.name)
        surveys.append((d, f))
    surveys.sort(key=lambda x: x[0])
    print(f"Found {len(surveys)} surveys:")
    for d, f in surveys:
        print(f"  {d.isoformat()}  {f.name}")
    return surveys


def load_weather(cfg: ProjectConfig) -> pd.DataFrame:
    """Load and convert weather data."""
    wx = load_and_convert(str(cfg.weather_csv), tz_output="UTC")
    print(f"Weather: {wx.index[0]} → {wx.index[-1]}, {len(wx)} hours")
    return wx


def station_hs_at(wx: pd.DataFrame, date: datetime.date,
                  hour_utc: int = 18) -> float:
    """Get station HS (in meters) at survey flight time."""
    ts = pd.Timestamp(datetime.datetime.combine(date, datetime.time(hour_utc)))
    val = wx['HS'].asof(ts)
    if pd.isna(val):
        # Try nearest within 6 hours
        window = wx['HS'].loc[ts - pd.Timedelta(hours=6): ts + pd.Timedelta(hours=6)]
        if len(window) > 0:
            val = window.iloc[len(window)//2]
    if pd.isna(val):
        print(f"  WARNING: No station HS available for {date}")
        return np.nan
    return val / 100.0  # cm → m


# =====================================================================
# Step 1: Resample
# =====================================================================

def step_resample(cfg: ProjectConfig):
    """Resample bare ground DEM and all surveys to 1m reference grid."""
    cfg.ensure_dirs()
    surveys = discover_surveys(cfg)

    ref_dem_path = cfg.resampled_dir / "dem_1m.tif"

    # Create reference 1m DEM
    if not ref_dem_path.exists():
        print(f"\nCreating 1m reference DEM from {cfg.dem_path.name}...")
        create_reference_dem(str(cfg.dem_path), str(ref_dem_path),
                             cfg.target_resolution_m)
    else:
        print(f"Reference DEM exists: {ref_dem_path}")

    # Resample each survey
    for date, path in surveys:
        out_path = cfg.resampled_dir / f"hs_{date.isoformat()}.npy"
        if out_path.exists():
            print(f"  {date}: exists, skipping")
            continue

        print(f"  {date}: resampling {path.name}...")
        hs = load_and_resample(str(path), str(ref_dem_path),
                               clip_min=cfg.min_valid_hs_m,
                               clip_max=cfg.max_valid_hs_m)
        valid = np.sum(~np.isnan(hs))
        median = np.nanmedian(hs)
        print(f"    valid: {valid}, median: {median:.3f} m")
        np.save(str(out_path), hs)

    print("\nResample complete.")


# =====================================================================
# Step 2: Compute transport fields
# =====================================================================

def step_transport(cfg: ProjectConfig):
    """Compute transport fields for all consecutive survey pairs."""
    cfg.ensure_dirs()
    surveys = discover_surveys(cfg)
    wx = load_weather(cfg)

    import rasterio
    ref_dem_path = cfg.resampled_dir / "dem_1m.tif"
    with rasterio.open(str(ref_dem_path)) as src:
        dem = src.read(1)
        dem[dem == src.nodata] = np.nan

    # Load all survey grids
    grids = {}
    for date, _ in surveys:
        npy = cfg.resampled_dir / f"hs_{date.isoformat()}.npy"
        if npy.exists():
            grids[date] = np.load(str(npy))
        else:
            print(f"  WARNING: {npy} not found, run 'resample' first")
            return

    # Compute transport for each consecutive pair
    transport_data = []
    dates = sorted(grids.keys())

    for i in range(len(dates) - 1):
        d_a, d_b = dates[i], dates[i + 1]
        hs_a, hs_b = grids[d_a], grids[d_b]

        stn_a = station_hs_at(wx, d_a, cfg.flight_hour_utc)
        stn_b = station_hs_at(wx, d_b, cfg.flight_hour_utc)

        has_station = not (np.isnan(stn_a) or np.isnan(stn_b))
        stn_dhs = (stn_b - stn_a) if has_station else np.nan

        valid = ~np.isnan(dem) & ~np.isnan(hs_a) & ~np.isnan(hs_b)

        if has_station:
            transport_raw = compute_transport_field(hs_a, hs_b, stn_a, stn_b, valid)
        else:
            # No station data — transport is undefined, save NaN field
            transport_raw = np.full_like(dem, np.nan)
        transport_smooth = smooth_field(transport_raw, cfg.transport_smoothing_window_m)

        # Wind stats for this period
        t0 = pd.Timestamp(datetime.datetime.combine(d_a, datetime.time(cfg.flight_hour_utc)))
        t1 = pd.Timestamp(datetime.datetime.combine(d_b, datetime.time(cfg.flight_hour_utc)))
        wp = wx.loc[t0:t1]
        wstats = compute_wind_stats(wp)

        pair_id = f"{d_a.isoformat()}__{d_b.isoformat()}"
        np.save(str(cfg.analysis_dir / f"transport_raw_{pair_id}.npy"), transport_raw)
        np.save(str(cfg.analysis_dir / f"transport_smooth_{pair_id}.npy"), transport_smooth)

        transport_data.append({
            'pair_id': pair_id,
            'date_a': d_a.isoformat(),
            'date_b': d_b.isoformat(),
            'stn_hs_a': None if np.isnan(stn_a) else float(stn_a),
            'stn_hs_b': None if np.isnan(stn_b) else float(stn_b),
            'stn_dhs': None if np.isnan(stn_dhs) else float(stn_dhs),
            'transport_median': float(np.nanmedian(transport_raw)) if has_station else None,
            'transport_std': float(np.nanstd(transport_raw)) if has_station else None,
            'n_valid': int(valid.sum()),
            'wind_mean_speed': wstats['mean_speed'] if wstats.get('has_data') else None,
            'wind_mean_dir': wstats['mean_dir'] if wstats.get('has_data') else None,
            'wind_total_energy': wstats['total_wind_energy'] if wstats.get('has_data') else None,
            'n_hours': len(wp),
            'has_weather': has_station and wstats.get('has_data', False),
        })

        if has_station and wstats.get('has_data'):
            print(f"  {pair_id}: stn_dhs={stn_dhs*100:+.1f}cm, "
                  f"transport median={np.nanmedian(transport_raw)*100:+.1f}cm, "
                  f"wind={wstats['mean_speed']:.1f}m/s@{wstats['mean_dir']:.0f}° [OK]")
        else:
            reasons = []
            if not has_station:
                reasons.append("no station HS")
            if not wstats.get('has_data'):
                reasons.append("no wind data")
            print(f"  {pair_id}: INCOMPLETE ({', '.join(reasons)})")

    # Save metadata
    with open(str(cfg.analysis_dir / "transport_metadata.json"), 'w') as f:
        json.dump(transport_data, f, indent=2)

    print(f"\nTransport fields computed for {len(transport_data)} pairs.")


# =====================================================================
# Step 3: Compute Sx and terrain features
# =====================================================================

def step_features(cfg: ProjectConfig):
    """Compute Sx from each snow surface DEM + terrain features."""
    cfg.ensure_dirs()
    surveys = discover_surveys(cfg)
    wx = load_weather(cfg)

    import rasterio
    ref_dem_path = cfg.resampled_dir / "dem_1m.tif"
    with rasterio.open(str(ref_dem_path)) as src:
        dem = src.read(1)
        dem[dem == src.nodata] = np.nan

    # Load transport metadata for date pairs
    meta_path = cfg.analysis_dir / "transport_metadata.json"
    with open(str(meta_path)) as f:
        transport_meta = json.load(f)

    grids = {}
    for date, _ in surveys:
        npy = cfg.resampled_dir / f"hs_{date.isoformat()}.npy"
        if npy.exists():
            grids[date] = np.load(str(npy))

    dates = sorted(grids.keys())
    fill_dem = np.where(np.isnan(dem), np.nanmean(dem), dem)

    for i in range(len(dates) - 1):
        d_a = dates[i]
        pair_id = f"{dates[i].isoformat()}__{dates[i+1].isoformat()}"
        sx_path = cfg.analysis_dir / f"sx_dict_{d_a.isoformat()}.pkl"
        feat_path = cfg.analysis_dir / f"features_{d_a.isoformat()}.pkl"
        wwsx_path = cfg.analysis_dir / f"wwsx_{pair_id}.npy"

        # Snow surface DEM at start of period
        hs_a = grids[d_a]
        snow_surface = np.where(~np.isnan(dem) & ~np.isnan(hs_a),
                                dem + hs_a, fill_dem)

        # Compute Sx (skip if cached)
        if sx_path.exists():
            print(f"  Sx for {d_a}: cached")
            with open(str(sx_path), 'rb') as f:
                sx_dict = pickle.load(f)
        else:
            print(f"  Computing Sx for {d_a} snow surface ({len(cfg.sx_azimuths_deg)} directions)...")
            sx_dict = compute_sx_multi(snow_surface, cfg.sx_azimuths_deg,
                                       cfg.sx_search_distance_m,
                                       cfg.target_resolution_m)
            with open(str(sx_path), 'wb') as f:
                pickle.dump(sx_dict, f)

        # Terrain features (skip if cached)
        if feat_path.exists():
            print(f"  Terrain features for {d_a}: cached")
        else:
            print(f"  Computing terrain features for {d_a}...")
            terrain = compute_terrain_features(snow_surface, cfg.target_resolution_m)
            with open(str(feat_path), 'wb') as f:
                pickle.dump(terrain, f)

        # Wind-weighted Sx for this specific period
        if not wwsx_path.exists():
            meta = next(m for m in transport_meta if m['pair_id'] == pair_id)
            t0 = pd.Timestamp(f"{meta['date_a']} {cfg.flight_hour_utc}:00")
            t1 = pd.Timestamp(f"{meta['date_b']} {cfg.flight_hour_utc}:00")
            wp = wx.loc[t0:t1]
            if len(wp) < 2:
                print(f"  WARNING: No weather data for {pair_id}, "
                      f"wwsx will be mean across all directions")
            wwsx = compute_wind_weighted_sx(sx_dict, wp)
            np.save(str(wwsx_path), wwsx)

    print("\nFeature computation complete.")


# =====================================================================
# Step 4: Train model with LOO-CV
# =====================================================================

def step_train(cfg: ProjectConfig):
    """Train Random Forest on all pairs, with leave-one-out CV."""
    cfg.ensure_dirs()

    import rasterio
    ref_dem_path = cfg.resampled_dir / "dem_1m.tif"
    with rasterio.open(str(ref_dem_path)) as src:
        dem = src.read(1)
        dem[dem == src.nodata] = np.nan

    with open(str(cfg.analysis_dir / "transport_metadata.json")) as f:
        transport_meta = json.load(f)

    wx = load_weather(cfg)

    # Load all features and transport fields
    print("Loading feature data for all periods...")
    period_data = []
    skipped = []
    for meta in transport_meta:
        pair_id = meta['pair_id']
        d_a = meta['date_a']

        # Skip periods without weather coverage
        if not meta.get('has_weather', True):
            skipped.append(f"{pair_id} (no weather data)")
            continue

        # Check for NaN wind stats
        wind_speed = meta.get('wind_mean_speed')
        wind_dir = meta.get('wind_mean_dir')
        if wind_speed is None or wind_dir is None:
            skipped.append(f"{pair_id} (missing wind stats)")
            continue

        # Transport (target)
        transport_path = cfg.analysis_dir / f"transport_smooth_{pair_id}.npy"
        if not transport_path.exists():
            skipped.append(f"{pair_id} (transport file missing)")
            continue
        transport = np.load(str(transport_path))

        # Terrain features
        feat_path = cfg.analysis_dir / f"features_{d_a}.pkl"
        if not feat_path.exists():
            skipped.append(f"{pair_id} (feature file missing)")
            continue
        with open(str(feat_path), 'rb') as f:
            terrain = pickle.load(f)

        # Wind-weighted Sx
        wwsx_path = cfg.analysis_dir / f"wwsx_{pair_id}.npy"
        if not wwsx_path.exists():
            skipped.append(f"{pair_id} (wwsx file missing)")
            continue
        wwsx = np.load(str(wwsx_path))

        # Wind stats
        wstats = {
            'mean_speed': meta['wind_mean_speed'],
            'dir_sin': np.sin(np.radians(meta['wind_mean_dir'])),
            'dir_cos': np.cos(np.radians(meta['wind_mean_dir'])),
        }

        # Valid mask — must have non-NaN in all arrays
        valid = (~np.isnan(dem) & ~np.isnan(transport) &
                 ~np.isnan(wwsx) & ~np.isnan(terrain.get('slope', dem)))
        n_valid = int(valid.sum())

        if n_valid < 100:
            skipped.append(f"{pair_id} (only {n_valid} valid cells)")
            continue

        period_data.append({
            'pair_id': pair_id,
            'meta': meta,
            'transport': transport,
            'terrain': terrain,
            'wwsx': wwsx,
            'wstats': wstats,
            'valid': valid,
        })

    n_periods = len(period_data)
    if skipped:
        print(f"Skipped {len(skipped)} periods:")
        for s in skipped:
            print(f"    {s}")
    print(f"Using {n_periods} periods for training")

    # --- Leave-one-out cross-validation ---
    print("\nLeave-one-out cross-validation:")
    cv_results = []

    for hold_idx in range(n_periods):
        hold = period_data[hold_idx]
        train_periods = [p for i, p in enumerate(period_data) if i != hold_idx]

        # Build training set
        X_parts, y_parts = [], []
        for p in train_periods:
            if p['valid'].sum() == 0:
                continue
            X, names = build_feature_array(p['terrain'], p['wwsx'],
                                            p['wstats'], p['valid'])
            y = p['transport'].ravel()[np.where(p['valid'].ravel())[0]]
            if len(y) > 0:
                X_parts.append(X)
                y_parts.append(y)

        if len(X_parts) == 0:
            print(f"  {hold['pair_id']}: SKIPPED (no valid training data)")
            cv_results.append({'pair_id': hold['pair_id'],
                               'rmse': np.nan, 'bias': np.nan,
                               'mae': np.nan, 'r': np.nan, 'n': 0})
            continue

        X_train = np.vstack(X_parts)
        y_train = np.concatenate(y_parts)

        # Train
        model = train_transport_model(X_train, y_train)

        # Predict held-out period
        if hold['valid'].sum() == 0:
            print(f"  {hold['pair_id']}: SKIPPED (no valid cells in hold-out)")
            cv_results.append({'pair_id': hold['pair_id'],
                               'rmse': np.nan, 'bias': np.nan,
                               'mae': np.nan, 'r': np.nan, 'n': 0})
            continue

        pred_transport = predict_transport(model, hold['terrain'], hold['wwsx'],
                                           hold['wstats'], hold['valid'])

        # Validate transport field
        metrics = validate_prediction(pred_transport, hold['transport'], hold['valid'])

        cv_results.append({
            'pair_id': hold['pair_id'],
            **{k: float(v) if isinstance(v, (np.floating, float)) else
               int(v) if isinstance(v, (np.integer, int)) else v
               for k, v in metrics.items()},
        })

        print(f"  {hold['pair_id']}: RMSE={metrics['rmse']*100:.1f}cm, "
              f"r={metrics['r']:.3f}, bias={metrics['bias']*100:.1f}cm")

    # Summary
    rmses = [r['rmse'] for r in cv_results if not np.isnan(r['rmse'])]
    rs = [r['r'] for r in cv_results if not np.isnan(r['r'])]
    print(f"\nLOO-CV Summary ({len(rmses)} periods):")
    print(f"  RMSE: mean={np.mean(rmses)*100:.1f}cm, "
          f"median={np.median(rmses)*100:.1f}cm")
    print(f"  r:    mean={np.mean(rs):.3f}, median={np.median(rs):.3f}")

    # Save CV results
    with open(str(cfg.analysis_dir / "cv_results.json"), 'w') as f:
        json.dump(cv_results, f, indent=2)

    # Train final model on all data
    print("\nTraining final model on all data...")
    X_all, y_all = [], []
    for p in period_data:
        X, names = build_feature_array(p['terrain'], p['wwsx'],
                                        p['wstats'], p['valid'])
        y = p['transport'].ravel()[np.where(p['valid'].ravel())[0]]
        X_all.append(X)
        y_all.append(y)

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)
    print(f"  Training set: {X_all.shape[0]} samples, {X_all.shape[1]} features")

    final_model = train_transport_model(X_all, y_all)

    # Feature importance
    print(f"\n  Feature importance:")
    for name, imp in sorted(zip(names, final_model.feature_importances_),
                             key=lambda x: -x[1]):
        print(f"    {name:25s}: {imp:.3f}")

    # Save model
    model_path = cfg.analysis_dir / "transport_model.pkl"
    with open(str(model_path), 'wb') as f:
        pickle.dump({'model': final_model, 'feature_names': names}, f)
    print(f"\nModel saved to {model_path}")


# =====================================================================
# Step 5: Gap-fill
# =====================================================================

def step_gap_fill(cfg: ProjectConfig, use_model: bool = True):
    """Generate hourly HS grids between all surveys."""
    cfg.ensure_dirs()

    import rasterio
    ref_dem_path = cfg.resampled_dir / "dem_1m.tif"
    with rasterio.open(str(ref_dem_path)) as src:
        dem = src.read(1)
        dem[dem == src.nodata] = np.nan

    with open(str(cfg.analysis_dir / "transport_metadata.json")) as f:
        transport_meta = json.load(f)

    wx = load_weather(cfg)

    # Load model if using regression
    model = None
    if use_model:
        model_path = cfg.analysis_dir / "transport_model.pkl"
        if model_path.exists():
            with open(str(model_path), 'rb') as f:
                model_data = pickle.load(f)
                model = model_data['model']
            print("Using trained RF model for transport prediction")
        else:
            print("No trained model found, using smoothed observed transport")
            use_model = False

    total_grids = 0
    all_grids = {}
    all_metrics = []

    for meta in transport_meta:
        pair_id = meta['pair_id']
        d_a = datetime.date.fromisoformat(meta['date_a'])
        d_b = datetime.date.fromisoformat(meta['date_b'])

        # Skip periods without weather data
        if not meta.get('has_weather', True):
            print(f"\nSkipping {pair_id}: no weather data coverage")
            continue

        print(f"\nGap-filling {pair_id}...")

        # Load survey grids
        hs_a = np.load(str(cfg.resampled_dir / f"hs_{d_a.isoformat()}.npy"))
        hs_b = np.load(str(cfg.resampled_dir / f"hs_{d_b.isoformat()}.npy"))
        valid = ~np.isnan(dem) & ~np.isnan(hs_a) & ~np.isnan(hs_b)

        # Get transport field
        if use_model:
            # Predict transport from terrain + wind
            feat_path = cfg.analysis_dir / f"features_{d_a.isoformat()}.pkl"
            wwsx_path = cfg.analysis_dir / f"wwsx_{pair_id}.npy"
            with open(str(feat_path), 'rb') as f:
                terrain = pickle.load(f)
            wwsx = np.load(str(wwsx_path))
            wstats = {
                'mean_speed': meta['wind_mean_speed'],
                'dir_sin': np.sin(np.radians(meta['wind_mean_dir'])),
                'dir_cos': np.cos(np.radians(meta['wind_mean_dir'])),
            }
            transport = predict_transport(model, terrain, wwsx, wstats, valid)
        else:
            transport = np.load(str(cfg.analysis_dir / f"transport_smooth_{pair_id}.npy"))

        # Station HS series for this period
        t0 = pd.Timestamp(datetime.datetime.combine(d_a, datetime.time(cfg.flight_hour_utc)))
        t1 = pd.Timestamp(datetime.datetime.combine(d_b, datetime.time(cfg.flight_hour_utc)))
        stn_hs = wx.loc[t0:t1, 'HS'] / 100.0  # cm → m
        wind_speed = wx.loc[t0:t1, 'VW']

        if len(stn_hs) < 2:
            print(f"  Insufficient station data ({len(stn_hs)} hours), skipping")
            continue

        # Gap-fill
        grids = gap_fill_period(hs_a, transport, stn_hs, wind_speed, valid)

        # Validate at endpoint
        pred_end = grids[t1]
        metrics = validate_prediction(pred_end, hs_b, valid)
        metrics['pair_id'] = pair_id
        metrics['date_a'] = d_a.isoformat()
        metrics['date_b'] = d_b.isoformat()
        metrics['stn_dhs'] = float(meta.get('stn_dhs', 0) or 0)
        metrics['n_hours'] = len(grids)
        # Store arrays for plotting
        metrics['pred'] = pred_end
        metrics['obs'] = hs_b
        metrics['valid'] = valid
        all_metrics.append(metrics)

        print(f"  {len(grids)} hours, endpoint RMSE={metrics['rmse']*100:.1f}cm, "
              f"r={metrics['r']:.3f}")

        # Save hourly grids as compressed npz
        out_path = cfg.grids_dir / f"hourly_{pair_id}.npz"
        timestamps = list(grids.keys())
        grid_stack = np.stack([grids[ts] for ts in timestamps])
        np.savez_compressed(str(out_path),
                            grids=grid_stack,
                            timestamps=np.array([str(ts) for ts in timestamps]))
        total_grids += len(grids)
        print(f"  Saved to {out_path}")

    print(f"\nGap-fill complete. {total_grids} total hourly grids.")

    # --- Validation summary plot ---
    if all_metrics:
        _plot_gap_fill_validation(all_metrics, cfg)


def _plot_gap_fill_validation(all_metrics: list, cfg):
    """Generate validation plots for gap-fill results."""
    import rasterio

    valid_metrics = [m for m in all_metrics if not np.isnan(m.get('r', np.nan))]
    n = len(valid_metrics)
    if n == 0:
        return

    # --- Figure 1: Summary bar chart + scatter grid ---
    n_scatter = min(n, 12)  # limit scatter panels
    n_cols = min(4, n_scatter)
    n_rows = (n_scatter + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows + 1, n_cols, figsize=(5 * n_cols, 4 * (n_rows + 1)))
    if n_rows + 1 == 1:
        axes = axes[np.newaxis, :]

    # Top row: RMSE and r² bar charts across the full width
    ax_rmse = fig.add_subplot(n_rows + 1, 2, 1)
    ax_r2 = fig.add_subplot(n_rows + 1, 2, 2)

    # Clear the grid axes in the top row since we're using add_subplot
    for j in range(n_cols):
        axes[0, j].set_visible(False)

    labels = [m['date_b'] for m in valid_metrics]
    rmses = [m['rmse'] * 100 for m in valid_metrics]
    r2s = [m['r'] ** 2 if not np.isnan(m['r']) else 0 for m in valid_metrics]

    colors_rmse = ['#d32f2f' if r > 100 else '#ff9800' if r > 60 else '#4caf50'
                   for r in rmses]
    ax_rmse.bar(range(n), rmses, color=colors_rmse, edgecolor='black', linewidth=0.5)
    ax_rmse.set_xticks(range(n))
    ax_rmse.set_xticklabels(labels, rotation=60, fontsize=7, ha='right')
    ax_rmse.set_ylabel('RMSE (cm)')
    ax_rmse.set_title('Endpoint RMSE by period')
    ax_rmse.axhline(np.median(rmses), color='black', linestyle='--', linewidth=0.8,
                     label=f'median={np.median(rmses):.0f} cm')
    ax_rmse.legend(fontsize=8)

    colors_r2 = ['#d32f2f' if r < 0.3 else '#ff9800' if r < 0.6 else '#4caf50'
                  for r in r2s]
    ax_r2.bar(range(n), r2s, color=colors_r2, edgecolor='black', linewidth=0.5)
    ax_r2.set_xticks(range(n))
    ax_r2.set_xticklabels(labels, rotation=60, fontsize=7, ha='right')
    ax_r2.set_ylabel('R²')
    ax_r2.set_title('Endpoint R² by period')
    ax_r2.axhline(np.median(r2s), color='black', linestyle='--', linewidth=0.8,
                   label=f'median={np.median(r2s):.2f}')
    ax_r2.set_ylim(0, 1.05)
    ax_r2.legend(fontsize=8)

    # Scatter panels: observed vs predicted for each period
    for idx in range(n_scatter):
        row = 1 + idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        m = valid_metrics[idx]
        mask = m['valid'] & ~np.isnan(m['pred']) & ~np.isnan(m['obs'])
        obs = m['obs'][mask].ravel()
        pred = m['pred'][mask].ravel()

        # Subsample for plotting speed
        step = max(1, len(obs) // 5000)
        ax.scatter(obs[::step], pred[::step], s=1, alpha=0.2, c='steelblue')
        lim = max(np.percentile(obs, 99), np.percentile(pred, 99), 0.5)
        ax.plot([0, lim], [0, lim], 'r--', linewidth=0.8)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect('equal')
        ax.set_title(f"{m['date_b']}\nR²={m['r']**2:.2f}, RMSE={m['rmse']*100:.0f}cm",
                      fontsize=8)
        ax.set_xlabel('Observed (m)', fontsize=7)
        ax.set_ylabel('Predicted (m)', fontsize=7)
        ax.tick_params(labelsize=6)

    # Hide unused scatter panels
    for idx in range(n_scatter, n_rows * n_cols):
        row = 1 + idx // n_cols
        col = idx % n_cols
        if row < axes.shape[0] and col < axes.shape[1]:
            axes[row, col].set_visible(False)

    plt.tight_layout()
    out_path = cfg.analysis_dir / "gap_fill_validation.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nValidation plot saved to {out_path}")

    # Print summary table
    print(f"\n{'Period':<35s} {'RMSE(cm)':>10s} {'R²':>8s} {'Bias(cm)':>10s} {'Hours':>6s}")
    print("-" * 75)
    for m in valid_metrics:
        r2 = m['r'] ** 2 if not np.isnan(m['r']) else np.nan
        print(f"{m['pair_id']:<35s} {m['rmse']*100:>10.1f} {r2:>8.3f} "
              f"{m['bias']*100:>10.1f} {m['n_hours']:>6d}")
    med_rmse = np.median([m['rmse'] * 100 for m in valid_metrics])
    med_r2 = np.median([m['r'] ** 2 for m in valid_metrics if not np.isnan(m['r'])])
    print("-" * 75)
    print(f"{'Median':<35s} {med_rmse:>10.1f} {med_r2:>8.3f}")


# =====================================================================
# Step 6: Build domain mask and cluster cells
# =====================================================================

def step_cluster(cfg: ProjectConfig):
    """Build domain mask from KML boundary, cluster cells by HS evolution."""
    cfg.ensure_dirs()

    import rasterio
    from clustering import (
        build_domain_mask, build_survey_hs_matrix, cluster_cells,
        auto_select_n_clusters
    )

    ref_dem_path = cfg.resampled_dir / "dem_1m.tif"
    with rasterio.open(str(ref_dem_path)) as src:
        dem = src.read(1)
        dem[dem == src.nodata] = np.nan
        transform = src.transform
        crs = src.crs

    # --- Build domain mask ---
    print("Building domain mask...")
    domain_mask = build_domain_mask(
        dem, transform, crs,
        kml_path=str(cfg.boundary_kml),
        min_slope_deg=cfg.min_slope_deg,
        resolution=cfg.target_resolution_m,
    )

    # Save mask
    mask_path = cfg.analysis_dir / "domain_mask.npy"
    np.save(str(mask_path), domain_mask)
    print(f"  Mask saved: {domain_mask.sum()} cells")

    # --- Cluster by survey HS evolution ---
    print("\nClustering cells by HS evolution...")
    survey_grids = {}
    for npy in sorted(cfg.resampled_dir.glob("hs_*.npy")):
        date_str = npy.stem.replace("hs_", "")
        survey_grids[date_str] = np.load(str(npy))

    hs_matrix, cell_idx, survey_dates = build_survey_hs_matrix(
        survey_grids, domain_mask)
    print(f"  HS matrix: {hs_matrix.shape[0]} cells × {hs_matrix.shape[1]} surveys")

    n_clusters = cfg.n_clusters_override or auto_select_n_clusters(
        len(cell_idx), cfg.target_cells_per_cluster)
    cluster_map = cluster_cells(hs_matrix, cell_idx, dem.shape,
                                 n_clusters=n_clusters)

    # Save cluster map as .npy and as GeoTIFF
    np.save(str(cfg.analysis_dir / "cluster_map.npy"), cluster_map)

    profile = {
        'driver': 'GTiff', 'dtype': 'int32', 'compress': 'lzw',
        'height': dem.shape[0], 'width': dem.shape[1], 'count': 1,
        'crs': crs, 'transform': transform, 'nodata': 0,
    }
    cluster_tif = cfg.analysis_dir / "cluster_map.tif"
    with rasterio.open(str(cluster_tif), 'w', **profile) as dst:
        dst.write(cluster_map, 1)

    cids = np.unique(cluster_map[cluster_map > 0])
    sizes = [int(np.sum(cluster_map == c)) for c in cids]
    print(f"\n  {len(cids)} clusters saved to {cluster_tif}")
    print(f"  Sizes: min={min(sizes)}, median={np.median(sizes):.0f}, max={max(sizes)}")

    # --- Visualization ---
    _plot_cluster_map(cluster_map, dem, transform, cids, sizes,
                       hs_matrix, cell_idx, survey_dates, cfg)


def _plot_cluster_map(cluster_map, dem, transform, cids, sizes,
                       hs_matrix, cell_idx, survey_dates, cfg):
    """Generate cluster visualization PNG."""
    bounds = [transform[2], transform[2] + dem.shape[1],
              transform[5] + dem.shape[0] * transform[4], transform[5]]

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    # 1) Cluster map on DEM
    ax = axes[0]
    # Hillshade background
    from matplotlib.colors import LightSource
    fill = np.where(np.isnan(dem), np.nanmean(dem), dem)
    ls = LightSource(azdeg=315, altdeg=45)
    hillshade = ls.hillshade(fill, dx=1.0, dy=1.0)
    ax.imshow(hillshade, cmap='gray', extent=bounds, alpha=0.6)
    # Clusters
    display = np.where(cluster_map > 0, cluster_map, np.nan)
    ax.imshow(display, cmap='nipy_spectral', interpolation='nearest',
              extent=bounds, alpha=0.7)
    ax.set_title(f'{len(cids)} clusters\n({int(np.sum(cluster_map > 0))} cells)')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')

    # 2) Sample cluster HS trajectories
    ax = axes[1]
    np.random.seed(42)
    # Sort clusters by median HS for visual clarity
    cluster_median_hs = []
    for c in cids:
        rows_in = hs_matrix[np.isin(np.arange(len(cell_idx)),
                            np.where((cluster_map[cell_idx[:, 0], cell_idx[:, 1]] == c))[0])]
        if len(rows_in) > 0:
            cluster_median_hs.append(np.median(rows_in[:, -1]))
        else:
            cluster_median_hs.append(0)

    sample_n = min(30, len(cids))
    sample_idx = np.random.choice(len(cids), sample_n, replace=False)
    for idx in sample_idx:
        c = cids[idx]
        mask_c = cluster_map[cell_idx[:, 0], cell_idx[:, 1]] == c
        if mask_c.any():
            mean_hs = np.mean(hs_matrix[mask_c], axis=0)
            ax.plot(range(len(survey_dates)), mean_hs, alpha=0.5, linewidth=1)

    ax.set_xticks(range(len(survey_dates)))
    short_dates = [d[-5:] if len(d) > 5 else d for d in survey_dates]
    ax.set_xticklabels(short_dates, rotation=60, fontsize=7, ha='right')
    ax.set_ylabel('HS (m)')
    ax.set_title(f'Cluster mean HS trajectories\n({sample_n} of {len(cids)} shown)')
    ax.set_xlim(-0.5, len(survey_dates) - 0.5)

    # 3) Cluster size distribution
    ax = axes[2]
    ax.hist(sizes, bins=max(10, len(cids) // 5), edgecolor='black',
            alpha=0.7, color='steelblue')
    ax.set_xlabel('Cells per cluster')
    ax.set_ylabel('Count')
    ax.set_title(f'Cluster sizes\nmedian={int(np.median(sizes))}, '
                 f'range=[{min(sizes)}, {max(sizes)}]')
    ax.axvline(np.median(sizes), color='red', linestyle='--', linewidth=1.5,
               label=f'median={int(np.median(sizes))}')
    ax.legend()

    plt.tight_layout()
    out_path = cfg.analysis_dir / "cluster_map.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Cluster plot saved to {out_path}")


# =====================================================================
# Step 7: Write SMET files
# =====================================================================

def step_smet(cfg: ProjectConfig):
    """Generate per-cluster SMET forcing files from hourly grids + saved clusters."""
    cfg.ensure_dirs()

    import rasterio
    from clustering import compute_cluster_representatives

    ref_dem_path = cfg.resampled_dir / "dem_1m.tif"
    with rasterio.open(str(ref_dem_path)) as src:
        dem = src.read(1)
        dem[dem == src.nodata] = np.nan
        transform = src.transform
        crs = src.crs

    wx = load_weather(cfg)

    # Terrain
    fill_dem = np.where(np.isnan(dem), np.nanmean(dem), dem)
    dy, dx = np.gradient(fill_dem, cfg.target_resolution_m)
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    aspect = np.degrees(np.arctan2(-dx, dy)) % 360

    # --- Load saved cluster map ---
    cluster_path = cfg.analysis_dir / "cluster_map.npy"
    if not cluster_path.exists():
        print("ERROR: No cluster map found. Run 'cluster' step first.")
        return
    cluster_map = np.load(str(cluster_path))
    cids = np.unique(cluster_map[cluster_map > 0])
    print(f"Loaded cluster map: {len(cids)} clusters")

    # --- Load hourly grids ---
    print("Loading hourly grids...")
    grid_files = sorted(cfg.grids_dir.glob("hourly_*.npz"))
    if not grid_files:
        print("ERROR: No hourly grid files. Run 'gap_fill' first.")
        return

    all_timestamps = []
    all_grids = []
    seen = set()

    for gf in grid_files:
        data = np.load(str(gf))
        timestamps = [str(ts) for ts in data['timestamps']]
        grids = data['grids']
        for t_idx, ts_str in enumerate(timestamps):
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

    # --- Compute cluster representatives ---
    print("Computing cluster representative HS series...")
    representatives = compute_cluster_representatives(
        cluster_map, grid_stack, all_timestamps)
    del grid_stack

    # --- Weather forcing ---
    wx_forcing = wx.reindex(pd.DatetimeIndex(all_timestamps),
                             method='nearest',
                             tolerance=pd.Timedelta(hours=1))

    # --- Coordinate conversion ---
    try:
        from pyproj import Transformer
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        has_proj = True
    except ImportError:
        print("  WARNING: pyproj not installed, using UTM coords")
        has_proj = False

    # --- Write SMET files ---
    smet_dir = cfg.smet_dir
    smet_dir.mkdir(parents=True, exist_ok=True)

    for cid, rep in representatives.items():
        cr, cc = rep['centroid_row'], rep['centroid_col']
        ri = max(0, min(int(round(cr)), dem.shape[0] - 1))
        ci = max(0, min(int(round(cc)), dem.shape[1] - 1))

        cell_easting = transform[2] + cc * transform[0]
        cell_northing = transform[5] + cr * transform[4]
        cell_alt = float(dem[ri, ci]) if not np.isnan(dem[ri, ci]) else float(np.nanmean(dem))

        if has_proj:
            lon, lat = transformer.transform(cell_easting, cell_northing)
        else:
            lat, lon = cell_northing, cell_easting

        cell_config = StationConfig(
            station_id=f"cluster_{cid:04d}",
            station_name=f"cluster_{cid}_n{rep['n_cells']}",
            latitude=lat,
            longitude=lon,
            altitude_m=cell_alt,
            slope_angle=float(slope[ri, ci]),
            slope_azi=float(aspect[ri, ci]),
            tz_offset=0,
        )

        cell_df = wx_forcing.copy()
        cell_df['HS'] = pd.Series(rep['hs_series'] * 100.0,
                                   index=all_timestamps)
        cell_df['HS'] = cell_df['HS'].interpolate(method='time').bfill().ffill()

        out_path = smet_dir / f"{cell_config.station_id}.smet"
        write_smet(cell_df, str(out_path), cell_config)

    sizes = [r['n_cells'] for r in representatives.values()]
    n_domain = int(np.sum(cluster_map > 0))
    print(f"\n{len(representatives)} SMET files written to {smet_dir}")
    print(f"  Coverage: {sum(sizes)} / {n_domain} domain cells")
    print(f"  Cluster sizes: min={min(sizes)}, median={np.median(sizes):.0f}, max={max(sizes)}")


# =====================================================================
# CLI
# =====================================================================

STEPS = {
    'resample': step_resample,
    'transport': step_transport,
    'features': step_features,
    'train': step_train,
    'gap_fill': step_gap_fill,
    'cluster': step_cluster,
    'smet': step_smet,
}

ALL_STEPS = ['resample', 'transport', 'features', 'train', 'gap_fill', 'cluster', 'smet']


def main():
    parser = argparse.ArgumentParser(
        description="Distributed SNOWPACK forcing generation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('step', choices=list(STEPS.keys()) + ['all', 'validate'],
                        help="Pipeline step to run")
    parser.add_argument('--project-dir', type=str, default='.',
                        help="Project root directory (default: current dir)")
    parser.add_argument('--no-model', action='store_true',
                        help="Gap-fill using smoothed observed transport (skip RF)")
    args = parser.parse_args()

    cfg = ProjectConfig(project_dir=Path(args.project_dir))

    if args.step == 'all':
        for step_name in ALL_STEPS:
            print(f"\n{'='*60}")
            print(f"  STEP: {step_name}")
            print(f"{'='*60}")
            if step_name == 'gap_fill':
                step_gap_fill(cfg, use_model=not args.no_model)
            else:
                STEPS[step_name](cfg)
    elif args.step == 'validate':
        step_train(cfg)
    elif args.step == 'gap_fill':
        step_gap_fill(cfg, use_model=not args.no_model)
    else:
        STEPS[args.step](cfg)


if __name__ == '__main__':
    main()

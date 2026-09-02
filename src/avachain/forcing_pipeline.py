#!/usr/bin/env python3
"""
Distributed SNOWPACK forcing generation pipeline.

Workflow:
  1. resample   - Resample all survey GeoTIFFs to 1m reference grid
  2. transport  - Compute transport fields for all consecutive survey pairs
  3. features   - Compute terrain features and WindNinja wind fields
  4. train      - Fit Random Forest transport model with leave-one-out CV
  5. avalanche  - Detect and correct avalanche events in transport fields
  6. cluster    - Build domain mask and cluster cells by HS evolution
  7. gap_fill   - Generate hourly HS grids between all surveys
  8. smet       - Write per-cluster SMET forcing files

Usage:
  python forcing_pipeline.py resample
  python forcing_pipeline.py transport
  python forcing_pipeline.py features
  python forcing_pipeline.py train
  python forcing_pipeline.py avalanche
  python forcing_pipeline.py cluster
  python forcing_pipeline.py gap_fill
  python forcing_pipeline.py smet
  python forcing_pipeline.py all          # run everything in order
  python forcing_pipeline.py validate     # LOO-CV report only (after features, no model saved)

Each step saves intermediate results to outputs/, so you can
restart from any step without recomputing previous ones.
"""

import sys
import argparse
import json
import pickle
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


def _mask_to_geojson_wgs84(mask: np.ndarray, transform, crs, out_path: Path) -> None:
    """Vectorise a boolean raster mask and write as a WGS84 GeoJSON file."""
    import json as _json
    from rasterio.features import shapes as _shapes
    from shapely.geometry import shape, mapping
    from shapely.ops import unary_union
    from pyproj import Transformer as _T

    mask_u8 = mask.astype(np.uint8)
    polygons = [shape(geom) for geom, val in
                _shapes(mask_u8, transform=transform) if val == 1]
    if not polygons:
        return
    merged = unary_union(polygons)

    try:
        src_epsg = crs.to_epsg() if hasattr(crs, 'to_epsg') else None
        if src_epsg and src_epsg != 4326:
            tr = _T.from_crs(f'EPSG:{src_epsg}', 'EPSG:4326', always_xy=True)
            from shapely.geometry import Polygon, MultiPolygon

            def _ring(r):
                xs, ys = zip(*r.coords)
                return list(zip(*tr.transform(list(xs), list(ys))))

            def _reproj(poly):
                if poly.geom_type == 'Polygon':
                    return Polygon(_ring(poly.exterior),
                                   [_ring(i) for i in poly.interiors])
                return MultiPolygon([_reproj(p) for p in poly.geoms])

            merged = _reproj(merged)
    except Exception as exc:
        print(f"  WARNING: GeoJSON CRS reprojection failed ({exc})")

    gj = {
        'type': 'FeatureCollection',
        'features': [{
            'type': 'Feature',
            'geometry': mapping(merged),
            'properties': {'source': 'auto_detect'},
        }],
    }
    out_path.write_text(_json.dumps(gj))


from config import ProjectConfig
from spatial_model import (
    create_reference_dem, load_and_resample,
    compute_terrain_features, compute_transport_field,
    smooth_field, compute_wind_stats,
    train_transport_model,
    gap_fill_period, gap_fill_station_only, validate_prediction,
    load_wind_library, interpolate_wind_field,
    resample_wind_to_dem, build_windninja_feature_array,
)
from smet_writer import write_smet, StationConfig
from pipeline_io import (
    discover_surveys, load_dem, load_survey_grids,
    load_transport_meta, load_weather, station_hs_at,
)
from plots import (
    plot_avalanche_results, plot_gap_fill_validation,
    plot_cluster_map, plot_cluster_variability,
)

def predict_transport_wn(model, terrain_features: dict,
                          wn_vel: np.ndarray, wn_ang: np.ndarray,
                          valid_mask: np.ndarray) -> np.ndarray:
    """Predict transport using WindNinja wind fields."""
    X, _ = build_windninja_feature_array(terrain_features, wn_vel,
                                          wn_ang, valid_mask)
    y_pred = model.predict(X)
    transport = np.full(valid_mask.shape, np.nan)
    transport.ravel()[np.where(valid_mask.ravel())[0]] = y_pred
    return transport




# =====================================================================
# Step 1: Resample
# =====================================================================

def step_resample(cfg: ProjectConfig):
    """Resample bare ground DEM and all surveys to 1m reference grid."""
    cfg.ensure_dirs()
    surveys = discover_surveys(cfg)

    ref_dem_path = cfg.resampled_dir / "dem_1m.tif"

    if not ref_dem_path.exists():
        print(f"\nCreating 1m reference DEM from {cfg.dem_path.name}...")
        create_reference_dem(str(cfg.dem_path), str(ref_dem_path),
                             cfg.target_resolution_m)
    else:
        print(f"Reference DEM exists: {ref_dem_path}")

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
    dem, transform, _ = load_dem(cfg)

    grids = load_survey_grids(cfg, surveys)
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
            transport_raw = np.full_like(dem, np.nan)
        transport_smooth = smooth_field(transport_raw, cfg.transport_smoothing_window_m)

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

    with open(str(cfg.analysis_dir / "transport_metadata.json"), 'w') as f:
        json.dump(transport_data, f, indent=2)

    print(f"\nTransport fields computed for {len(transport_data)} pairs.")


# =====================================================================
# Step 3: Compute terrain features and WindNinja wind fields
# =====================================================================

def step_features(cfg: ProjectConfig):
    """Compute terrain features and WindNinja wind fields for each survey pair."""
    cfg.ensure_dirs()
    surveys = discover_surveys(cfg)
    wx = load_weather(cfg)
    dem, dem_transform, _ = load_dem(cfg)
    transport_meta = load_transport_meta(cfg)
    grids = load_survey_grids(cfg, surveys)

    wind_lib_dir = cfg.windninja_library_dir
    if not wind_lib_dir.exists():
        raise FileNotFoundError(
            f"Wind library not found at {wind_lib_dir}. "
            f"Run generate_wind_library.sh first.")
    print("Loading WindNinja wind library...")
    wind_library = load_wind_library(str(wind_lib_dir))

    dates = sorted(grids.keys())
    fill_dem = np.where(np.isnan(dem), np.nanmean(dem), dem)

    for i in range(len(dates) - 1):
        d_a = dates[i]
        pair_id = f"{dates[i].isoformat()}__{dates[i+1].isoformat()}"
        feat_path  = cfg.analysis_dir / f"features_{d_a.isoformat()}.pkl"
        wn_vel_path = cfg.analysis_dir / f"wn_vel_{pair_id}.npy"
        wn_ang_path = cfg.analysis_dir / f"wn_ang_{pair_id}.npy"

        hs_a = grids[d_a]
        snow_surface = np.where(~np.isnan(dem) & ~np.isnan(hs_a),
                                dem + hs_a, fill_dem)

        # Terrain features on snow surface DEM
        if feat_path.exists():
            print(f"  Terrain features for {d_a}: cached")
        else:
            print(f"  Computing terrain features for {d_a}...")
            terrain = compute_terrain_features(snow_surface, cfg.target_resolution_m)
            with open(str(feat_path), 'wb') as f:
                pickle.dump(terrain, f)

        # WindNinja interpolated wind field for this period
        if wn_vel_path.exists() and wn_ang_path.exists():
            print(f"  WindNinja fields for {pair_id}: cached")
        else:
            meta = next(m for m in transport_meta if m['pair_id'] == pair_id)
            mean_spd = meta.get('wind_mean_speed') or 5.0
            mean_dir = meta.get('wind_mean_dir') or 270.0

            print(f"  WindNinja interpolation for {pair_id}: "
                  f"{mean_spd:.1f} m/s @ {mean_dir:.0f}deg...")
            vel_raw, ang_raw = interpolate_wind_field(
                wind_library, mean_spd, mean_dir)

            # Get wind transform from first library entry
            first_entry = next(iter(wind_library.values()))
            wind_transform = first_entry['transform']

            vel = resample_wind_to_dem(vel_raw, wind_transform,
                                        dem.shape, dem_transform)
            ang = resample_wind_to_dem(ang_raw, wind_transform,
                                        dem.shape, dem_transform)
            np.save(str(wn_vel_path), vel)
            np.save(str(wn_ang_path), ang)

    print("\nFeature computation complete.")


# =====================================================================
# Step 4: Train model with LOO-CV
# =====================================================================

def _load_period_data(cfg: ProjectConfig,
                      transport_meta: list,
                      dem: np.ndarray) -> tuple:
    """
    Load features, transport fields, and wind stats for all usable periods.

    Returns (period_data, skipped) where period_data is a list of dicts
    and skipped is a list of human-readable skip reasons.
    """
    period_data = []
    skipped = []

    for meta in transport_meta:
        pair_id = meta['pair_id']
        d_a = meta['date_a']

        if not meta.get('has_weather', True):
            skipped.append(f"{pair_id} (no weather data)")
            continue

        wind_speed = meta.get('wind_mean_speed')
        wind_dir = meta.get('wind_mean_dir')
        if wind_speed is None or wind_dir is None:
            skipped.append(f"{pair_id} (missing wind stats)")
            continue

        transport_path = cfg.analysis_dir / f"transport_smooth_{pair_id}.npy"
        if not transport_path.exists():
            skipped.append(f"{pair_id} (transport file missing)")
            continue
        transport = np.load(str(transport_path))

        feat_path = cfg.analysis_dir / f"features_{d_a}.pkl"
        if not feat_path.exists():
            skipped.append(f"{pair_id} (feature file missing)")
            continue
        with open(str(feat_path), 'rb') as f:
            terrain = pickle.load(f)

        wn_vel_path = cfg.analysis_dir / f"wn_vel_{pair_id}.npy"
        wn_ang_path = cfg.analysis_dir / f"wn_ang_{pair_id}.npy"
        if not wn_vel_path.exists() or not wn_ang_path.exists():
            skipped.append(f"{pair_id} (WindNinja fields missing)")
            continue
        wn_vel = np.load(str(wn_vel_path))
        wn_ang = np.load(str(wn_ang_path))

        valid = (~np.isnan(dem) & ~np.isnan(transport) &
                 ~np.isnan(wn_vel) & ~np.isnan(terrain.get('slope', dem)))
        n_valid = int(valid.sum())

        if n_valid < 100:
            skipped.append(f"{pair_id} (only {n_valid} valid cells)")
            continue

        period_data.append({
            'pair_id': pair_id,
            'meta': meta,
            'transport': transport,
            'terrain': terrain,
            'wn_vel': wn_vel,
            'wn_ang': wn_ang,
            'valid': valid,
        })

    return period_data, skipped


def _run_loo_cv(period_data: list) -> list:
    """Run leave-one-out cross-validation. Returns list of per-period metric dicts."""
    cv_results = []
    print("\nLeave-one-out cross-validation:")

    for hold_idx in range(len(period_data)):
        hold = period_data[hold_idx]
        train_periods = [p for i, p in enumerate(period_data) if i != hold_idx]

        X_parts, y_parts = [], []
        for p in train_periods:
            if p['valid'].sum() == 0:
                continue
            X, names = build_windninja_feature_array(p['terrain'], p['wn_vel'],
                                                        p['wn_ang'], p['valid'])
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
        model = train_transport_model(X_train, y_train)

        if hold['valid'].sum() == 0:
            print(f"  {hold['pair_id']}: SKIPPED (no valid cells in hold-out)")
            cv_results.append({'pair_id': hold['pair_id'],
                               'rmse': np.nan, 'bias': np.nan,
                               'mae': np.nan, 'r': np.nan, 'n': 0})
            continue

        pred_transport = predict_transport_wn(model, hold['terrain'], hold['wn_vel'],
                                               hold['wn_ang'], hold['valid'])
        metrics = validate_prediction(pred_transport, hold['transport'], hold['valid'])

        cv_results.append({
            'pair_id': hold['pair_id'],
            **{k: float(v) if isinstance(v, (np.floating, float)) else
               int(v) if isinstance(v, (np.integer, int)) else v
               for k, v in metrics.items()},
        })

        print(f"  {hold['pair_id']}: RMSE={metrics['rmse']*100:.1f}cm, "
              f"r={metrics['r']:.3f}, bias={metrics['bias']*100:.1f}cm")

    rmses = [r['rmse'] for r in cv_results if not np.isnan(r['rmse'])]
    rs = [r['r'] for r in cv_results if not np.isnan(r['r'])]
    print(f"\nLOO-CV Summary ({len(rmses)} periods):")
    print(f"  RMSE: mean={np.mean(rmses)*100:.1f}cm, "
          f"median={np.median(rmses)*100:.1f}cm")
    print(f"  r:    mean={np.mean(rs):.3f}, median={np.median(rs):.3f}")

    return cv_results


def step_train(cfg: ProjectConfig):
    """Train Random Forest on all pairs, with leave-one-out CV."""
    cfg.ensure_dirs()
    dem, _, _ = load_dem(cfg)
    transport_meta = load_transport_meta(cfg)

    print("Loading feature data for all periods...")
    period_data, skipped = _load_period_data(cfg, transport_meta, dem)

    if skipped:
        print(f"Skipped {len(skipped)} periods:")
        for s in skipped:
            print(f"    {s}")
    print(f"Using {len(period_data)} periods for training")

    cv_results = _run_loo_cv(period_data)

    with open(str(cfg.analysis_dir / "cv_results.json"), 'w') as f:
        json.dump(cv_results, f, indent=2)

    # Train final model on all data
    print("\nTraining final model on all data...")
    X_all, y_all, names = [], [], None
    for p in period_data:
        X, names = build_windninja_feature_array(p['terrain'], p['wn_vel'],
                                                    p['wn_ang'], p['valid'])
        y = p['transport'].ravel()[np.where(p['valid'].ravel())[0]]
        X_all.append(X)
        y_all.append(y)

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)
    print(f"  Training set: {X_all.shape[0]} samples, {X_all.shape[1]} features")

    final_model = train_transport_model(X_all, y_all)

    print(f"\n  Feature importance:")
    for name, imp in sorted(zip(names, final_model.feature_importances_),
                             key=lambda x: -x[1]):
        print(f"    {name:25s}: {imp:.3f}")

    model_path = cfg.analysis_dir / "transport_model.pkl"
    with open(str(model_path), 'wb') as f:
        pickle.dump({'model': final_model, 'feature_names': names}, f)
    print(f"\nModel saved to {model_path}")


def step_validate(cfg: ProjectConfig):
    """
    Run LOO-CV and print results without training or saving the final model.

    Requires the 'features' step to have been run. Useful for evaluating
    transport model skill without overwriting an existing trained model.
    """
    cfg.ensure_dirs()
    dem, _, _ = load_dem(cfg)
    transport_meta = load_transport_meta(cfg)

    print("Loading feature data for all periods...")
    period_data, skipped = _load_period_data(cfg, transport_meta, dem)

    if skipped:
        print(f"Skipped {len(skipped)} periods:")
        for s in skipped:
            print(f"    {s}")
    print(f"Using {len(period_data)} periods for validation")

    cv_results = _run_loo_cv(period_data)

    with open(str(cfg.analysis_dir / "cv_results.json"), 'w') as f:
        json.dump(cv_results, f, indent=2)
    print(f"\nCV results saved to {cfg.analysis_dir / 'cv_results.json'}")
    print("No model was trained or saved.")


# =====================================================================
# Step 5: Avalanche detection and correction
# =====================================================================

def step_avalanche(cfg: ProjectConfig):
    """
    Classify inter-survey periods and correct transport fields.

    Classification (per start-zone frac_loss + crown detection):
      frac_loss = fraction of start-zone cells with ΔHS < -min_scour_depth_m

      if frac_loss < frac_loss_threshold (0.60) OR crown detected:
          → localized mass removal (avalanche or wind_scour)
          → label: "avalanche" if manual known_events or CAIC API obs match,
                   else "wind_scour"
      else:
          → settlement_melt — no reinit, no transport correction

    For avalanche (crown detected):
        - Correct transport field, save avalanche_dhs, export crown GeoJSON
        - reinit_needed = True, transport_corrected = True

    For wind_scour (no crown, but localized loss):
        - Export loss-cell GeoJSON (for cluster reinit target)
        - reinit_needed = True, transport_corrected = False
        - Transport field is NOT corrected (wind redistribution is real signal)

    Auto-detected events are appended to avalanche_events.json so they
    become known on subsequent runs.
    """
    cfg.ensure_dirs()

    from avalanche import (
        delineate_crown, load_known_events,
        separate_avalanche_from_wind
    )
    from snowpack_io import kml_to_mask
    from datetime import datetime as _dt, timedelta as _td

    dem, transform, crs = load_dem(cfg)
    transport_meta = load_transport_meta(cfg)
    valid_dem = ~np.isnan(dem)

    # Start-zone mask for frac_loss
    start_zone_mask = kml_to_mask(cfg.start_zone_kml, dem.shape, transform)

    # Manual known events (manual label takes precedence over auto-detect)
    known_events = (load_known_events(str(cfg.avalanche_events_path))
                    if cfg.avalanche_events_path.exists() else [])
    known_periods = {ev.get('period') for ev in known_events}
    if known_events:
        print(f"Loaded {len(known_events)} known event(s) from "
              f"{cfg.avalanche_events_path.name}")
    else:
        print("No known events file — all events auto-detected")

    # One-shot CAIC API fetch for the full season date range
    all_caic_obs = []
    try:
        from fetch_avalanche_obs import (
            fetch_observations, extract_observation, filter_to_boundary
        )
        all_dates = sorted(
            {m['date_a'] for m in transport_meta} |
            {m['date_b'] for m in transport_meta}
        )
        season_start = (_dt.strptime(all_dates[0],  '%Y-%m-%d')
                        - _td(days=cfg.caic_obs_window_days))
        season_end   = (_dt.strptime(all_dates[-1], '%Y-%m-%d')
                        + _td(days=cfg.caic_obs_window_days))
        raw_obs = fetch_observations(season_start, season_end)
        all_caic_obs = [extract_observation(r) for r in raw_obs]
        if cfg.boundary_kml.exists():
            all_caic_obs = filter_to_boundary(
                all_caic_obs, str(cfg.boundary_kml),
                buffer_m=cfg.caic_spatial_buffer_m)
        print(f"CAIC API: {len(all_caic_obs)} observations in study area")
    except Exception as exc:
        print(f"  CAIC API unavailable ({exc}) — auto events labeled 'wind_scour'")

    print(f"\nScanning {len(transport_meta)} periods...")
    period_results = {}
    corrected_periods = {}
    new_auto_events = []

    for meta in transport_meta:
        if not meta.get('has_weather', True):
            continue
        pair_id = meta['pair_id']
        d_a, d_b = meta['date_a'], meta['date_b']
        stn_dhs  = meta.get('stn_dhs') or 0

        hs_a_path = cfg.resampled_dir / f"hs_{d_a}.npy"
        hs_b_path = cfg.resampled_dir / f"hs_{d_b}.npy"
        if not hs_a_path.exists() or not hs_b_path.exists():
            continue

        hs_a = np.clip(np.load(str(hs_a_path)), 0, None)
        hs_b = np.clip(np.load(str(hs_b_path)), 0, None)
        dhs  = hs_b - hs_a

        # --- frac_loss ---
        valid_sz = start_zone_mask & ~np.isnan(hs_a) & ~np.isnan(hs_b)
        n_valid  = int(valid_sz.sum())
        frac_loss = (float(((dhs < -cfg.min_scour_depth_m) & valid_sz).sum()) / n_valid
                     if n_valid > 0 else 0.0)

        # --- Crown detection (always, regardless of frac_loss) ---
        regions   = delineate_crown(hs_a, hs_b, dem, stn_dhs)
        has_crown = bool(regions)

        # --- Classification ---
        is_known      = pair_id in known_periods
        crown_override = has_crown and frac_loss >= cfg.frac_loss_threshold
        is_candidate  = frac_loss < cfg.frac_loss_threshold or has_crown

        period_results[pair_id] = {
            'regions':   regions,
            'is_known':  is_known,
            'stn_dhs':   stn_dhs,
            'frac_loss': round(frac_loss, 3),
            'has_crown': has_crown,
        }

        if not is_candidate:
            print(f"  {pair_id}: settlement_melt  "
                  f"frac_loss={frac_loss:.2f}")
            continue

        # --- Determine label via CAIC lookup ---
        if is_known:
            label = 'avalanche'
            period_caic = []
        else:
            t_lo = _dt.strptime(d_a, '%Y-%m-%d') - _td(days=cfg.caic_obs_window_days)
            t_hi = _dt.strptime(d_b, '%Y-%m-%d') + _td(days=cfg.caic_obs_window_days)
            period_caic = [
                o for o in all_caic_obs
                if o.get('date_parsed') and
                t_lo <= o['date_parsed'].replace(tzinfo=None) <= t_hi
            ]
            label = 'avalanche' if period_caic else 'wind_scour'

        override_note = " *** CROWN OVERRIDE (widespread) ***" if crown_override else ""
        known_note    = " *** KNOWN ***" if is_known else ""

        if has_crown:
            r = regions[0]
            print(f"  {pair_id}: {label}  frac_loss={frac_loss:.2f}  "
                  f"{len(regions)} region(s)  largest={r['n_cells']} cells "
                  f"(crown={r['n_crown_cells']}, flank={r['n_flank_cells']})"
                  f"{known_note}{override_note}")
        else:
            print(f"  {pair_id}: {label}  frac_loss={frac_loss:.2f}  "
                  f"no crown (wind_scour inferred){known_note}")

        # --- Mid-period timestamp (fallback for auto events) ---
        if is_known:
            known_ev = next(ev for ev in known_events if ev.get('period') == pair_id)
            event_ts = known_ev.get('timestamp', None)
        else:
            mid = (_dt.strptime(d_a, '%Y-%m-%d') +
                   (_dt.strptime(d_b, '%Y-%m-%d') -
                    _dt.strptime(d_a, '%Y-%m-%d')) / 2)
            event_ts = mid.strftime('%Y-%m-%dT12:00:00Z')

        event_date_str = (pd.Timestamp(event_ts).strftime('%Y%m%d')
                         if event_ts else d_b.replace('-', ''))

        # === AVALANCHE path (crown detected) ===
        if has_crown:
            combined_mask = np.zeros(dem.shape, dtype=bool)
            for r in regions:
                combined_mask |= r['mask']

            transport_path = cfg.analysis_dir / f"transport_smooth_{pair_id}.npy"
            if transport_path.exists():
                transport = np.load(str(transport_path))
                avy_dhs, corrected_transport = separate_avalanche_from_wind(
                    transport, combined_mask, valid_dem)
                np.save(str(cfg.analysis_dir / f"transport_corrected_{pair_id}.npy"),
                        corrected_transport)
                np.save(str(cfg.analysis_dir / f"avalanche_dhs_{pair_id}.npy"), avy_dhs)
                print(f"    → Transport corrected, {combined_mask.sum()} cells")

            corrected_periods[pair_id] = {
                'n_regions':           len(regions),
                'total_cells':         int(combined_mask.sum()),
                'total_volume_m3':     float(np.nansum(avy_dhs[combined_mask]))
                                       if transport_path.exists() else 0.0,
                'corrected':           True,
                'transport_corrected': True,
                'reinit_needed':       True,
                'event_timestamp':     event_ts,
                'label':               label,
                'frac_loss':           round(frac_loss, 3),
            }
            event_mask = combined_mask

        # === WIND SCOUR path (no crown, localized loss) ===
        else:
            event_mask = (dhs < -cfg.min_scour_depth_m) & valid_sz
            corrected_periods[pair_id] = {
                'n_regions':           0,
                'total_cells':         int(event_mask.sum()),
                'total_volume_m3':     float(np.nansum(dhs[event_mask])),
                'corrected':           False,
                'transport_corrected': False,
                'reinit_needed':       True,
                'event_timestamp':     event_ts,
                'label':               'wind_scour',
                'frac_loss':           round(frac_loss, 3),
            }
            print(f"    → Loss mask: {event_mask.sum()} cells  [no transport correction]")

        # --- Export event GeoJSON ---
        gj_out = cfg.boundaries_dir / f"avalanche_release_area_{event_date_str}.geojson"
        if not gj_out.exists():
            _mask_to_geojson_wgs84(event_mask, transform, crs, gj_out)
            print(f"    → Event mask exported: {gj_out.name}")

        # --- Auto-append to avalanche_events.json ---
        if not is_known:
            new_auto_events.append({
                'period':    pair_id,
                'timestamp': event_ts,
                'label':     label,
                'size':      'unknown',
                'trigger':   'natural' if label == 'avalanche' else 'auto_detect',
                'source':    'auto_detect' + (' + CAIC API' if period_caic else ''),
                'frac_loss': round(frac_loss, 3),
                'n_regions': len(regions),
                'notes':     (f"Auto-detected. frac_loss={frac_loss:.2f}"
                              + override_note),
            })

    # --- Write outputs/analysis/avalanche_events.json ---
    avy_output = {}
    for pair_id, pr in period_results.items():
        if pair_id in corrected_periods:
            avy_output[pair_id] = corrected_periods[pair_id]
        else:
            avy_output[pair_id] = {
                'n_regions':   0,
                'corrected':   False,
                'reinit_needed': False,
                'frac_loss':   pr['frac_loss'],
                'label':       'settlement_melt',
            }

    with open(str(cfg.analysis_dir / "avalanche_events.json"), 'w') as f:
        json.dump(avy_output, f, indent=2)

    # --- Append new auto-events to data/boundaries/avalanche_events.json ---
    if new_auto_events:
        all_events = list(known_events) + new_auto_events
        with open(str(cfg.avalanche_events_path), 'w') as f:
            json.dump(all_events, f, indent=2)
        print(f"\n{len(new_auto_events)} new event(s) appended to "
              f"{cfg.avalanche_events_path.name}")

    n_corrected = sum(1 for v in avy_output.values() if v.get('corrected'))
    n_reinit    = sum(1 for v in avy_output.values() if v.get('reinit_needed'))
    print(f"\nAvalanche step complete: {n_corrected} transport correction(s), "
          f"{n_reinit} reinit candidate(s)")

    plot_avalanche_results(period_results, corrected_periods, dem, transform, cfg)


# =====================================================================
# Step 6: Build domain mask and cluster cells
# =====================================================================

def step_cluster(cfg: ProjectConfig):
    """Build domain mask from KML boundary, cluster cells by HS evolution."""
    cfg.ensure_dirs()

    from clustering import (
        build_domain_mask, build_survey_hs_matrix, cluster_cells,
        auto_select_n_clusters
    )

    dem, transform, crs = load_dem(cfg)

    print("Building domain mask...")
    domain_mask = build_domain_mask(
        dem, transform, crs,
        kml_path=str(cfg.boundary_kml),
        min_slope_deg=cfg.min_slope_deg,
        resolution=cfg.target_resolution_m,
    )

    mask_path = cfg.analysis_dir / "domain_mask.npy"
    np.save(str(mask_path), domain_mask)
    print(f"  Mask saved: {domain_mask.sum()} cells")

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
                                n_clusters=n_clusters,
                                max_cells_per_cluster=cfg.max_cells_per_cluster,
                                max_cluster_std_m=cfg.max_cluster_std_m,
                                n_pca_components=cfg.n_pca_components,
                                min_cluster_size=cfg.min_cluster_size)

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

    plot_cluster_map(cluster_map, dem, transform, cids, sizes,
                     hs_matrix, cell_idx, survey_dates, cfg)
    plot_cluster_variability(cluster_map, dem, transform, survey_grids, cfg, cids)


# =====================================================================
# Step 7: Gap-fill
# =====================================================================

def step_gap_fill(cfg: ProjectConfig, use_model: bool = False,
                  station_only: bool = False):
    """Generate hourly HS grids between all surveys."""
    cfg.ensure_dirs()

    dem, _, _ = load_dem(cfg)
    transport_meta = load_transport_meta(cfg)
    wx = load_weather(cfg)

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
    all_metrics = []

    for meta in transport_meta:
        pair_id = meta['pair_id']
        d_a = datetime.date.fromisoformat(meta['date_a'])
        d_b = datetime.date.fromisoformat(meta['date_b'])

        if not meta.get('has_weather', True):
            print(f"\nSkipping {pair_id}: no weather data coverage")
            continue

        print(f"\nGap-filling {pair_id}...")

        hs_a = np.load(str(cfg.resampled_dir / f"hs_{d_a.isoformat()}.npy"))
        hs_b = np.load(str(cfg.resampled_dir / f"hs_{d_b.isoformat()}.npy"))
        valid = ~np.isnan(dem) & ~np.isnan(hs_a) & ~np.isnan(hs_b)

        if use_model:
            feat_path   = cfg.analysis_dir / f"features_{d_a.isoformat()}.pkl"
            wn_vel_path = cfg.analysis_dir / f"wn_vel_{pair_id}.npy"
            wn_ang_path = cfg.analysis_dir / f"wn_ang_{pair_id}.npy"
            with open(str(feat_path), 'rb') as f:
                terrain = pickle.load(f)
            wn_vel = np.load(str(wn_vel_path))
            wn_ang = np.load(str(wn_ang_path))
            transport = predict_transport_wn(model, terrain, wn_vel, wn_ang, valid)
        else:
            corrected_path = cfg.analysis_dir / f"transport_corrected_{pair_id}.npy"
            smooth_path = cfg.analysis_dir / f"transport_smooth_{pair_id}.npy"
            if corrected_path.exists():
                transport = np.load(str(corrected_path))
                print(f"  Using avalanche-corrected transport")
            else:
                transport = np.load(str(smooth_path))

        t0 = pd.Timestamp(datetime.datetime.combine(d_a, datetime.time(cfg.flight_hour_utc)))
        t1 = pd.Timestamp(datetime.datetime.combine(d_b, datetime.time(cfg.flight_hour_utc)))
        stn_hs = wx.loc[t0:t1, 'HS'] / 100.0  # cm → m
        wind_speed = wx.loc[t0:t1, 'VW']

        if len(stn_hs) < 2:
            print(f"  Insufficient station data ({len(stn_hs)} hours), skipping")
            continue

        if station_only:
            grids = gap_fill_station_only(hs_a, stn_hs, valid)
        else:
            grids = gap_fill_period(hs_a, transport, stn_hs, wind_speed, valid)

        avy_path = cfg.analysis_dir / f"avalanche_dhs_{pair_id}.npy"
        if avy_path.exists():
            from avalanche import apply_avalanche_event
            avy_dhs = np.load(str(avy_path))
            event_ts = t0 + (t1 - t0) / 2
            avy_meta_path = cfg.analysis_dir / "avalanche_events.json"
            if avy_meta_path.exists():
                with open(str(avy_meta_path)) as f:
                    avy_meta = json.load(f)
                if pair_id in avy_meta and 'event_timestamp' in avy_meta[pair_id]:
                    loaded_ts = pd.Timestamp(avy_meta[pair_id]['event_timestamp'])
                    event_ts = loaded_ts.tz_localize(None) if loaded_ts.tzinfo else loaded_ts
            grids = apply_avalanche_event(grids, avy_dhs, event_ts, valid)
            print(f"  Applied avalanche event at {event_ts}")

        pred_end = grids[t1]
        metrics = validate_prediction(pred_end, hs_b, valid)
        metrics.update({
            'pair_id': pair_id,
            'date_a': d_a.isoformat(),
            'date_b': d_b.isoformat(),
            'stn_dhs': float(meta.get('stn_dhs', 0) or 0),
            'n_hours': len(grids),
            'pred': pred_end,
            'obs': hs_b,
            'valid': valid,
        })
        all_metrics.append(metrics)

        print(f"  {len(grids)} hours, endpoint RMSE={metrics['rmse']*100:.1f}cm, "
              f"r={metrics['r']:.3f}")

        out_path = cfg.grids_dir / f"hourly_{pair_id}.npz"
        timestamps = list(grids.keys())
        grid_stack = np.stack([grids[ts] for ts in timestamps])
        np.savez_compressed(str(out_path),
                            grids=grid_stack,
                            timestamps=np.array([str(ts) for ts in timestamps]))
        total_grids += len(grids)
        print(f"  Saved to {out_path}")

    print(f"\nGap-fill complete. {total_grids} total hourly grids.")

    if all_metrics:
        plot_gap_fill_validation(all_metrics, cfg)


# =====================================================================
# Step 8: Write SMET files
# =====================================================================

def step_smet(cfg: ProjectConfig):
    """Generate per-cluster SMET forcing files from hourly grids + saved clusters."""
    cfg.ensure_dirs()

    from clustering import compute_cluster_representatives

    dem, transform, crs = load_dem(cfg)
    wx = load_weather(cfg)

    fill_dem = np.where(np.isnan(dem), np.nanmean(dem), dem)
    dy, dx = np.gradient(fill_dem, cfg.target_resolution_m)
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    aspect = np.degrees(np.arctan2(-dx, dy)) % 360

    cluster_path = cfg.analysis_dir / "cluster_map.npy"
    if not cluster_path.exists():
        print("ERROR: No cluster map found. Run 'cluster' step first.")
        return
    cluster_map = np.load(str(cluster_path))
    cids = np.unique(cluster_map[cluster_map > 0])
    print(f"Loaded cluster map: {len(cids)} clusters")

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

    print("Computing cluster representative HS series...")
    representatives = compute_cluster_representatives(
        cluster_map, grid_stack, all_timestamps)
    del grid_stack

    wx_forcing = wx.reindex(pd.DatetimeIndex(all_timestamps),
                            method='nearest',
                            tolerance=pd.Timedelta(hours=1))

    try:
        from pyproj import Transformer
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        has_proj = True
    except ImportError:
        print("  WARNING: pyproj not installed, using UTM coords")
        has_proj = False

    smet_dir = cfg.smet_dir
    smet_dir.mkdir(parents=True, exist_ok=True)

    for cid, rep in representatives.items():
        cr, cc = rep['centroid_row'], rep['centroid_col']
        ri = max(0, min(int(round(cr)), dem.shape[0] - 1))
        ci = max(0, min(int(round(cc)), dem.shape[1] - 1))

        cell_easting = transform[2] + cc * transform[0]
        cell_northing = transform[5] + cr * transform[4]
        cell_alt = (float(dem[ri, ci]) if not np.isnan(dem[ri, ci])
                    else float(np.nanmean(dem)))

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
        cell_df['HS'] = pd.Series(rep['hs_series'] * 100.0, index=all_timestamps)
        cell_df['HS'] = cell_df['HS'].interpolate(method='time').bfill().ffill()

        out_path = smet_dir / f"{cell_config.station_id}.smet"
        write_smet(cell_df, str(out_path), cell_config)

    sizes = [r['n_cells'] for r in representatives.values()]
    n_domain = int(np.sum(cluster_map > 0))
    print(f"\n{len(representatives)} SMET files written to {smet_dir}")
    print(f"  Coverage: {sum(sizes)} / {n_domain} domain cells")
    print(f"  Cluster sizes: min={min(sizes)}, median={np.median(sizes):.0f}, "
          f"max={max(sizes)}")

    # --- Augment with NWP refill ---
    refill_path = cfg.project_dir / "data" / "weather" / "refill.smet"
    if refill_path.exists():
        from smet_writer import read_smet, augment_smet_with_refill, fill_smet_gaps_from_refill
        print(f"\nAugmenting SMET files from {refill_path.name}...")
        refill_header, refill_df = read_smet(str(refill_path))
        added_fields = None
        n_augmented = 0
        for cid in representatives:
            smet_file = smet_dir / f"cluster_{cid:04d}.smet"
            if smet_file.exists():
                added = augment_smet_with_refill(
                    str(smet_file), refill_header, refill_df)
                if added_fields is None and added:
                    added_fields = added
                n_augmented += 1
        if added_fields:
            print(f"  {n_augmented} files augmented, "
                  f"{len(added_fields)} fields added: {', '.join(added_fields)}")
        else:
            print("  No new fields to add (all refill fields already present).")

        # --- Fill temporal gaps from refill (e.g. Dec 18 station outage) ---
        print(f"\nFilling SMET gaps from {refill_path.name} (max 48h)...")
        total_inserted = total_filled = 0
        for cid in representatives:
            smet_file = smet_dir / f"cluster_{cid:04d}.smet"
            if smet_file.exists():
                result = fill_smet_gaps_from_refill(
                    str(smet_file), refill_header, refill_df,
                    max_gap_hours=48)
                total_inserted += result['n_rows_inserted']
                total_filled += result['n_rows_filled']
        print(f"  Rows inserted: {total_inserted}, "
              f"cells filled: {total_filled}")
    else:
        print(f"\nNo NWP refill SMET at {refill_path}, skipping augmentation.")


# =====================================================================
# Step: reinit — Post-avalanche SNOWPACK reinitialization
# =====================================================================

def _reinit_single(cfg, args, event_date, event_time, date_before,
                    date_after, snapshot_date, release_geojson):
    """Run reinit for one event. Shared by single-event and multi-event paths."""
    from reinitialize_snowpack import run_reinit
    run_reinit(
        cfg=cfg,
        date_before=date_before,
        date_after=date_after,
        event_date=event_date,
        event_time=event_time,
        snapshot_date=snapshot_date,
        release_geojson=release_geojson,
        kernel_size=getattr(args, 'kernel_size_reinit', 7),
        threshold_sigma=getattr(args, 'threshold_sigma_reinit', 1.2),
        dry_run=getattr(args, 'reinit_dry_run', False),
        no_backup=getattr(args, 'reinit_no_backup', False),
    )


def step_reinit(cfg: ProjectConfig):
    """
    Scour release cluster .sno files after detected avalanche event(s).

    Single-event mode (default):
      Uses --event-date, --date-before, --date-after from CLI.

    Multi-event mode (--all-events):
      Reads outputs/analysis/avalanche_events.json and reruns all
      corrected events in chronological order.  For each event,
      uses the corresponding avalanche_release_area_{YYYYMMDD}.geojson
      if it exists (written by step_avalanche), else falls back to
      auto-detection.
    """
    args = cfg._reinit_args

    if getattr(args, 'all_events', False):
        avy_path = cfg.analysis_dir / "avalanche_events.json"
        if not avy_path.exists():
            print("ERROR: avalanche_events.json not found. Run 'avalanche' step first.")
            return

        with open(avy_path) as f:
            avy_events = json.load(f)

        corrected = sorted(
            [(pid, ev) for pid, ev in avy_events.items()
             if ev.get('reinit_needed', ev.get('corrected', False))],
            key=lambda x: x[1].get('event_timestamp', ''),
        )
        if not corrected:
            print("No reinit_needed events found in avalanche_events.json.")
            return

        print(f"Multi-event reinit: {len(corrected)} event(s)")
        for pair_id, ev in corrected:
            d_a, d_b = pair_id.split('__')
            event_ts   = ev.get('event_timestamp', '')
            if event_ts:
                ts = pd.Timestamp(event_ts)
                event_date = ts.strftime('%Y-%m-%d')
                event_time = ts.strftime('%H:%M:%S')
            else:
                event_date = d_b
                event_time = '12:00:00'

            event_date_str = event_date.replace('-', '')
            gj_path = (cfg.boundaries_dir /
                       f"avalanche_release_area_{event_date_str}.geojson")
            release_geojson = str(gj_path) if gj_path.exists() else None

            print(f"\n  Event {pair_id}  ({event_date} {event_time})"
                  + (f"  GeoJSON: {gj_path.name}" if release_geojson else
                     "  (auto-detect)"))
            _reinit_single(
                cfg, args,
                event_date=event_date,
                event_time=event_time,
                date_before=d_a,
                date_after=d_b,
                snapshot_date=d_a,
                release_geojson=release_geojson,
            )
    else:
        _reinit_single(
            cfg, args,
            event_date=args.event_date,
            event_time=getattr(args, 'event_time', '12:00:00'),
            date_before=args.date_before,
            date_after=args.date_after,
            snapshot_date=args.snapshot_date,
            release_geojson=getattr(args, 'release_geojson', None),
        )


# =====================================================================
# CLI
# =====================================================================

STEPS = {
    'resample':  step_resample,
    'transport': step_transport,
    'features':  step_features,
    'train':     step_train,
    'avalanche': step_avalanche,
    'cluster':   step_cluster,
    'gap_fill':  step_gap_fill,
    'smet':      step_smet,
    'reinit':    step_reinit,
}

# Canonical execution order.
# Note: 'cluster' only requires 'resample' and can run before 'gap_fill',
# but is placed here so that the domain mask is available during QA review
# of gap-fill results before writing SMET files.
# 'reinit' is not in ALL_STEPS — it's event-driven, not part of routine runs.
ALL_STEPS = ['resample', 'transport', 'features', 'train', 'avalanche',
             'cluster', 'gap_fill', 'smet']


def main():
    parser = argparse.ArgumentParser(
        description="Distributed SNOWPACK forcing generation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('step',
                        choices=list(STEPS.keys()) + ['all', 'validate'],
                        help="Pipeline step to run")
    parser.add_argument('--project-dir', type=str, default='.',
                        help="Project root directory (default: current dir)")
    parser.add_argument('--use-model', action='store_true',
                        help="Gap-fill using RF model for transport (default: observed transport)")
    parser.add_argument('--station-only', action='store_true',
                        help="Gap-fill using station dHS only — no spatial transport")

    # reinit step arguments
    parser.add_argument('--event-date', default='2026-01-18',
                        help="Avalanche event date YYYY-MM-DD (reinit step)")
    parser.add_argument('--event-time', default='12:00:00',
                        help="Event time UTC HH:MM:SS (reinit step)")
    parser.add_argument('--date-before', default='2026-01-14',
                        help="Pre-event survey date (reinit step)")
    parser.add_argument('--date-after', default='2026-01-20',
                        help="Post-event survey date (reinit step)")
    parser.add_argument('--snapshot-date', default='2026-01-18',
                        help="SNOWPACK snapshot for slab thickness (reinit step)")
    parser.add_argument('--release-geojson', default=None,
                        help="Pre-drawn release GeoJSON (reinit step, "
                             "bypasses auto-detection)")
    parser.add_argument('--kernel-size-reinit', type=int, default=7,
                        help="Min-kernel filter size (reinit step)")
    parser.add_argument('--threshold-sigma-reinit', type=float, default=1.2,
                        help="Min-kernel threshold sigma (reinit step)")
    parser.add_argument('--reinit-dry-run', action='store_true',
                        help="Show what reinit would do without writing")
    parser.add_argument('--reinit-no-backup', action='store_true',
                        help="Skip .sno.bak backup files (reinit step)")
    parser.add_argument('--all-events', action='store_true',
                        help="Reinit all corrected events from avalanche_events.json "
                             "(ignores --event-date/--date-before/--date-after)")

    args = parser.parse_args()

    cfg = ProjectConfig(project_dir=Path(args.project_dir))

    if args.step == 'all':
        for step_name in ALL_STEPS:
            print(f"\n{'='*60}")
            print(f"  STEP: {step_name}")
            print(f"{'='*60}")
            if step_name == 'gap_fill':
                step_gap_fill(cfg, use_model=args.use_model,
                              station_only=args.station_only)
            else:
                STEPS[step_name](cfg)
    elif args.step == 'validate':
        step_validate(cfg)
    elif args.step == 'gap_fill':
        step_gap_fill(cfg, use_model=args.use_model,
                      station_only=args.station_only)
    elif args.step == 'reinit':
        # Stash args on cfg for step_reinit to access
        cfg._reinit_args = args
        step_reinit(cfg)
    else:
        STEPS[args.step](cfg)


if __name__ == '__main__':
    main()
    
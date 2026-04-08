"""
Avalanche event detection and handling for gap-fill pipeline.

Problem:
  An avalanche redistributes snow mass instantaneously, but the gap-fill
  smears it across the entire inter-survey period proportional to wind speed².
  This produces physically wrong hourly HS sequences — gradual erosion where
  there should be an abrupt slab release, and gradual deposition where a
  debris pile appeared in minutes.

Approach:
  1. DETECT: Identify which survey periods contain avalanche events
  2. DELINEATE: Map the avalanche footprint (crown, track, deposit)
  3. SEPARATE: Remove avalanche ΔHS from the transport field
  4. APPLY: Insert the avalanche as an instantaneous event at estimated time
  5. GAP-FILL: Use the corrected (wind-only) transport for the rest

Detection strategy:
  Simple thresholding on ΔHS doesn't work because rocks, trees, and survey
  noise produce the same signature as avalanche crowns. Better approaches:

  a) Anomaly detection: Compare each period's ΔHS field to the "expected"
     pattern from smoothed Sx-based transport. Cells that deviate strongly
     AND form a spatially coherent downslope feature are avalanche candidates.

  b) Flow-path connectivity: A real avalanche follows gravitational flow
     lines. Crown → track → deposit should be connected downslope.
     Wind transport has no such directional coherence.

  c) Multi-period consistency: If a cell's ΔHS is anomalous in one period
     but normal in all others, it's more likely an event than a persistent
     terrain feature. Rocks/trees show up as anomalies in every period.

  d) Known event integration: When avalanche occurrence data is available
     (obs records, forecaster notes, control records), use it to identify
     which periods contain events and approximately when they occurred.

Known events for this dataset:
  - 2026-01-18: Avalanche on Little Professor slope (within Jan14→Jan20 period)
"""

import numpy as np
from scipy.ndimage import label as connected_components
from typing import Optional, Tuple
import warnings


def detect_avalanche_candidates(dhs: np.ndarray,
                                 dem: np.ndarray,
                                 slope: np.ndarray,
                                 aspect: np.ndarray,
                                 stn_dhs: float,
                                 persistent_anomaly_mask: np.ndarray = None,
                                 min_crown_area_m2: float = 100.0,
                                 crown_dhs_threshold: float = -0.5,
                                 resolution: float = 1.0) -> dict:
    """
    Detect potential avalanche events in a survey-to-survey ΔHS field.

    Looks for spatially coherent zones of anomalous HS loss on steep terrain
    that are NOT persistent across all periods (i.e., not rocks/trees).

    Parameters
    ----------
    dhs : 2D array of HS change (meters), survey_after - survey_before
    dem : 2D elevation array
    slope : 2D slope angle (degrees)
    aspect : 2D aspect (degrees)
    stn_dhs : station ΔHS for this period (meters)
    persistent_anomaly_mask : boolean mask of cells that are anomalous
        in ALL periods (rocks, trees). If None, no filtering applied.
    min_crown_area_m2 : minimum crown area to flag (meters²)
    crown_dhs_threshold : ΔHS threshold for crown candidates (meters, negative)
    resolution : grid cell size (meters)

    Returns
    -------
    dict with:
      'detected': bool - whether an avalanche was detected
      'crown_mask': 2D bool array - identified crown cells
      'deposit_mask': 2D bool array - identified deposit cells
      'avalanche_dhs': 2D array - estimated avalanche ΔHS (crown + deposit)
      'wind_dhs': 2D array - residual ΔHS attributed to wind
      'crown_area_m2': float
      'crown_volume_m3': float
      'mean_crown_dhs': float
      'confidence': str - 'high', 'medium', 'low'
    """
    valid = ~np.isnan(dhs) & ~np.isnan(dem)

    # Transport = ΔHS minus station background
    transport = np.where(valid, dhs - stn_dhs, np.nan)

    # --- Crown candidates ---
    # Steep terrain with large negative departure from station
    crown_cand = (valid &
                  (slope > 25) &
                  (transport < crown_dhs_threshold))

    # Remove persistent anomalies (rocks, trees)
    if persistent_anomaly_mask is not None:
        crown_cand = crown_cand & ~persistent_anomaly_mask

    # Connected components
    labeled, n_regions = connected_components(crown_cand)

    min_cells = int(min_crown_area_m2 / (resolution ** 2))

    # Filter by size and find the largest crown-like feature
    candidates = []
    for region_id in range(1, n_regions + 1):
        region_mask = labeled == region_id
        n_cells = region_mask.sum()

        if n_cells < min_cells:
            continue

        rows, cols = np.where(region_mask)
        mean_elev = np.nanmean(dem[region_mask])
        mean_dhs = np.nanmean(dhs[region_mask])
        mean_slope = np.nanmean(slope[region_mask])
        total_volume = np.nansum(transport[region_mask]) * resolution ** 2

        candidates.append({
            'region_id': region_id,
            'mask': region_mask,
            'n_cells': n_cells,
            'area_m2': n_cells * resolution ** 2,
            'mean_elev': mean_elev,
            'mean_dhs': mean_dhs,
            'mean_slope': mean_slope,
            'volume_m3': total_volume,
            'row_range': (rows.min(), rows.max()),
            'col_range': (cols.min(), cols.max()),
        })

    if not candidates:
        return {
            'detected': False,
            'crown_mask': np.zeros_like(valid, dtype=bool),
            'deposit_mask': np.zeros_like(valid, dtype=bool),
            'avalanche_dhs': np.zeros_like(dhs),
            'wind_dhs': dhs.copy(),
            'crown_area_m2': 0,
            'crown_volume_m3': 0,
            'mean_crown_dhs': 0,
            'confidence': 'none',
            'candidates': [],
        }

    # Sort by volume (most mass lost)
    candidates.sort(key=lambda x: x['volume_m3'])

    return {
        'detected': len(candidates) > 0,
        'candidates': candidates,
        'crown_mask': candidates[0]['mask'] if candidates else np.zeros_like(valid, dtype=bool),
        'deposit_mask': np.zeros_like(valid, dtype=bool),  # TODO: deposit detection
        'avalanche_dhs': np.zeros_like(dhs),  # TODO: full footprint
        'wind_dhs': dhs.copy(),
        'crown_area_m2': candidates[0]['area_m2'] if candidates else 0,
        'crown_volume_m3': candidates[0]['volume_m3'] if candidates else 0,
        'mean_crown_dhs': candidates[0]['mean_dhs'] if candidates else 0,
        'confidence': 'low',  # until validated
    }


def build_persistent_anomaly_mask(survey_grids: dict,
                                    dem: np.ndarray,
                                    slope: np.ndarray,
                                    stn_hs_values: dict,
                                    threshold: float = -0.3,
                                    min_periods_frac: float = 0.7) -> np.ndarray:
    """
    Identify cells that show anomalous negative ΔHS in most periods.
    These are likely rocks, trees, or terrain features — not avalanches.

    A cell is "persistent" if its transport is below threshold in
    >min_periods_frac of all survey periods.

    Parameters
    ----------
    survey_grids : dict of date_str -> 2D HS array (sorted by date)
    dem : 2D elevation array
    slope : 2D slope array
    stn_hs_values : dict of date_str -> station HS in meters
    threshold : transport threshold (meters, negative)
    min_periods_frac : fraction of periods cell must be anomalous

    Returns
    -------
    boolean mask (True = persistent anomaly, likely rock/tree)
    """
    dates = sorted(survey_grids.keys())
    valid = ~np.isnan(dem)
    n_anomalous = np.zeros(dem.shape, dtype=int)
    n_valid_periods = 0

    for i in range(len(dates) - 1):
        d_a, d_b = dates[i], dates[i + 1]
        hs_a = survey_grids[d_a]
        hs_b = survey_grids[d_b]

        if d_a not in stn_hs_values or d_b not in stn_hs_values:
            continue

        stn_dhs = stn_hs_values[d_b] - stn_hs_values[d_a]
        pair_valid = valid & ~np.isnan(hs_a) & ~np.isnan(hs_b)
        transport = np.where(pair_valid, (hs_b - hs_a) - stn_dhs, np.nan)

        anomalous = pair_valid & (transport < threshold) & (slope > 25)
        n_anomalous += anomalous.astype(int)
        n_valid_periods += 1

    if n_valid_periods == 0:
        return np.zeros(dem.shape, dtype=bool)

    frac = n_anomalous / max(n_valid_periods, 1)
    persistent = frac >= min_periods_frac

    return persistent


def separate_avalanche_from_wind(dhs: np.ndarray,
                                  avalanche_mask: np.ndarray,
                                  stn_dhs: float,
                                  valid_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separate avalanche ΔHS from wind transport ΔHS.

    In the avalanche footprint, replace the observed ΔHS with an estimate
    of what wind-only transport would have been (interpolated from
    surrounding non-avalanche cells).

    Returns (avalanche_dhs, wind_transport)
    """
    from scipy.ndimage import uniform_filter

    transport = np.where(valid_mask, dhs - stn_dhs, np.nan)

    # Estimate wind-only transport in the avalanche zone by
    # interpolating from surrounding cells
    wind_transport = transport.copy()

    # Replace avalanche zone with NaN, then fill by spatial smoothing
    wind_transport[avalanche_mask] = np.nan

    # Iterative gap-fill by expanding the smoothing window
    for window in [10, 20, 50]:
        still_nan = np.isnan(wind_transport) & valid_mask
        if not still_nan.any():
            break
        fill = np.where(np.isnan(wind_transport), 0, wind_transport)
        weight = (~np.isnan(wind_transport)).astype(float)
        num = uniform_filter(fill, size=window, mode='constant')
        den = uniform_filter(weight, size=window, mode='constant')
        filled = np.where(den > 0.1, num / den, np.nan)
        wind_transport = np.where(still_nan, filled, wind_transport)

    # Avalanche ΔHS = total transport minus wind-only estimate
    avy_dhs = np.where(avalanche_mask & valid_mask,
                        transport - wind_transport, 0)

    return avy_dhs, wind_transport


def apply_avalanche_event(hourly_grids: dict,
                           avalanche_dhs: np.ndarray,
                           event_timestamp,
                           valid_mask: np.ndarray) -> dict:
    """
    Insert an avalanche as an instantaneous event at the given timestamp.

    Finds the nearest hourly timestep and applies the avalanche ΔHS
    in a single step rather than distributing it across the period.
    """
    timestamps = sorted(hourly_grids.keys())

    # Find nearest timestamp
    nearest_ts = min(timestamps, key=lambda ts: abs(ts - event_timestamp))
    nearest_idx = timestamps.index(nearest_ts)

    # Apply avalanche ΔHS at this timestep and propagate forward
    for i in range(nearest_idx, len(timestamps)):
        ts = timestamps[i]
        grid = hourly_grids[ts]
        grid = np.where(valid_mask, grid + avalanche_dhs, grid)
        grid = np.maximum(grid, 0)
        hourly_grids[ts] = grid

    return hourly_grids


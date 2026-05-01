"""
Avalanche event handling for gap-fill pipeline.

Two modes:
  1. MANUAL: Known events specified in a JSON file with approximate
     location and timing. The system delineates the footprint using
     HS derivatives and corrects the transport field.

  2. AUTO: Scans all periods for anomalous HS discontinuities.
     Reports candidates for review — does NOT auto-correct without
     confirmation. Useful for discovering unknown events.

Manual event file format (data/boundaries/avalanche_events.json):
[
  {
    "period": "2026-01-14__2026-01-20",
    "timestamp": "2026-01-18T12:00:00",
    "size": "D2",
    "trigger": "skier",
    "location_hint": "upper slope, skiers left",
    "notes": "Skier-triggered persistent slab"
  }
]
"""

import numpy as np
import json
import pandas as pd
from pathlib import Path
from scipy.ndimage import uniform_filter, label as connected_components
from typing import Tuple, Optional


# =====================================================================
# Crown delineation (given approximate location or full-slope scan)
# =====================================================================

def delineate_crown(hs_before: np.ndarray,
                     hs_after: np.ndarray,
                     dem: np.ndarray,
                     stn_dhs: float,
                     new_gradient_threshold: float = 0.10,
                     dhs_anomaly_threshold: float = -0.15,
                     min_region_cells: int = 20,
                     min_crown_cells: int = 3,
                     smoothing: int = 3) -> list:
    """
    Delineate avalanche crown and slab regions from survey pair.

    Finds regions that have BOTH:
      - Crown cells: where downslope HS gradient INCREASED between surveys
        (a new step appeared in the snow surface)
      - Slab cells: negative ΔHS anomaly relative to station
        (cell lost more snow than background)

    Regions must have net negative volume (mass removal).

    Returns list of candidate regions sorted by volume (most erosion first).
    """
    valid = ~np.isnan(dem) & ~np.isnan(hs_before) & ~np.isnan(hs_after)
    fill_dem = np.where(np.isnan(dem), np.nanmean(dem), dem)

    # Terrain
    dy_dem, dx_dem = np.gradient(fill_dem, 1.0)
    slope = np.degrees(np.arctan(np.sqrt(dx_dem**2 + dy_dem**2)))
    slope_mag = np.maximum(np.sqrt(dx_dem**2 + dy_dem**2), 0.001)
    ds_x = dx_dem / slope_mag
    ds_y = dy_dem / slope_mag

    # Smooth HS
    hs_b = uniform_filter(np.where(np.isnan(hs_before), 0, hs_before), size=smoothing)
    hs_a = uniform_filter(np.where(np.isnan(hs_after), 0, hs_after), size=smoothing)

    # Downslope HS gradient change
    dy_b, dx_b = np.gradient(hs_b, 1.0)
    dy_a, dx_a = np.gradient(hs_a, 1.0)
    dhs_ds_before = dx_b * ds_x + dy_b * ds_y
    dhs_ds_after = dx_a * ds_x + dy_a * ds_y
    gradient_change = dhs_ds_after - dhs_ds_before

    # Cross-slope gradient change (flanks)
    cs_x, cs_y = -ds_y, ds_x
    dhs_cs_before = dx_b * cs_x + dy_b * cs_y
    dhs_cs_after = dx_a * cs_x + dy_a * cs_y
    cross_gradient_change = np.abs(dhs_cs_after) - np.abs(dhs_cs_before)

    # Crown line: new step DOWN in downslope direction
    crown_line = valid & (gradient_change < -new_gradient_threshold) & (slope > 20)

    # Flank line: new cross-slope gradient increase
    flank_line = valid & (cross_gradient_change > new_gradient_threshold) & (slope > 20)

    # ΔHS anomaly
    dhs = np.where(valid, hs_after - hs_before, np.nan)
    dhs_anomaly = dhs - stn_dhs

    # Slab zone: erosion anomaly on steep terrain
    slab_zone = valid & (dhs_anomaly < dhs_anomaly_threshold) & (slope > 20)

    # Combine boundary (crown + flanks) with slab, find connected regions
    boundary = crown_line | flank_line
    combined = boundary | slab_zone
    labeled, n_regions = connected_components(combined)

    regions = []
    for rid in range(1, n_regions + 1):
        mask = labeled == rid
        n_cells = mask.sum()
        if n_cells < min_region_cells:
            continue

        n_crown = int((mask & crown_line).sum())
        n_flank = int((mask & flank_line).sum())
        n_boundary = int((mask & boundary).sum())

        # Must have crown or flank boundary cells
        if n_boundary < min_crown_cells:
            continue

        vol = float(np.nansum(dhs_anomaly[mask]))
        if vol >= 0:  # must be net erosion
            continue

        rows, cols = np.where(mask)
        regions.append({
            'mask': mask,
            'n_cells': int(n_cells),
            'n_crown_cells': n_crown,
            'n_flank_cells': n_flank,
            'n_boundary_cells': n_boundary,
            'volume_m3': vol,
            'mean_dhs': float(np.nanmean(dhs[mask])),
            'mean_dhs_anomaly': float(np.nanmean(dhs_anomaly[mask])),
            'mean_slope': float(np.nanmean(slope[mask])),
            'mean_elev': float(np.nanmean(dem[mask])),
            'elev_range': (float(np.nanmin(dem[mask])), float(np.nanmax(dem[mask]))),
            'row_range': (int(rows.min()), int(rows.max())),
        })

    regions.sort(key=lambda r: r['volume_m3'])  # most negative first
    return regions


# =====================================================================
# Manual event specification
# =====================================================================

def load_known_events(events_path: str) -> list:
    """
    Load known avalanche events from JSON file.

    Expected format:
    [
      {
        "period": "2026-01-14__2026-01-20",
        "timestamp": "2026-01-18T12:00:00",
        "size": "D2",
        "notes": "Skier-triggered persistent slab"
      }
    ]
    """
    path = Path(events_path)
    if not path.exists():
        return []
    with open(str(path)) as f:
        return json.load(f)


# =====================================================================
# Transport correction
# =====================================================================

def separate_avalanche_from_wind(transport: np.ndarray,
                                  avalanche_mask: np.ndarray,
                                  valid_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate wind-only transport in avalanche zone by spatial interpolation.
    Returns (avalanche_dhs, corrected_wind_transport).
    """
    wind_transport = transport.copy()
    wind_transport[avalanche_mask] = np.nan

    for window in [10, 20, 50]:
        still_nan = np.isnan(wind_transport) & valid_mask
        if not still_nan.any():
            break
        fill = np.where(np.isnan(wind_transport), 0, wind_transport)
        weight = (~np.isnan(wind_transport)).astype(float)
        num = uniform_filter(fill, size=window, mode='constant')
        den = uniform_filter(weight, size=window, mode='constant')
        with np.errstate(divide='ignore', invalid='ignore'):
            filled = np.where(den > 0.1, num / den, np.nan)
        wind_transport = np.where(still_nan, filled, wind_transport)

    avalanche_dhs = np.where(avalanche_mask & valid_mask,
                              transport - wind_transport, 0)
    return avalanche_dhs, wind_transport


def apply_avalanche_event(hourly_grids: dict,
                           avalanche_dhs: np.ndarray,
                           event_timestamp,
                           valid_mask: np.ndarray) -> dict:
    """Insert avalanche as instantaneous event at nearest timestep."""
    
    if not (timestamps := sorted(hourly_grids.keys())):
        return hourly_grids

    # Ensure event_timestamp has the same tz-awareness as the grid timestamps
    event_ts = pd.Timestamp(event_timestamp)
    if event_ts.tzinfo is None:
        event_ts = event_ts.tz_localize('UTC')
    grid_is_aware = timestamps[0].tzinfo is not None
    
    if grid_is_aware and event_ts.tzinfo is None:
        event_ts = event_ts.tz_localize('UTC')
    elif not grid_is_aware and event_ts.tzinfo is not None:
        event_ts = event_ts.tz_localize(None)

    nearest_ts = min(timestamps, key=lambda ts: abs(ts - event_timestamp))
    nearest_idx = timestamps.index(nearest_ts)

    for i in range(nearest_idx, len(timestamps)):
        ts = timestamps[i]
        hourly_grids[ts] = np.maximum(
            hourly_grids[ts] + np.where(valid_mask, avalanche_dhs, 0), 0)
    return hourly_grids

# =====================================================================
# Image-processing boundary detection
# =====================================================================

def load_slope_mask(kml_path, dem_shape: tuple, transform) -> np.ndarray:
    """
    Rasterize a KML polygon boundary to a boolean mask matching the DEM grid.
    KML coordinates are WGS84 lon/lat — reprojects to UTM before rasterizing.
    Returns all-True mask on any failure.
    """
    try:
        from xml.etree import ElementTree as ET
        import rasterio.features
        import rasterio.transform as rt
        from shapely.geometry import Polygon, mapping
        from pyproj import Transformer

        tree = ET.parse(str(kml_path))
        coords_text = None
        for tag in ['.//coordinates',
                    './/{http://www.opengis.net/kml/2.2}coordinates']:
            coords_text = tree.find(tag)
            if coords_text is not None:
                break
        if coords_text is None:
            print("  KML mask: no coordinates element found, using full domain")
            return np.ones(dem_shape, dtype=bool)

        pts_lonlat = []
        for part in coords_text.text.strip().split():
            xyz = part.split(',')
            if len(xyz) >= 2:
                pts_lonlat.append((float(xyz[0]), float(xyz[1])))

        if len(pts_lonlat) < 3:
            print("  KML mask: fewer than 3 points, using full domain")
            return np.ones(dem_shape, dtype=bool)

        # KML is always WGS84 lon/lat.
        # DEM transform origin tells us if it's projected (large coords = UTM)
        if hasattr(transform, 'c'):
            x0 = transform.c
        else:
            x0 = transform[2]

        if abs(x0) > 180:
            # Projected — reproject lon/lat to UTM zone 13N (A-Basin)
            transformer = Transformer.from_crs(
                'EPSG:4326', 'EPSG:32613', always_xy=True)
            pts = [transformer.transform(lon, lat)
                   for lon, lat in pts_lonlat]
        else:
            pts = pts_lonlat

        poly = Polygon(pts)
        aff  = transform if hasattr(transform, 'c') else                rt.Affine(transform[0], transform[1], transform[2],
                         transform[3], transform[4], transform[5])

        mask = rasterio.features.geometry_mask(
            [mapping(poly)], out_shape=dem_shape, transform=aff, invert=True)

        n_inside = int(mask.sum())
        print(f"  KML mask: {n_inside} / {mask.size} cells inside boundary")
        if n_inside == 0:
            print("  KML mask: 0 cells after rasterize — using full domain")
            return np.ones(dem_shape, dtype=bool)
        return mask

    except Exception as e:
        print(f"  KML mask: failed ({e}), using full domain")
        return np.ones(dem_shape, dtype=bool)


def build_persistent_noise_mask(survey_paths: list,
                                 dem: np.ndarray,
                                 min_negative_fraction: float = 0.6,
                                 smoothing: int = 3) -> np.ndarray:
    """
    Build a persistent noise mask from multiple survey pairs.

    Cells that show negative dHS anomaly in a majority of periods are
    structural noise (trees, rocks, persistent scour) rather than
    real avalanche signals. Returns True where cells are NOISE (to exclude).

    Parameters
    ----------
    survey_paths : list of (hs_before_path, hs_after_path, stn_dhs) tuples
    min_negative_fraction : fraction of periods a cell must be negative
        to be flagged as noise (default 0.6 = majority of periods)
    smoothing : uniform filter size to smooth the mask (pixels)
    """
    from scipy.ndimage import uniform_filter

    valid = ~np.isnan(dem)
    negative_count = np.zeros(dem.shape, dtype=np.float32)
    total_count    = np.zeros(dem.shape, dtype=np.float32)

    for hs_before_path, hs_after_path, stn_dhs in survey_paths:
        hs_b = np.load(str(hs_before_path))
        hs_a = np.load(str(hs_after_path))
        ok   = valid & ~np.isnan(hs_b) & ~np.isnan(hs_a)
        dhs  = np.where(ok, hs_a - hs_b, np.nan)
        anom = np.where(ok, dhs - stn_dhs, np.nan)

        negative_count += np.where(ok & (anom < 0), 1, 0)
        total_count    += ok.astype(float)

    with np.errstate(invalid='ignore', divide='ignore'):
        neg_frac = np.where(total_count > 0,
                             negative_count / total_count, 0)

    noise_mask = neg_frac >= min_negative_fraction

    # Smooth slightly to avoid single-pixel holes
    if smoothing > 1:
        noise_mask = uniform_filter(
            noise_mask.astype(float), size=smoothing) > 0.5

    n_noise = int(noise_mask.sum())
    pct     = 100 * n_noise / valid.sum() if valid.any() else 0
    print(f"  Persistent noise mask: {n_noise} cells ({pct:.1f}%) flagged")
    return noise_mask


def detect_boundaries(hs_before: np.ndarray,
                       hs_after: np.ndarray,
                       dem: np.ndarray,
                       stn_dhs: float,
                       transform=None,
                       kml_path=None,
                       hs_pre_event: np.ndarray = None,
                       stn_dhs_pre: float = None,
                       canny_sigma: float = 2.0,
                       canny_low: float = 0.05,
                       canny_high: float = 0.08,
                       erosion_threshold_sigma: float = 1.5,
                       deposit_threshold_sigma: float = 1.5,
                       min_area_m2: float = 225.0,
                       morph_radius: int = 3,
                       dem_roughness_threshold: float = 0.5,
                       min_slope_deg: float = 15.0,
                       persistent_noise_mask: np.ndarray = None,
                       start_zone_kml=None) -> dict:
    """
    Detect avalanche release and deposit boundaries using image processing.

    Pipeline:
      1. Apply KML slope boundary mask to restrict domain
      2. Canny edge detection on smoothed dHS anomaly — finds crown + flanks
      3. Watershed seeded from Canny crown edges — grows complete release region
         rather than relying solely on threshold amplitude
      4. Threshold-based deposit zone
      5. Morphological cleanup + minimum area filter

    Parameters
    ----------
    kml_path : path to KML slope boundary file (optional)
    hs_pre_event : HS grid from the survey BEFORE hs_before (optional).
        Used to compute a pre-event loading trend. Cells that did NOT
        accumulate snow in the period leading up to the event are excluded
        from the release zone — this filters persistent noise like trees
        and stable wind-scour features that show negative dHS every period.
    stn_dhs_pre : station dHS for the pre-event period (m), used to
        compute the pre-event anomaly. Defaults to 0 if not provided.
    canny_sigma : Gaussian smoothing before edge detection (meters)
    canny_low / canny_high : Canny hysteresis thresholds (0-1)
    erosion_threshold_sigma : dHS anomaly sigma for release seed
    deposit_threshold_sigma : dHS anomaly sigma for deposit seed
    min_area_m2 : minimum region area to retain (m² at 1m resolution)
    morph_radius : morphological disk radius for cleanup (pixels)
    """
    try:
        import warnings as _warnings
        from skimage.feature import canny
        from skimage.filters import gaussian
        with _warnings.catch_warnings():
            _warnings.simplefilter('ignore', FutureWarning)
            from skimage.morphology import (closing, opening, remove_small_objects,
                                             disk, dilation)
        from skimage.measure import find_contours
        from skimage.segmentation import watershed
        from scipy.ndimage import label as nd_label, distance_transform_edt
    except ImportError as e:
        raise ImportError(
            f"scikit-image required for boundary detection: {e}\n"
            "Install: pip install scikit-image") from e

    valid = ~np.isnan(dem) & ~np.isnan(hs_before) & ~np.isnan(hs_after)

    # Apply KML slope mask
    if kml_path is not None:
        slope_mask = load_slope_mask(kml_path, dem.shape, transform)
        valid = valid & slope_mask

    dhs         = np.where(valid, hs_after - hs_before, np.nan)
    dhs_anomaly = np.where(valid, dhs - stn_dhs, np.nan)

    # Smooth for edge detection
    dhs_fill   = np.where(np.isnan(dhs_anomaly), 0, dhs_anomaly)
    dhs_smooth = gaussian(dhs_fill, sigma=canny_sigma)

    # Normalize to [0,1] within valid domain
    vals = dhs_smooth[valid]
    if len(vals) == 0:
        raise ValueError(
            "No valid cells after masking — check KML boundary covers the DEM")
    vmin, vmax = vals.min(), vals.max()
    dhs_range = vmax - vmin if (vmax - vmin) > 1e-6 else 1.0
    dhs_norm  = (dhs_smooth - vmin) / dhs_range
    dhs_norm[~valid] = 0.5

    # --- Canny edges ---
    edges = canny(dhs_norm, sigma=1.0,
                  low_threshold=canny_low,
                  high_threshold=canny_high,
                  mask=valid)

    # --- Watershed seeded from Canny crown edges ---
    # Markers: 1=background, 2=release (seeded from strong erosion near edges)
    # --- Statistics over valid domain ---
    anom_valid = dhs_anomaly[valid]
    anom_std   = np.nanstd(anom_valid)
    anom_mean  = np.nanmean(anom_valid)

    # Terrain slope
    fill_dem = np.where(np.isnan(dem), np.nanmean(dem), dem)
    dy, dx   = np.gradient(fill_dem, 1.0)
    slope    = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

    # DEM local roughness mask — use Laplacian (slope-independent).
    # Local std of DEM values captures slope gradient, not true roughness.
    # Laplacian (second derivative) is ~0 on uniform slopes and high on
    # rocks, tree stumps, small gullies — exactly what we want to filter.
    from scipy.ndimage import uniform_filter, laplace
    dem_roughness  = np.abs(laplace(fill_dem))
    # Smooth the roughness field to avoid single-pixel artifacts
    dem_roughness  = uniform_filter(dem_roughness, size=3)
    smooth_terrain = dem_roughness < dem_roughness_threshold

    # Pre-event loading filter — exclude very persistently eroding cells
    if hs_pre_event is not None:
        pre_stn      = stn_dhs_pre if stn_dhs_pre is not None else 0.0
        dhs_pre      = np.where(valid & ~np.isnan(hs_pre_event),
                                 hs_before - hs_pre_event, np.nan)
        pre_anomaly  = np.where(~np.isnan(dhs_pre), dhs_pre - pre_stn, np.nan)
        pre_anom_std = np.nanstd(pre_anomaly[valid])
        loading_mask = (np.isnan(pre_anomaly) |
                        (pre_anomaly >= -2.0 * pre_anom_std))
    else:
        loading_mask = np.ones(dem.shape, dtype=bool)

    # Dilate edges for boundary use
    edges_dilated = dilation(edges, disk(1))

    # --- Release zone: direct threshold (more robust than watershed) ---
    # Watershed was unreliable when dense Canny edges covered the domain.
    # Direct thresholding on dHS anomaly is simpler and more predictable.
    # Persistent noise mask — cells that are chronically negative
    if persistent_noise_mask is not None:
        noise_free = ~persistent_noise_mask
    else:
        noise_free = np.ones(dem.shape, dtype=bool)

    # Start zone mask — restrict release detection to known release terrain
    if start_zone_kml is not None:
        start_zone_mask = load_slope_mask(start_zone_kml, dem.shape, transform)
        n_sz = start_zone_mask.sum()
        print(f"  Start zone mask: {n_sz} cells")
        if n_sz > 0:
            # Recompute statistics within start zone for better threshold sensitivity
            sz_valid = valid & start_zone_mask
            sz_anom  = dhs_anomaly[sz_valid]
            if len(sz_anom) > 10:
                anom_std  = np.nanstd(sz_anom)
                anom_mean = np.nanmean(sz_anom)
                print(f"  Start zone stats: mean={anom_mean:.3f}m  std={anom_std:.3f}m  "
                      f"threshold={anom_mean - erosion_threshold_sigma*anom_std:.3f}m")
    else:
        start_zone_mask = np.ones(dem.shape, dtype=bool)

    # Strong seeds: cells clearly below threshold — starting points for growing
    strong_seeds = (valid &
                    start_zone_mask &
                    (dhs_anomaly < anom_mean - erosion_threshold_sigma * anom_std) &
                    (slope >= min_slope_deg) &
                    smooth_terrain &
                    loading_mask &
                    noise_free)

    if strong_seeds.any():
        # Seeded region growing via watershed.
        # Seeds grow into adjacent negative-anomaly cells within start zone,
        # but are stopped by Canny edges (set as barriers = high surface value).
        # This captures the full release zone including cells with moderate
        # negative dHS adjacent to the strongly-eroded core.
        ws_surface = np.where(valid, -dhs_anomaly, 0)
        # Canny edges are hard barriers — set maximum surface value
        ws_surface = np.where(edges_dilated, 1e6, ws_surface)

        markers = np.zeros(dem.shape, dtype=np.int32)
        # Background seeds: cells with POSITIVE anomaly inside start zone
        # (snow was added here, definitely not part of the release zone)
        positive_cells = (valid & start_zone_mask &
                          (dhs_anomaly > anom_mean + 0.5 * anom_std))
        markers[valid & ~start_zone_mask] = 1   # outside start zone
        markers[positive_cells]            = 1   # positive cells = background
        markers[edges_dilated & valid]     = 1   # Canny edges = background
        markers[strong_seeds]              = 2   # release seeds

        ws_labels = watershed(ws_surface, markers, mask=valid & start_zone_mask)
        release_raw = (ws_labels == 2) & valid & start_zone_mask
    else:
        print("  No strong seeds in start zone — release zone not detected")
        release_raw = np.zeros(dem.shape, dtype=bool)

    # --- Deposit zone: threshold + smooth terrain ---
    deposit_raw = (valid &
                   (dhs_anomaly > anom_mean + deposit_threshold_sigma * anom_std) &
                   smooth_terrain &
                   noise_free)

    # --- Morphological cleanup + area filter ---
    import warnings as _w
    selem      = disk(morph_radius)
    min_cells  = max(1, int(min_area_m2))

    with _w.catch_warnings():
        _w.simplefilter('ignore', FutureWarning)
        release_mask = remove_small_objects(
            closing(opening(release_raw, selem), selem),
            min_size=min_cells)
        deposit_mask = remove_small_objects(
            closing(opening(deposit_raw, selem), selem),
            min_size=min_cells)

    # --- Contours ---
    release_contours = find_contours(release_mask.astype(float), 0.5)
    deposit_contours = find_contours(deposit_mask.astype(float), 0.5)

    rel_area = float(release_mask.sum())
    dep_area = float(deposit_mask.sum())
    rel_vol  = float(np.nansum(dhs_anomaly[release_mask])) if release_mask.any() else 0.0
    dep_vol  = float(np.nansum(dhs_anomaly[deposit_mask])) if deposit_mask.any() else 0.0

    return {
        'release_mask':     release_mask,
        'deposit_mask':     deposit_mask,
        'crown_edges':      edges_dilated,
        'dhs_anomaly':      dhs_anomaly,
        'release_contours': release_contours,
        'deposit_contours': deposit_contours,
        'release_area_m2':  rel_area,
        'deposit_area_m2':  dep_area,
        'release_volume_m3': rel_vol,
        'deposit_volume_m3': dep_vol,
        'start_zone_mask':  (start_zone_mask
                             if start_zone_kml is not None else None),
        'params': {
            'canny_sigma':             canny_sigma,
            'canny_low':               canny_low,
            'canny_high':              canny_high,
            'erosion_threshold_sigma': erosion_threshold_sigma,
            'deposit_threshold_sigma': deposit_threshold_sigma,
            'min_area_m2':             min_area_m2,
            'morph_radius':            morph_radius,
            'pre_event_filter':        hs_pre_event is not None,
            'dem_roughness_threshold': dem_roughness_threshold,
            'min_slope_deg':            min_slope_deg,
            'persistent_noise_mask':    persistent_noise_mask is not None,
            'start_zone_kml':           start_zone_kml is not None,
        },
    }


# =====================================================================
# Min-kernel boundary detection (alternative to Canny+watershed)
# =====================================================================

def detect_minkernel(hs_before: np.ndarray,
                     hs_after: np.ndarray,
                     dem: np.ndarray,
                     stn_dhs: float,
                     kernel_size: int = 7,
                     threshold_sigma: float = 1.2,
                     min_area_m2: float = 200.0,
                     min_slope_deg: float = 20.0,
                     max_deposit_slope_deg: float = 25.0,
                     smooth_size: int = 3,
                     persistent_noise_mask: np.ndarray = None,
                     start_zone_mask: np.ndarray = None,
                     domain_mask: np.ndarray = None,
                     transform=None) -> dict:
    """
    Detect avalanche release and deposit zones using min/max kernel filters.

    Simpler and more robust than Canny+watershed — only 3 tunable parameters.
    The minimum filter replaces each cell with the minimum dHS in its
    neighborhood: connected erosion zones are preserved, isolated noise
    is suppressed.

    Parameters
    ----------
    kernel_size      : min/max filter window size in pixels (default 7).
    threshold_sigma  : std devs below mean anomaly for release threshold
                       (default 1.2).
    min_area_m2      : minimum region area to keep.
    min_slope_deg    : minimum slope for release zone.
    max_deposit_slope_deg : maximum slope for deposit zone.
    smooth_size      : pre-smoothing kernel size for dHS.
    persistent_noise_mask : True where cells are chronic noise (excluded).
    start_zone_mask  : True where release is physically possible.
    domain_mask      : True where data is valid (overrides NaN check).
    transform        : rasterio Affine (for contour generation).

    Returns
    -------
    dict matching detect_boundaries() output format:
        release_mask, deposit_mask, dhs_anomaly,
        release_area_m2, deposit_area_m2,
        release_volume_m3, deposit_volume_m3,
        release_contours, deposit_contours,
        start_zone_mask, params
    """
    from scipy.ndimage import (minimum_filter, maximum_filter,
                               binary_fill_holes, binary_dilation,
                               binary_erosion)
    from skimage.measure import find_contours

    valid = ~np.isnan(dem) & ~np.isnan(hs_before) & ~np.isnan(hs_after)
    if domain_mask is not None:
        valid = valid & domain_mask

    # Terrain slope
    fill_dem = np.where(np.isnan(dem), np.nanmean(dem), dem)
    dy, dx = np.gradient(fill_dem, 1.0)
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

    # dHS anomaly
    dhs = np.where(valid, hs_after - hs_before, np.nan)
    dhs_anomaly = np.where(valid, dhs - stn_dhs, np.nan)

    # Pre-smooth then apply min/max filters
    dhs_fill = np.where(np.isnan(dhs_anomaly), 0, dhs_anomaly)
    dhs_smooth = uniform_filter(dhs_fill, size=smooth_size)
    dhs_minfiltered = minimum_filter(dhs_smooth, size=kernel_size)
    dhs_maxfiltered = maximum_filter(dhs_smooth, size=kernel_size)

    # Threshold statistics
    if start_zone_mask is not None:
        ref_vals = dhs_anomaly[valid & start_zone_mask]
    else:
        ref_vals = dhs_anomaly[valid]

    if len(ref_vals) == 0:
        return _empty_minkernel_result(dem.shape, dhs_anomaly,
                                       start_zone_mask)

    anom_mean = float(np.nanmean(ref_vals))
    anom_std = float(np.nanstd(ref_vals))
    release_threshold = anom_mean - threshold_sigma * anom_std
    deposit_threshold = anom_mean + threshold_sigma * anom_std

    print(f"  Min-kernel stats: mean={anom_mean:.3f}m  std={anom_std:.3f}m")
    print(f"  Release threshold: {release_threshold:.3f}m  "
          f"Deposit threshold: {deposit_threshold:.3f}m")

    # --- Release zone ---
    release_candidates = (
        valid &
        (dhs_minfiltered < release_threshold) &
        (slope >= min_slope_deg)
    )
    if start_zone_mask is not None:
        release_candidates = release_candidates & start_zone_mask
    if persistent_noise_mask is not None:
        release_candidates = release_candidates & ~persistent_noise_mask

    release_candidates = binary_fill_holes(release_candidates)
    release_candidates = binary_erosion(release_candidates, iterations=1)
    release_candidates = binary_dilation(release_candidates, iterations=1)

    release_labeled, n_release = connected_components(release_candidates)
    release_mask = np.zeros(dem.shape, dtype=bool)
    for rid in range(1, n_release + 1):
        region = release_labeled == rid
        if int(region.sum()) >= min_area_m2:
            release_mask |= region

    # --- Deposit zone ---
    deposit_candidates = (
        valid &
        (dhs_maxfiltered > deposit_threshold) &
        (slope <= max_deposit_slope_deg) &
        ~release_mask
    )
    if persistent_noise_mask is not None:
        deposit_candidates = deposit_candidates & ~persistent_noise_mask

    deposit_candidates = binary_fill_holes(deposit_candidates)
    deposit_labeled, n_deposit = connected_components(deposit_candidates)
    deposit_mask = np.zeros(dem.shape, dtype=bool)
    for rid in range(1, n_deposit + 1):
        region = deposit_labeled == rid
        if int(region.sum()) >= min_area_m2:
            deposit_mask |= region

    # Contours
    release_contours = find_contours(release_mask.astype(float), 0.5)
    deposit_contours = find_contours(deposit_mask.astype(float), 0.5)

    rel_area = float(release_mask.sum())
    dep_area = float(deposit_mask.sum())
    rel_vol = float(np.nansum(dhs_anomaly[release_mask])) \
        if release_mask.any() else 0.0
    dep_vol = float(np.nansum(dhs_anomaly[deposit_mask])) \
        if deposit_mask.any() else 0.0

    return {
        'release_mask':      release_mask,
        'deposit_mask':      deposit_mask,
        'crown_edges':       np.zeros(dem.shape, dtype=bool),
        'dhs_anomaly':       dhs_anomaly,
        'release_contours':  release_contours,
        'deposit_contours':  deposit_contours,
        'release_area_m2':   rel_area,
        'deposit_area_m2':   dep_area,
        'release_volume_m3': rel_vol,
        'deposit_volume_m3': dep_vol,
        'start_zone_mask':   start_zone_mask,
        'params': {
            'method':              'minkernel',
            'kernel_size':         kernel_size,
            'threshold_sigma':     threshold_sigma,
            'min_area_m2':         min_area_m2,
            'min_slope_deg':       min_slope_deg,
            'smooth_size':         smooth_size,
            'persistent_noise_mask': persistent_noise_mask is not None,
            'start_zone_mask':     start_zone_mask is not None,
        },
    }


def _empty_minkernel_result(shape, dhs_anomaly, start_zone_mask):
    from skimage.measure import find_contours
    return {
        'release_mask':      np.zeros(shape, dtype=bool),
        'deposit_mask':      np.zeros(shape, dtype=bool),
        'crown_edges':       np.zeros(shape, dtype=bool),
        'dhs_anomaly':       dhs_anomaly,
        'release_contours':  [],
        'deposit_contours':  [],
        'release_area_m2':   0.0,
        'deposit_area_m2':   0.0,
        'release_volume_m3': 0.0,
        'deposit_volume_m3': 0.0,
        'start_zone_mask':   start_zone_mask,
        'params': {'method': 'minkernel'},
    }


def contours_to_geojson(contours: list, transform) -> dict:
    """
    Convert skimage pixel-coordinate contours to a GeoJSON FeatureCollection.

    transform: rasterio Affine transform for the DEM.
    """
    def px_to_geo(row, col):
        x = transform[2] + col * transform[0]
        y = transform[5] + row * transform[4]
        return [x, y]

    features = []
    for contour in contours:
        coords = [px_to_geo(r, c) for r, c in contour]
        if len(coords) >= 4:
            coords.append(coords[0])  # close ring
            features.append({
                'type': 'Feature',
                'geometry': {'type': 'Polygon', 'coordinates': [coords]},
                'properties': {}
            })
    return {'type': 'FeatureCollection', 'features': features}


# =====================================================================
# Standalone tuning runner — Jan 14-20 surveys
# =====================================================================

if __name__ == '__main__':
    import argparse
    import sys
    import rasterio
    from pathlib import Path

    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))

    from config import ProjectConfig
    from plots import plot_avalanche_boundaries

    parser = argparse.ArgumentParser(
        description="Tune avalanche boundary detection on Jan 14-20 survey pair")
    parser.add_argument('--project-dir',
                        default='/home/ron/snowpack_model_feeder')
    parser.add_argument('--canny-sigma',   type=float, default=2.0)
    parser.add_argument('--canny-low',     type=float, default=0.05)
    parser.add_argument('--canny-high',    type=float, default=0.08)
    parser.add_argument('--erosion-sigma', type=float, default=1.5)
    parser.add_argument('--deposit-sigma', type=float, default=1.5)
    parser.add_argument('--min-area',      type=float, default=225.0)
    parser.add_argument('--morph-radius',  type=int,   default=3)
    parser.add_argument('--no-kml',        action='store_true',
                        help="Disable KML slope boundary masking")
    parser.add_argument('--min-slope',     type=float, default=15.0,
                        help="Minimum slope angle for release zone (degrees, default 15)")
    parser.add_argument('--dem-roughness', type=float, default=0.5,
                        help="Local DEM std threshold to exclude rough terrain (m)")
    parser.add_argument('--no-pre-event',  action='store_true',
                        help="Disable pre-event loading trend filter")
    parser.add_argument('--start-zone-kml', default=None,
                        help="KML file for release zone boundary "
                             "(default: data/boundaries/Litte_prof_start_zone.kml)")
    parser.add_argument('--no-noise-mask', action='store_true',
                        help="Disable persistent noise mask")
    parser.add_argument('--noise-fraction', type=float, default=0.6,
                        help="Fraction of periods negative to flag as noise (default 0.6)")
    parser.add_argument('--pre-event-survey', default=None,
                        help="Pre-event survey filename (default: survey before --date-before)")
    parser.add_argument('--date-before',   default='2026-01-14',
                        help="Date of 'before' survey (YYYY-MM-DD, default: 2026-01-14)")
    parser.add_argument('--date-after',    default='2026-01-20',
                        help="Date of 'after' survey (YYYY-MM-DD, default: 2026-01-20)")
    parser.add_argument('--method', default='minkernel',
                        choices=['minkernel', 'canny'],
                        help="Detection method: minkernel (default) or canny")
    parser.add_argument('--kernel-size',  type=int,   default=7,
                        help="Min-kernel filter size in pixels (default 7)")
    parser.add_argument('--threshold-sigma-mk', type=float, default=1.2,
                        help="Min-kernel threshold in std devs (default 1.2)")
    parser.add_argument('--output-dir',    default=None)
    args = parser.parse_args()

    cfg = ProjectConfig(project_dir=Path(args.project_dir))
    out_dir = Path(args.output_dir) if args.output_dir else cfg.plots_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load DEM
    with rasterio.open(str(cfg.resampled_dir / "dem_1m.tif")) as src:
        dem = src.read(1).astype(np.float32)
        dem[dem == src.nodata] = np.nan
        transform = src.transform

    # Load survey grids
    import pandas as pd
    date_before = args.date_before
    date_after  = args.date_after
    period_id   = f"{date_before}__{date_after}"

    hs_before_path = cfg.resampled_dir / f"hs_{date_before}.npy"
    hs_after_path  = cfg.resampled_dir / f"hs_{date_after}.npy"

    if not hs_before_path.exists() or not hs_after_path.exists():
        print(f"ERROR: Survey grids not found in {cfg.resampled_dir}")
        print(f"  Expected: {hs_before_path.name}, {hs_after_path.name}")
        available = sorted(cfg.resampled_dir.glob("hs_*.npy"))
        print(f"  Available: {[p.name for p in available]}")
        sys.exit(1)

    hs_before = np.load(str(hs_before_path))
    hs_after  = np.load(str(hs_after_path))
    print(f"Period: {date_before} -> {date_after}")

    # Station dHS for this period
    try:
        wx_path = cfg.project_dir / "data" / "weather" / "weather_data.csv"
        wx = pd.read_csv(str(wx_path), parse_dates=[0], index_col=0)
        hs_col = wx.iloc[:, 11]
        period = hs_col.loc[date_before:date_after]
        stn_dhs = float((period.iloc[-1] - period.iloc[0]) * 0.1 * 2.54)
        print(f"Station dHS {date_before}->{date_after}: {stn_dhs:.1f} cm")
    except Exception as e:
        print(f"Warning: could not load station dHS ({e}), using 0.0")
        stn_dhs = 0.0

    # Initialize optional inputs
    hs_pre_event = None
    stn_dhs_pre  = None

    # Load pre-event survey for loading trend filter
    if not args.no_pre_event:
        # Auto-detect: find the survey immediately before date_before
        if args.pre_event_survey:
            pre_candidates = [cfg.resampled_dir / args.pre_event_survey]
        else:
            all_surveys = sorted(cfg.resampled_dir.glob("hs_*.npy"))
            pre_candidates = [s for s in all_surveys
                              if s.name < f"hs_{date_before}.npy"]
            pre_candidates = pre_candidates[-1:] if pre_candidates else []

        if pre_candidates and pre_candidates[0].exists():
            pre_path = pre_candidates[0]
            pre_date = pre_path.name.replace('hs_','').replace('.npy','')
            hs_pre_event = np.load(str(pre_path))
            try:
                wx_path = cfg.project_dir / "data" / "weather" / "weather_data.csv"
                wx = pd.read_csv(str(wx_path), parse_dates=[0], index_col=0)
                hs_col = wx.iloc[:, 11]
                pre_period = hs_col.loc[pre_date:date_before]
                stn_dhs_pre = float(
                    (pre_period.iloc[-1] - pre_period.iloc[0]) * 0.1 * 2.54)
                print(f"Pre-event survey: {pre_path.name}  "
                      f"station dHS: {stn_dhs_pre:.1f} cm")
            except Exception as e:
                print(f"  Pre-event station dHS: failed ({e}), using 0")
                stn_dhs_pre = 0.0
        else:
            print("  No pre-event survey found, skipping loading filter")

    kml_path = None if args.no_kml else cfg.boundary_kml
    if kml_path and not kml_path.exists():
        print(f"  KML not found at {kml_path}, running without boundary mask")
        kml_path = None

    # Build persistent noise mask from all non-event survey pairs
    persistent_noise_mask = None
    if not args.no_noise_mask:
        all_surveys = sorted(cfg.resampled_dir.glob("hs_*.npy"))
        survey_pairs = []
        for i in range(len(all_surveys) - 1):
            d_b = all_surveys[i].name.replace("hs_","").replace(".npy","")
            d_a = all_surveys[i+1].name.replace("hs_","").replace(".npy","")
            if d_b == date_before and d_a == date_after:
                continue
            survey_pairs.append((all_surveys[i], all_surveys[i+1], 0.0))
        if survey_pairs:
            print(f"Building noise mask from {len(survey_pairs)} survey pairs...")
            persistent_noise_mask = build_persistent_noise_mask(
                survey_pairs, dem,
                min_negative_fraction=args.noise_fraction)
        else:
            print("  No survey pairs available for noise mask")

    # Resolve start zone KML
    if args.start_zone_kml:
        start_zone_kml_path = Path(args.start_zone_kml)
    else:
        start_zone_kml_path = cfg.start_zone_kml \
            if cfg.start_zone_kml.exists() else None
    if start_zone_kml_path:
        print(f"Start zone KML: {start_zone_kml_path.name}")
    else:
        print("Start zone KML: not found, using full domain for release")

    if args.method == 'minkernel':
        # --- Min-kernel detection ---
        # Load start zone mask for min-kernel
        mk_start_zone = None
        if start_zone_kml_path:
            mk_start_zone = load_slope_mask(
                start_zone_kml_path, dem.shape, transform)

        print(f"\nRunning MIN-KERNEL boundary detection:")
        print(f"  kernel_size={args.kernel_size}  "
              f"threshold_sigma={args.threshold_sigma_mk}")
        print(f"  min_area={args.min_area}m²  min_slope={args.min_slope}°")

        result = detect_minkernel(
            hs_before, hs_after, dem, stn_dhs,
            kernel_size=args.kernel_size,
            threshold_sigma=args.threshold_sigma_mk,
            min_area_m2=args.min_area,
            min_slope_deg=args.min_slope,
            persistent_noise_mask=persistent_noise_mask,
            start_zone_mask=mk_start_zone,
            transform=transform,
        )
        param_tag = (f"mk_k{args.kernel_size}_t{args.threshold_sigma_mk}"
                     f"_sl{args.min_slope}_ma{int(args.min_area)}")

    else:
        # --- Canny + watershed detection ---
        print(f"\nRunning CANNY+WATERSHED boundary detection:")
        print(f"  canny_sigma={args.canny_sigma}  low={args.canny_low}  "
              f"high={args.canny_high}")
        print(f"  erosion_sigma={args.erosion_sigma}  "
              f"deposit_sigma={args.deposit_sigma}")
        print(f"  min_area={args.min_area}m²  morph_radius={args.morph_radius}px")
        print(f"  kml_mask={'yes' if kml_path else 'no'}")

        result = detect_boundaries(
            hs_before, hs_after, dem, stn_dhs,
            transform=transform,
            kml_path=kml_path,
            hs_pre_event=hs_pre_event,
            stn_dhs_pre=stn_dhs_pre,
            canny_sigma=args.canny_sigma,
            canny_low=args.canny_low,
            canny_high=args.canny_high,
            erosion_threshold_sigma=args.erosion_sigma,
            deposit_threshold_sigma=args.deposit_sigma,
            min_area_m2=args.min_area,
            morph_radius=args.morph_radius,
            dem_roughness_threshold=args.dem_roughness,
            min_slope_deg=args.min_slope,
            persistent_noise_mask=persistent_noise_mask,
            start_zone_kml=start_zone_kml_path,
        )
        pre_tag = "" if args.no_pre_event else "_prefilter"
        noise_tag = "" if args.no_noise_mask else "_noise"
        param_tag = (f"cs{args.canny_sigma}_lo{args.canny_low}"
                     f"_hi{args.canny_high}_sl{args.min_slope}"
                     f"{noise_tag}_es{args.erosion_sigma}"
                     f"_ma{int(args.min_area)}{pre_tag}")

    print(f"\nResults ({args.method}):")
    print(f"  Release area:  {result['release_area_m2']:.0f} m²")
    print(f"  Deposit area:  {result['deposit_area_m2']:.0f} m²")
    print(f"  Release vol:   {result['release_volume_m3']:.1f} m³")
    print(f"  Deposit vol:   {result['deposit_volume_m3']:.1f} m³")
    print(f"  Release contours: {len(result['release_contours'])}")
    print(f"  Deposit contours: {len(result['deposit_contours'])}")

    out_path = plot_avalanche_boundaries(
        hs_before, hs_after, dem, transform, result,
        period_id=period_id,
        out_dir=out_dir,
        param_tag=param_tag,
        start_zone_mask=result.get('start_zone_mask'),
    )
    print(f"\nPlot saved to: {out_path}")
    
"""
Spatial snow distribution model for distributed SNOWPACK forcing.

Modules:
  - Sx computation (Winstral wind exposure parameter)
  - Terrain feature extraction
  - Transport field computation and smoothing
  - Terrain regression (Random Forest) for transport prediction
  - Hourly gap-filling between surveys
  - Leave-one-out cross-validation
"""

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.ndimage import uniform_filter
from pathlib import Path
from typing import Optional
import pandas as pd
import warnings


# =====================================================================
# Sx computation
# =====================================================================

def compute_sx_vectorized(dem: np.ndarray, azimuth_deg: float,
                          d_max: float = 300.0, resolution: float = 1.0) -> np.ndarray:
    """
    Winstral Sx for a single wind direction (vectorized).

    Sx(i) = max over distance d in [0, d_max]:
        arctan((z_upwind(d) - z(i)) / d)

    Positive Sx = sheltered, Negative Sx = exposed.
    """
    nrows, ncols = dem.shape
    sx = np.full_like(dem, -90.0, dtype=np.float64)
    nodata = np.isnan(dem)

    az_rad = np.radians(azimuth_deg)
    dr = -np.cos(az_rad)
    dc = np.sin(az_rad)

    max_steps = int(d_max / resolution)

    for step in range(1, max_steps + 1):
        ri_off = int(round(step * dr))
        ci_off = int(round(step * dc))

        if ri_off >= 0:
            src_r = slice(0, nrows - ri_off)
            dst_r = slice(ri_off, nrows) if ri_off > 0 else slice(None)
        else:
            src_r = slice(-ri_off, nrows)
            dst_r = slice(0, nrows + ri_off)

        if ci_off >= 0:
            src_c = slice(0, ncols - ci_off)
            dst_c = slice(ci_off, ncols) if ci_off > 0 else slice(None)
        else:
            src_c = slice(-ci_off, ncols)
            dst_c = slice(0, ncols + ci_off)

        upwind_z = np.full_like(dem, np.nan)
        upwind_z[src_r, src_c] = dem[dst_r, dst_c]

        dist = step * resolution
        angle = np.degrees(np.arctan2(upwind_z - dem, dist))

        valid = ~np.isnan(angle) & ~nodata
        sx = np.where(valid & (angle > sx), angle, sx)

    sx[nodata] = np.nan
    return sx


def compute_sx_multi(dem: np.ndarray, azimuths: list = None,
                     d_max: float = 300.0, resolution: float = 1.0,
                     verbose: bool = True) -> dict:
    """Compute Sx for multiple wind directions."""
    if azimuths is None:
        azimuths = np.arange(0, 360, 22.5).tolist()

    sx_dict = {}
    for i, az in enumerate(azimuths):
        if verbose:
            print(f"  Sx {i+1}/{len(azimuths)}: azimuth {az:.1f}°", end="\r")
        sx_dict[az] = compute_sx_vectorized(dem, az, d_max, resolution)
    if verbose:
        print()
    return sx_dict


def compute_wind_weighted_sx(sx_dict: dict, wind_df: pd.DataFrame,
                             speed_col: str = "VW",
                             dir_col: str = "DW") -> np.ndarray:
    """
    Wind-weighted Sx: each directional Sx weighted by wind energy
    (speed² × frequency) from that direction.
    """
    azimuths = sorted(sx_dict.keys())
    bin_width = azimuths[1] - azimuths[0] if len(azimuths) > 1 else 360

    speeds = wind_df[speed_col].values
    directions = wind_df[dir_col].values

    weights = {}
    total_weight = 0
    for az in azimuths:
        half = bin_width / 2
        if az == 0:
            in_bin = (directions >= 360 - half) | (directions < half)
        else:
            in_bin = (directions >= az - half) & (directions < az + half)
        w = np.sum(speeds[in_bin] ** 2)
        weights[az] = w
        total_weight += w

    if total_weight == 0:
        return np.nanmean(list(sx_dict.values()), axis=0)

    shape = list(sx_dict.values())[0].shape
    weighted_sx = np.zeros(shape, dtype=np.float64)
    for az in azimuths:
        sx_az = np.where(np.isnan(sx_dict[az]), 0, sx_dict[az])
        weighted_sx += (weights[az] / total_weight) * sx_az

    mask = np.isnan(list(sx_dict.values())[0])
    weighted_sx[mask] = np.nan
    return weighted_sx


# =====================================================================
# Terrain features
# =====================================================================

def compute_terrain_features(dem: np.ndarray, resolution: float = 1.0) -> dict:
    """
    Terrain features from DEM: slope, aspect (sin/cos), curvatures.
    """
    filled = np.where(np.isnan(dem), np.nanmean(dem), dem)
    dy, dx = np.gradient(filled, resolution)

    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    aspect = np.degrees(np.arctan2(-dx, dy)) % 360

    dyy, dyx = np.gradient(dy, resolution)
    dxy, dxx = np.gradient(dx, resolution)

    p = dx**2 + dy**2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        profile_curv = np.where(p > 1e-10,
            -(dxx * dx**2 + 2 * dxy * dx * dy + dyy * dy**2) / (p * np.sqrt(p + 1)), 0)
        plan_curv = np.where(p > 1e-10,
            -(dxx * dy**2 - 2 * dxy * dx * dy + dyy * dx**2) / (p ** 1.5), 0)

    mask = np.isnan(dem)
    for arr in [slope, profile_curv, plan_curv]:
        arr[mask] = np.nan

    return {
        "slope": slope,
        "aspect_sin": np.sin(np.radians(aspect)),
        "aspect_cos": np.cos(np.radians(aspect)),
        "profile_curvature": np.clip(profile_curv, -2, 2),
        "plan_curvature": np.clip(plan_curv, -2, 2),
        "elevation": dem,
    }


# =====================================================================
# DEM I/O
# =====================================================================

def load_and_resample(src_path: str, ref_path: str,
                      clip_min: float = None,
                      clip_max: float = None) -> np.ndarray:
    """
    Load raster and resample to match a reference grid (resolution, extent, CRS).
    """
    with rasterio.open(ref_path) as ref:
        ref_transform = ref.transform
        ref_shape = (ref.height, ref.width)
        ref_crs = ref.crs

    with rasterio.open(src_path) as src:
        dst = np.full(ref_shape, np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.average,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
        )

    if clip_min is not None:
        dst = np.where(~np.isnan(dst), np.maximum(dst, clip_min), np.nan)
    if clip_max is not None:
        dst = np.where(~np.isnan(dst) & (dst <= clip_max), dst, np.nan)

    return dst


def create_reference_dem(src_path: str, dst_path: str,
                         target_res: float = 1.0) -> None:
    """Resample DEM to target resolution and save as reference grid."""
    with rasterio.open(src_path) as src:
        scale = src.res[0] / target_res
        new_h = int(src.height * scale)
        new_w = int(src.width * scale)

        data = src.read(1, out_shape=(new_h, new_w),
                        resampling=Resampling.average)

        transform = src.transform * src.transform.scale(
            src.width / new_w, src.height / new_h)

        profile = src.profile.copy()
        profile.update(height=new_h, width=new_w,
                       transform=transform, compress='lzw')

        with rasterio.open(dst_path, 'w', **profile) as dst:
            dst.write(data, 1)


# =====================================================================
# Transport computation
# =====================================================================

def compute_transport_field(hs_before: np.ndarray, hs_after: np.ndarray,
                            stn_hs_before: float, stn_hs_after: float,
                            valid_mask: np.ndarray) -> np.ndarray:
    """
    Absolute transport field: cell ΔHS minus station ΔHS.
    Positive = cell gained more than station (deposition).
    Negative = cell gained less (erosion or differential settlement).
    """
    stn_dhs = stn_hs_after - stn_hs_before
    cell_dhs = np.where(valid_mask, hs_after - hs_before, np.nan)
    transport = cell_dhs - stn_dhs
    return transport


def smooth_field(field: np.ndarray, window: int = 15,
                 min_coverage: float = 0.3) -> np.ndarray:
    """Spatial smoothing with NaN handling."""
    fill = np.where(np.isnan(field), 0, field)
    weight = (~np.isnan(field)).astype(float)
    num = uniform_filter(fill, size=window, mode='constant')
    den = uniform_filter(weight, size=window, mode='constant')
    return np.where(den > min_coverage, num / den, np.nan)


# =====================================================================
# Terrain regression for transport prediction
# =====================================================================

def build_feature_array(terrain_features: dict,
                        wind_weighted_sx: np.ndarray,
                        wind_stats: dict,
                        valid_mask: np.ndarray) -> tuple:
    """
    Build feature matrix for regression. Each row = one cell.

    Features:
      - Terrain: slope, aspect_sin, aspect_cos, plan_curv, profile_curv, elev
      - Wind exposure: wind-weighted Sx
      - Period wind stats: mean_speed, dir_sin, dir_cos (broadcast to all cells)

    Returns (X, feature_names).
    """
    idx = np.where(valid_mask.ravel())[0]
    features = []
    names = []

    for name in ['slope', 'aspect_sin', 'aspect_cos',
                 'plan_curvature', 'profile_curvature', 'elevation']:
        arr = terrain_features[name]
        features.append(arr.ravel()[idx])
        names.append(name)

    features.append(wind_weighted_sx.ravel()[idx])
    names.append('wind_weighted_sx')

    # Period-level wind stats (broadcast)
    n = len(idx)
    features.append(np.full(n, wind_stats['mean_speed']))
    names.append('period_mean_wspd')
    features.append(np.full(n, wind_stats['dir_sin']))
    names.append('period_dir_sin')
    features.append(np.full(n, wind_stats['dir_cos']))
    names.append('period_dir_cos')

    X = np.column_stack(features)
    # Replace any remaining NaN with 0
    X = np.nan_to_num(X, nan=0.0)
    return X, names


def compute_wind_stats(wind_df: pd.DataFrame) -> dict:
    """Compute summary wind statistics for a period."""
    if len(wind_df) == 0 or 'VW' not in wind_df.columns:
        return {
            'mean_speed': np.nan, 'dir_sin': np.nan, 'dir_cos': np.nan,
            'mean_dir': np.nan, 'total_wind_energy': np.nan,
            'has_data': False,
        }
    speeds = wind_df['VW'].dropna().values
    dirs = wind_df['DW'].dropna().values
    if len(speeds) == 0 or len(dirs) == 0:
        return {
            'mean_speed': np.nan, 'dir_sin': np.nan, 'dir_cos': np.nan,
            'mean_dir': np.nan, 'total_wind_energy': np.nan,
            'has_data': False,
        }
    u = -speeds * np.sin(np.radians(dirs[:len(speeds)]))
    v = -speeds * np.cos(np.radians(dirs[:len(speeds)]))
    mean_dir = np.degrees(np.arctan2(-np.mean(u), -np.mean(v))) % 360
    return {
        'mean_speed': float(np.mean(speeds)),
        'dir_sin': float(np.sin(np.radians(mean_dir))),
        'dir_cos': float(np.cos(np.radians(mean_dir))),
        'mean_dir': float(mean_dir),
        'total_wind_energy': float(np.sum(speeds ** 2)),
        'has_data': True,
    }


def train_transport_model(X_train: np.ndarray, y_train: np.ndarray,
                          n_estimators: int = 100,
                          max_depth: int = 15,
                          max_samples: float = 0.5):
    """
    Train Random Forest to predict transport from terrain + wind features.
    """
    from sklearn.ensemble import RandomForestRegressor

    # Subsample if dataset is very large
    n = len(y_train)
    if n > 500_000:
        idx = np.random.choice(n, 500_000, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_samples=max_samples,
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    return rf


def predict_transport(model, terrain_features: dict,
                      wind_weighted_sx: np.ndarray,
                      wind_stats: dict,
                      valid_mask: np.ndarray) -> np.ndarray:
    """Predict transport field using trained model."""
    X, _ = build_feature_array(terrain_features, wind_weighted_sx,
                               wind_stats, valid_mask)
    y_pred = model.predict(X)

    transport = np.full(valid_mask.shape, np.nan)
    transport.ravel()[np.where(valid_mask.ravel())[0]] = y_pred
    return transport


# =====================================================================
# Gap-filling
# =====================================================================

def gap_fill_period(hs_start: np.ndarray,
                    transport_field: np.ndarray,
                    stn_hs_series: pd.Series,
                    wind_speed_series: pd.Series,
                    valid_mask: np.ndarray) -> dict:
    """
    Generate hourly HS grids for one inter-survey period.

    Each hour:
      - Background: station hourly ΔHS (uniform)
      - Transport: distributed proportional to hourly wind_speed²

    The total transport sums to the observed field by construction.
    """
    ws2 = wind_speed_series.values ** 2
    total_we = np.sum(ws2)

    stn_dhs = stn_hs_series.diff().fillna(0)
    transport_clean = np.where(np.isnan(transport_field), 0, transport_field)

    cumulative = hs_start.copy()
    grids = {}

    for idx, (ts, dhs_stn) in enumerate(stn_dhs.items()):
        if idx == 0:
            grids[ts] = cumulative.copy()
            continue

        frac = ws2[idx] / total_we if total_we > 0 else 0
        cell_dhs = dhs_stn + transport_clean * frac

        cumulative = np.where(valid_mask,
                              np.maximum(cumulative + cell_dhs, 0),
                              cumulative + dhs_stn)
        cumulative = np.maximum(cumulative, 0)
        grids[ts] = cumulative.copy()

    return grids


def validate_prediction(predicted: np.ndarray, observed: np.ndarray,
                        valid_mask: np.ndarray) -> dict:
    """Compute validation metrics."""
    mask = valid_mask & ~np.isnan(predicted) & ~np.isnan(observed)
    if mask.sum() == 0:
        return {'rmse': np.nan, 'bias': np.nan, 'mae': np.nan, 'r': np.nan, 'n': 0}

    p = predicted[mask]
    o = observed[mask]
    resid = o - p
    return {
        'rmse': np.sqrt(np.mean(resid**2)),
        'bias': np.mean(resid),
        'mae': np.mean(np.abs(resid)),
        'r': np.corrcoef(p, o)[0, 1] if len(p) > 1 else np.nan,
        'n': mask.sum(),
    }


# =====================================================================
# Transport model abstraction (for future physics-based integration)
# See TRANSPORT_MODELS.md for design notes.
# =====================================================================

class TransportModel:
    """
    Abstract interface for snow transport computation.

    Subclass this to swap between empirical and physics-based transport.
    The pipeline calls compute_period_transport() to get the total transport
    field for a survey interval, and compute_hourly_transport() for the
    temporal disaggregation within gap_fill_period().
    """

    def compute_period_transport(self, dem: np.ndarray,
                                 hs_start: np.ndarray,
                                 wind_df: pd.DataFrame,
                                 valid_mask: np.ndarray) -> np.ndarray:
        """
        Predict total transport [meters] over an entire inter-survey period.

        Parameters
        ----------
        dem : bare ground DEM (meters)
        hs_start : snow depth at start of period (meters)
        wind_df : hourly wind data for the period (VW in m/s, DW in degrees)
        valid_mask : boolean mask of cells to compute

        Returns
        -------
        2D array of net transport [meters]. Positive = deposition, negative = erosion.
        """
        raise NotImplementedError

    def compute_hourly_transport(self, dem: np.ndarray,
                                  hs_current: np.ndarray,
                                  wind_speed: float,
                                  wind_dir: float,
                                  temperature: float,
                                  valid_mask: np.ndarray) -> np.ndarray:
        """
        Compute transport for a single hourly timestep.

        For the empirical model, this is just wind²-weighted fraction of total.
        For a physics model, this would compute actual saltation/suspension fluxes.

        Returns
        -------
        2D array of hourly transport [meters/hour].
        """
        raise NotImplementedError


class EmpiricalTransportModel(TransportModel):
    """
    Current approach: learn transport from survey pairs + RF regression.
    See gap_fill_period() for the hourly disaggregation.
    """

    def __init__(self, rf_model=None, transport_field: np.ndarray = None):
        self.rf_model = rf_model
        self.transport_field = transport_field

    def compute_period_transport(self, dem, hs_start, wind_df, valid_mask):
        if self.rf_model is not None:
            snow_surface = np.where(~np.isnan(dem) & ~np.isnan(hs_start),
                                    dem + hs_start,
                                    np.where(np.isnan(dem), np.nanmean(dem), dem))
            terrain = compute_terrain_features(snow_surface)
            sx_dict = compute_sx_multi(snow_surface, verbose=False)
            wwsx = compute_wind_weighted_sx(sx_dict, wind_df)
            wstats = compute_wind_stats(wind_df)
            return predict_transport(self.rf_model, terrain, wwsx, wstats, valid_mask)
        elif self.transport_field is not None:
            return self.transport_field
        else:
            raise ValueError("Need either rf_model or transport_field")


class WindNinjaTransportModel(TransportModel):
    """
    Placeholder for WindNinja-based physics transport.

    Integration plan:
    1. Precompute WindNinja wind fields for 16 directions × 3 speeds
       → wind_library[azimuth][speed_class] = 2D wind speed array
    2. At each timestep, look up + interpolate the distributed wind field
    3. Apply Pomeroy (1993) equilibrium saltation transport:
       Q_salt = C * rho / (u* * g) * u_t * (u*² - u_t²)
       where u_t is threshold friction velocity
    4. Compute suspension flux and sublimation loss
    5. Divergence of flux → deposition/erosion at each cell

    Required inputs beyond current pipeline:
    - WindNinja installation (CLI or Python binding)
    - Snow surface properties for threshold velocity (age, density, temperature)
    - Atmospheric temperature + humidity for sublimation computation
    """

    def __init__(self, wind_library_dir: str = None,
                 threshold_wind_speed: float = 5.0):
        self.wind_library_dir = wind_library_dir
        self.u_threshold = threshold_wind_speed
        raise NotImplementedError(
            "WindNinja transport not yet implemented. "
            "See TRANSPORT_MODELS.md for integration plan."
        )

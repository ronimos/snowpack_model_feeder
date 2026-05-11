"""
snowpack_analysis.py — Core SNOWPACK analysis functions.

Provides:
    geojson_to_mask()            Rasterize a GeoJSON polygon to a boolean mask
    assign_cluster_groups()      Assign clusters to release / adjacent / reference groups
    split_wl_slab()              Identify WL / slab boundary via grain type
    profile_features()           Per-cluster WL/slab/stability features at one timestep
    extract_group_timeseries()   Time series of features for a cluster group
    compute_meloche_features()   Meloche et al. (2025) crack arrest parameters
    reduce_scalars_at_time()     Per-cluster scalar reduction for visualization

No CLI. Import from analysis_pipeline.py, visualize_snowpack.py,
and analyze_release_zone.py.
"""

from __future__ import annotations

import json
import warnings
import rasterio
import rasterio.features
import rasterio.transform as rt
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

# SNOWPACK grain type codes (first digit = grain class)
WL_TYPES   = set(range(400, 500)) | set(range(500, 600))  # FC (4xx), DH (5xx)
SLAB_TYPES = set(range(200, 300)) | set(range(300, 400))  # DF (2xx), RG (3xx)

EVENT_DATE = pd.Timestamp("2026-01-18")

GROUP_COLORS = {
    'release':   '#d32f2f',
    'adjacent':  '#1976d2',
    'reference': '#666666',
}
GROUP_LABELS = {
    'release':   'Release zone',
    'adjacent':  'Adjacent slope (start zone)',
    'reference': 'Reference (terrain-matched)',
}


# -----------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------

def _reproject_lonlat_to_utm(pts_lonlat, epsg: int = 32613):
    """Reproject list of (lon, lat) points to UTM."""
    from pyproj import Transformer
    t = Transformer.from_crs('EPSG:4326', f'EPSG:{epsg}', always_xy=True)
    return [t.transform(x, y) for x, y in pts_lonlat]


def geojson_to_mask(path, dem_shape, transform) -> np.ndarray:
    from shapely.geometry import shape
    from shapely.ops import unary_union

    with open(str(path)) as f:
        gj = json.load(f)
    polys = [shape(feat['geometry']) for feat in gj['features']]
    merged = unary_union(polys)

    def reproj(poly):
        from shapely.geometry import Polygon, MultiPolygon
        def ring(r): return _reproject_lonlat_to_utm(list(r.coords))
        if poly.geom_type == 'Polygon':
            return Polygon(ring(poly.exterior),
                           [ring(i) for i in poly.interiors])
        return MultiPolygon([reproj(p) for p in poly.geoms])

    merged_utm = reproj(merged)
    aff = transform if hasattr(transform, 'c') else rt.Affine(*transform[:6])
    mask = rasterio.features.geometry_mask(
        [merged_utm.__geo_interface__], out_shape=dem_shape,
        transform=aff, invert=True)
    print(f"  Release zone mask:  {mask.sum()} cells")
    return mask


# -----------------------------------------------------------------------
# Cluster group assignment
# -----------------------------------------------------------------------

def assign_cluster_groups(cluster_map, release_mask, start_zone_mask,
                           dem, domain_mask) -> dict:
    """
    Assign clusters to groups. Reference group is terrain-matched to
    release zone (similar elevation and slope angle).
    """
    fill_dem = np.where(np.isnan(dem), np.nanmean(dem), dem)
    dy, dx   = np.gradient(fill_dem, 1.0)
    slope    = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

    # Release zone terrain statistics for matching
    rel_elev_mean  = float(np.nanmean(dem[release_mask & domain_mask]))
    rel_elev_std   = float(np.nanstd(dem[release_mask & domain_mask]))
    rel_slope_mean = float(np.nanmean(slope[release_mask & domain_mask]))

    groups = {'release': set(), 'adjacent': set(), 'reference': set()}

    for cid in np.unique(cluster_map[domain_mask]):
        if cid <= 0:
            continue
        cells = cluster_map == cid
        n     = cells.sum()
        if n == 0:
            continue

        frac_rel   = (cells & release_mask).sum() / n
        frac_start = (cells & start_zone_mask).sum() / n

        if frac_rel >= 0.3:
            groups['release'].add(cid)
        elif frac_start >= 0.3:
            groups['adjacent'].add(cid)
        else:
            # Terrain-match: within 1 std of release zone elevation
            c_elev  = float(np.nanmean(dem[cells]))
            c_slope = float(np.nanmean(slope[cells]))
            if (abs(c_elev - rel_elev_mean) < rel_elev_std * 1.5 and
                    abs(c_slope - rel_slope_mean) < 10.0):
                groups['reference'].add(cid)

    for g, ids in groups.items():
        print(f"  {GROUP_LABELS[g]}: {len(ids)} clusters")
    return groups


# -----------------------------------------------------------------------
# WL / slab boundary detection
# -----------------------------------------------------------------------

def split_wl_slab(grain_type, z):
    """
    Find the BASAL weak layer by scanning upward from the bottom.

    The persistent weak layer (early-season facets/depth hoar) sits at
    the base. Near-surface facets are NOT the target WL.

    Strategy: scan from the bottom upward through FC/DH layers (4xx/5xx),
    stop at the FIRST transition to slab types (code < 400).
    That initial FC/DH block = basal WL. Everything above = slab.

    Returns (slab_mask, wl_mask, interface_z) or (None, None, None).
    """
    ok = ~np.isnan(grain_type) & ~np.isnan(z) & (z != 0) & (grain_type > 0)
    if ok.sum() < 3:
        return None, None, None

    gt_ok = np.round(grain_type[ok]).astype(int)
    z_ok  = z[ok]

    # Sort bottom-to-top: z is negative (depth below surface), most-negative = deepest
    order        = np.argsort(z_ok)
    gt_sorted    = gt_ok[order]
    z_sorted     = z_ok[order]
    # Only FC (4xx) and DH (5xx) are weak layer grain types
    is_wl_sorted = (gt_sorted // 100 == 4) | (gt_sorted // 100 == 5)

    # Must start with at least one WL layer at the base
    if not is_wl_sorted[0]:
        return None, None, None

    # Walk upward: find top of the first contiguous basal WL block
    wl_top_sorted = 0
    for i in range(len(is_wl_sorted)):
        if is_wl_sorted[i]:
            wl_top_sorted = i
        else:
            break   # first non-WL layer = interface

    interface_z = float(z_sorted[wl_top_sorted])

    # Map back to original array indices
    ok_indices = np.where(ok)[0]
    orig_order = ok_indices[order]

    slab_mask = np.zeros(len(grain_type), dtype=bool)
    wl_mask   = np.zeros(len(grain_type), dtype=bool)
    for sorted_i, orig_i in enumerate(orig_order):
        if sorted_i <= wl_top_sorted:
            wl_mask[orig_i] = True
        else:
            slab_mask[orig_i] = True

    return slab_mask, wl_mask, interface_z


# -----------------------------------------------------------------------
# Per-cluster feature extraction
# -----------------------------------------------------------------------

def profile_features(ds_t_loc, min_depth_cm: float) -> dict:
    """
    Extract slab and WL features for one cluster at one timestep.

    Stability indices: min value within ±5cm of WL/slab interface.
    Slab properties:   mean over slab layers above interface.
    WL properties:     mean over WL layers.
    """
    z  = ds_t_loc['z'].values.ravel()
    gt = ds_t_loc['grain_type'].values.ravel()
    hs_arr = ds_t_loc['HS'].values.ravel()
    hs = float(np.nanmean(hs_arr))

    ok = ~np.isnan(z) & ~np.isnan(gt) & (z != 0) & (gt > 0)
    if ok.sum() < 2:
        return {}

    result = {'hs': hs}
    nu = 0.3   # Poisson's ratio (fixed, Gaume et al. 2018)

    slab_m, wl_m, interface_z = split_wl_slab(gt, z)

    if slab_m is None:
        return result

    # --- Stability at interface (±5cm window, z is in cm) ---
    near_interface = ok & (np.abs(z - interface_z) <= 5.0) # TODO # Fix — ±5 cm 5.0
    for var in ('sk38', 'ssi', 'sn38', 'stab_deformation_rate'):
        try:
            v = ds_t_loc[var].values.ravel()
            vals = v[near_interface & ~np.isnan(v)]
            result[f'min_{var}'] = float(np.nanmin(vals))                 if len(vals) else np.nan
        except Exception:
            result[f'min_{var}'] = np.nan

    # --- Slab properties ---
    if slab_m.any():
        for var, agg in [('density',       np.nanmean),
                         ('hand_hardness', np.nanmean),
                         ('grain_size',    np.nanmean)]:
            try:
                v = ds_t_loc[var].values.ravel()
                vals = v[slab_m & ~np.isnan(v)]
                result[f'slab_{var}'] = float(agg(vals)) if len(vals) else np.nan
            except Exception:
                result[f'slab_{var}'] = np.nan

        # Slab thickness: distance from interface to snow surface
        slab_z = z[slab_m & ok]
        # interface_z is in cm (negative from surface), convert to metres
        result['slab_thickness'] = float(-interface_z / 100.0)                   if len(slab_z) else np.nan

        # Most common grain type in slab (first digit = grain class)
        gt_slab = np.round(gt[slab_m & ok]).astype(int)
        gt_slab = gt_slab[gt_slab > 0]
        if len(gt_slab):
            classes, counts = np.unique(gt_slab // 100, return_counts=True)
            result['slab_dominant_grain_class'] = int(classes[np.argmax(counts)])
        else:
            result['slab_dominant_grain_class'] = np.nan

        # --- Crust/ice layer detection (MF=7xx, IF=8xx) ---
        gt_slab_all = np.round(gt[slab_m & ok]).astype(int)
        z_slab_all  = z[slab_m & ok]
        is_crust    = (gt_slab_all // 100) >= 7
        result['has_crust']      = bool(is_crust.any())
        result['n_crust_layers'] = int(is_crust.sum())
        result['crust_thickness'] = float(
            (z_slab_all[is_crust].max() - z_slab_all[is_crust].min()) / 100.0
        ) if is_crust.any() else 0.0
        # Crust immediately above WL (<10cm of interface) — bridging layer
        near_iface = np.abs(z_slab_all - interface_z) <= 10.0
        result['crust_at_interface'] = bool((is_crust & near_iface).any())
        # Depth of shallowest crust (m below surface)
        result['crust_top_depth'] = float(
            -z_slab_all[is_crust].max() / 100.0
        ) if is_crust.any() else np.nan
    else:
        for k in ('slab_density', 'slab_hand_hardness', 'slab_grain_size',
                  'slab_thickness', 'slab_dominant_grain_class',
                  'has_crust', 'n_crust_layers', 'crust_thickness',
                  'crust_at_interface', 'crust_top_depth'):
            result[k] = np.nan

    # --- WL properties ---
    if wl_m.any():
        for var, agg in [('shear_strength', np.nanmean),
                         ('grain_size',     np.nanmean),
                         ('density',        np.nanmean),
                         ('hand_hardness',  np.nanmean)]:
            try:
                v = ds_t_loc[var].values.ravel()
                vals = v[wl_m & ~np.isnan(v)]
                result[f'wl_{var}'] = float(agg(vals)) if len(vals) else np.nan
            except Exception:
                result[f'wl_{var}'] = np.nan

        wl_z = z[wl_m & ok]
        result['wl_burial_depth'] = float(-wl_z.max() / 100.0)          if len(wl_z) else np.nan
        result['wl_thickness']    = float(wl_z.max() - wl_z.min())      if len(wl_z) else np.nan
        result['interface_z']     = float(interface_z)
    else:
        for k in ('wl_shear_strength', 'wl_grain_size', 'wl_density',
                  'wl_hand_hardness', 'wl_burial_depth', 'wl_thickness',
                  'interface_z'):
            result[k] = np.nan

    # --- Meloche et al. (2025) parameters (per-cluster, no neighbors needed) ---
    rho   = result.get('slab_density',      np.nan)
    h_m   = result.get('slab_thickness',    np.nan)   # m
    D_wl  = result.get('wl_thickness',      np.nan)   # cm (from z)
    tau_p = result.get('wl_shear_strength', np.nan)   # Pa

    if all(not np.isnan(v) for v in [rho, h_m, D_wl, tau_p]) and D_wl > 0:
        D_wl_m  = D_wl / 100.0                           # cm -> m
        # Slab elastic modulus — power law fit to van Herwijnen et al. (2016) range
        # ~2 MPa at ρ=200, ~4 MPa at ρ=300, ~6 MPa at ρ=350 kg/m³
        E_slab  = (rho / 300.0)**2.5 * 4.0e6             # Pa
        # Slab tensile strength — ~5 kPa at ρ=300 kg/m³ (Meloche 2-10 kPa range)
        sigma_t = (rho / 300.0)**1.4 * 5.0e3             # Pa
        G_wl    = 0.2e6                                   # Pa (Reiweger et al. 2010)
        E_prime = E_slab / (1.0 - nu**2)                 # apparent Young's modulus
        K_wl    = G_wl / D_wl_m                          # WL stiffness (Pa/m)
        Lambda  = float(np.sqrt(E_prime * h_m / K_wl))  # characteristic elastic length (m)
        result['E_slab']  = E_slab
        result['sigma_t'] = sigma_t
        result['Lambda']  = Lambda
        result['K_wl']    = K_wl
    else:
        for k in ('E_slab', 'sigma_t', 'Lambda', 'K_wl'):
            result[k] = np.nan
    return result


# -----------------------------------------------------------------------
# Group time series extraction
# -----------------------------------------------------------------------

def extract_group_timeseries(ds, group_ids: set,
                              location_names,
                              min_depth_cm: float,
                              end_ts: pd.Timestamp,
                              per_cluster: bool = False) -> pd.DataFrame:
    """
    Extract feature time series for a group of clusters.

    per_cluster=False (default): returns group-median DataFrame indexed by time.
      Used for plotting.

    per_cluster=True: returns one row per (cluster_id, time), with columns
      for all features. Used for the classifier — each cluster is a sample.
    """
    if not group_ids:
        return pd.DataFrame()

    loc_mask = np.array([
        int(str(loc).split('_')[-1]) in group_ids
        for loc in location_names])
    if not loc_mask.any():
        return pd.DataFrame()

    ds_group   = ds.isel(location=loc_mask)
    loc_ids    = [int(str(loc).split('_')[-1])
                  for loc in location_names[loc_mask]]
    times      = pd.DatetimeIndex(ds_group.coords['time'].values)
    noon_ts    = [t for t in times if t.hour == 12 and t <= end_ts]

    all_rows = []
    for ts in noon_ts:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ds_t = ds_group.sel(time=ts, method='nearest').compute()

        n_locs = ds_t.sizes.get('location', 1)
        ts_rows = []
        for i in range(n_locs):
            try:
                ds_loc = ds_t.isel(location=i) \
                    if 'location' in ds_t.dims else ds_t
                feat = profile_features(ds_loc, min_depth_cm)
                if feat:
                    feat['time']       = ts
                    feat['cluster_id'] = loc_ids[i] if i < len(loc_ids) else -1
                    ts_rows.append(feat)
            except Exception:
                continue

        if not ts_rows:
            continue

        if per_cluster:
            all_rows.extend(ts_rows)
        else:
            # Group median for plotting
            keys = [k for k in ts_rows[0]
                    if k not in ('time', 'cluster_id')]
            row = {'time': ts}
            for k in keys:
                vals = [f[k] for f in ts_rows
                        if k in f and f[k] is not None
                        and not np.isnan(f[k])]
                row[k] = float(np.nanmedian(vals)) if vals else np.nan
            all_rows.append(row)

    if not all_rows:
        return pd.DataFrame()

    if per_cluster:
        df = pd.DataFrame(all_rows)
        df['time'] = pd.to_datetime(df['time'])
        return df.set_index(['time', 'cluster_id']).sort_index()
    return pd.DataFrame(all_rows).set_index('time').sort_index()


# -----------------------------------------------------------------------
# Meloche et al. (2025) crack arrest parameters
# -----------------------------------------------------------------------

def compute_meloche_features(snap_data: dict, cluster_map: np.ndarray,
                              dem: np.ndarray, transform,
                              snap_ts) -> pd.DataFrame:
    """
    Compute Meloche et al. (2025) spatial features requiring neighbor information.

    For each cluster computes:
      τ_g  — gravitational shear stress (Pa)
      θ    — WL shear strength gradient to k nearest neighbors (Pa/m)
      Π₁   — elastic dimensionless number: τ_g / (θ·Λ·√(1+δ))
      Π₂   — brittle dimensionless number: Π₁ · √(σ_t/τ_g)
      A_ca — crack arrest length estimate from brittle scaling (m)
      L_t  — quasi-static tensile length (m)

    Parameters match Meloche et al. (2025) Table 1:
      ψ  = slope angle per cluster (from DEM)
      ϕ  = 27° (snow friction angle, fixed)
      δ  = 1   (softening coefficient, fixed — slope scale default)
      L_ss = 20 m (steady-state length, fixed — paper default)

    Returns DataFrame indexed by cluster_id with Meloche features.
    """
    PHI_DEG = 27.0          # snow friction angle (degrees)
    DELTA   = 1.0           # softening coefficient
    L_SS    = 20.0          # steady-state length (m)
    K_NEIGHBORS = 6         # nearest neighbors for θ computation
    G_GRAV  = 9.81          # m/s²

    phi_rad = np.radians(PHI_DEG)

    # Compute slope angle per cluster from DEM
    fill_dem = np.where(np.isnan(dem), np.nanmean(dem), dem)
    dy, dx   = np.gradient(fill_dem, 1.0)
    slope    = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

    # Compute cluster centroids in pixel space
    cluster_ids = np.unique(cluster_map[~np.isnan(dem)])
    cluster_ids = cluster_ids[cluster_ids > 0]

    centroids = {}  # cluster_id -> (row_mean, col_mean)
    slopes_cl = {}  # cluster_id -> mean slope angle (deg)
    for cid in cluster_ids:
        mask = cluster_map == cid
        rows, cols = np.where(mask)
        if len(rows):
            centroids[cid]  = (float(rows.mean()), float(cols.mean()))
            slopes_cl[cid]  = float(slope[mask].mean())

    # Collect per-cluster τ_p, Λ, E, σ_t from snap_data
    all_rows = []
    for grp, df in snap_data.items():
        if df.empty:
            continue
        df2 = df.copy()
        df2['group'] = grp
        all_rows.append(df2)

    if not all_rows:
        return pd.DataFrame()

    features = pd.concat(all_rows)
    if 'wl_shear_strength' not in features.columns:
        return pd.DataFrame()

    # Build array of centroid positions for distance computation
    cids_arr   = np.array([cid for cid in features.index
                           if cid in centroids])
    if len(cids_arr) < 2:
        return pd.DataFrame()

    cents = np.array([centroids[c] for c in cids_arr])  # (N, 2) pixel coords
    # Convert pixel distance to meters (1m resolution DEM)
    # cents are in pixels, pixel size = 1m
    def _scalar(df, cid, col):
        """Safely extract a scalar from a possibly multi-row index."""
        if cid not in df.index:
            return np.nan
        val = df.loc[cid, col]
        if isinstance(val, pd.Series):
            return float(val.iloc[0])
        return float(val)

    # wl_shear_strength from SNOWPACK is in kPa; Meloche uses Pa
    tau_p_arr = np.array([_scalar(features, c, 'wl_shear_strength')
                          for c in cids_arr]) * 1000.0  # kPa -> Pa

    # KNN for θ computation
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=min(K_NEIGHBORS+1, len(cids_arr)),
                             algorithm='ball_tree').fit(cents)
    distances, indices = nbrs.kneighbors(cents)

    rows_out = []
    for i, cid in enumerate(cids_arr):
        if cid not in features.index:
            continue
        loc_data = features.loc[cid]
        row = (loc_data.iloc[0].copy()
               if isinstance(loc_data, pd.DataFrame)
               else loc_data.copy())
        tau_p = row.get('wl_shear_strength', np.nan) * 1000.0  # kPa -> Pa
        Lambda = row.get('Lambda', np.nan)
        E_slab = row.get('E_slab', np.nan)
        sigma_t = row.get('sigma_t', np.nan)
        rho   = row.get('slab_density', np.nan)
        h_m   = row.get('slab_thickness', np.nan)
        psi_deg = slopes_cl.get(cid, np.nan)

        if any(np.isnan(v) for v in [tau_p, Lambda, rho, h_m, psi_deg]):
            rows_out.append({'cluster_id': cid})
            continue

        psi_rad = np.radians(psi_deg)
        sin_psi = np.sin(psi_rad)
        if sin_psi < 0.01:
            rows_out.append({'cluster_id': cid})
            continue

        # Gravitational shear stress (driving stress)
        tau_g = rho * G_GRAV * h_m * sin_psi * (1.0 - np.tan(phi_rad)/np.tan(psi_rad))
        tau_g = max(tau_g, 0.1)   # avoid negative/zero on flat terrain

        # Scaling law breaks down when driving stress approaches zero
        # (slope near friction angle). Flag these clusters explicitly.
        tau_g_min_valid = 50.0   # Pa — empirical minimum for D2+ events
        if tau_g < tau_g_min_valid:
            rows_out.append({'cluster_id': cid, 'tau_g': tau_g,
                             'slope_angle': psi_deg,
                             'note': 'tau_g below valid range for scaling law'})
            continue

        # θ — mean WL shear strength gradient to k nearest neighbors
        neighbor_idx = indices[i][1:]   # exclude self
        neighbor_dist = distances[i][1:]
        theta_vals = []
        for j, d in zip(neighbor_idx, neighbor_dist):
            if d < 1e-6:
                continue
            tau_j = tau_p_arr[j]
            if not np.isnan(tau_j):
                theta_vals.append(abs(tau_p - tau_j) / d)

        theta = float(np.mean(theta_vals)) if theta_vals else np.nan

        if np.isnan(theta) or theta < 1e-6:
            rows_out.append({'cluster_id': cid, 'tau_g': tau_g, 'theta': theta,
                             'slope_angle': psi_deg})
            continue

        # Dimensionless numbers (Eqs. 19, 20)
        denom   = theta * Lambda * np.sqrt(1.0 + DELTA)
        Pi1     = tau_g / denom                           # elastic number
        Pi2     = Pi1 * np.sqrt(sigma_t / tau_g)         # brittle number

        # Quasi-static tensile length (normalisation for brittle Aca)
        L_t     = sigma_t / (rho * G_GRAV * sin_psi *
                              (1.0 - np.tan(phi_rad)/np.tan(psi_rad)))

        # Crack arrest length estimates (proportionality from Eqs. 19, 20)
        # Proportionality constant ≈ 1 from Figure 7/10 (1:1 line in log scale)
        A_ca_elastic = L_SS * Pi1**(1.5)
        A_ca_brittle = L_t  * Pi1 * np.sqrt(sigma_t / tau_g)

        rows_out.append({
            'cluster_id':    cid,
            'slope_angle':   psi_deg,
            'tau_g':         tau_g,
            'theta':         theta,
            'Pi1_elastic':   Pi1,
            'Pi2_brittle':   Pi2,
            'Lambda':        Lambda,
            'L_t':           L_t,
            'A_ca_elastic':  A_ca_elastic,
            'A_ca_brittle':  A_ca_brittle,
        })

    if not rows_out:
        return pd.DataFrame()
    return pd.DataFrame(rows_out).set_index('cluster_id')


# -----------------------------------------------------------------------
# Per-frame scalar reduction (for visualization)
# -----------------------------------------------------------------------

def reduce_scalars_at_time(ds: xr.Dataset,
                   timestamp: pd.Timestamp,
                   prev_hs: dict,
                   min_depth_cm: float,
                   max_depth_cm: float,
                   wl_method: str = 'simple') -> dict:
    """
    Reduce all variables to per-cluster scalars at one timestep.

    prev_hs: dict mapping location -> HS value at previous timestep,
             used to compute dHS/dt.
    """
    ds_t = ds.sel(time=timestamp, method='nearest')

    z   = ds_t['z']
    in_depth = (z <= -min_depth_cm) & (z >= -max_depth_cm) & (~np.isnan(z))

    def layer_min(var):
        return (ds_t[var].where(in_depth)
                         .min(dim='layer')
                         .squeeze([d for d in ['slope','realization'] if d in ds_t.dims])
                         .compute().values)

    def layer_mean(var):
        return (ds_t[var].where(in_depth)
                         .mean(dim='layer')
                         .squeeze([d for d in ['slope','realization'] if d in ds_t.dims])
                         .compute().values)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        hs       = ds_t['HS'].squeeze([d for d in ['slope','realization'] if d in ds_t.dims]).compute().values
        sk38_min = layer_min('sk38')
        ssi_min  = layer_min('ssi')
        sn38_min = layer_min('sn38')
        tg_min   = layer_min('temperature_gradient')
        atg_mean = layer_mean('accumulated_temperature_gradient')
        sdr_min  = layer_min('stab_deformation_rate')

    # dHS/dt in cm/day from previous timestep
    locs = ds.coords['location'].values
    if prev_hs:
        dt_days = (timestamp - prev_hs['time']).total_seconds() / 86400
        if dt_days > 0:
            dhs_dt = (hs - prev_hs['hs']) / dt_days
        else:
            dhs_dt = np.zeros_like(hs)
    else:
        dhs_dt = np.zeros_like(hs)

    # --- Structure variables (WL/slab properties) ---
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sq_dims   = [d for d in ['slope','realization'] if d in ds_t.dims]
        hs_vals   = ds_t['HS'].squeeze(sq_dims).compute().values
        z_da      = ds_t['z'].squeeze(sq_dims).compute()   # (location, layer)

        if wl_method == 'simple':
            # Fast: bottom 20% of HS = WL proxy, top 80% = slab
            hs_bc     = xr.DataArray(hs_vals, dims=['location'])
            wl_zone   = (z_da < -0.8 * hs_bc) & ~np.isnan(z_da)
            slab_zone = (z_da > -0.8 * hs_bc) & (z_da < 0) & ~np.isnan(z_da)
            wl_burial_raw  = hs_vals * 0.8 / 100.0
            slab_thick_raw = hs_vals * 0.8 / 100.0
        else:
            # Proper: FC/DH grain-type detection per cluster
            import sys as _sys
            from pathlib import Path as _Path
            _sys.path.insert(0, str(_Path(__file__).resolve().parent))
            from analyze_release_zone import split_wl_slab
            z_np  = z_da.values          # (location, layer)
            gt_np = ds_t['grain_type'].squeeze(sq_dims).compute().values
            n_loc = z_np.shape[0]
            wl_mask_np   = np.zeros(z_np.shape, dtype=bool)
            slab_mask_np = np.zeros(z_np.shape, dtype=bool)
            wl_burial_raw  = np.full(n_loc, np.nan)
            slab_thick_raw = np.full(n_loc, np.nan)
            for li in range(n_loc):
                sm, wm, iz = split_wl_slab(gt_np[li], z_np[li])
                if sm is not None:
                    wl_mask_np[li]   = wm
                    slab_mask_np[li] = sm
                    wl_burial_raw[li]  = -iz / 100.0      # cm -> m
                    slab_thick_raw[li] = -iz / 100.0
            wl_zone   = xr.DataArray(wl_mask_np,   dims=z_da.dims)
            slab_zone = xr.DataArray(slab_mask_np, dims=z_da.dims)

        wl_str_raw = (ds_t['shear_strength'].squeeze(sq_dims).where(wl_zone)
                      .mean(dim='layer').compute().values)
        wl_grain_raw = (ds_t['grain_size'].squeeze(sq_dims).where(wl_zone)
                        .mean(dim='layer').compute().values)
        slab_dens_raw = (ds_t['density'].squeeze(sq_dims).where(slab_zone)
                         .mean(dim='layer').compute().values)

        # Meloche et al. (2025) per-cluster parameters
        # E and σ_t from slab density (power laws, van Herwijnen 2016)
        E_slab_raw   = (slab_dens_raw / 300.0)**2.5 * 4.0    # MPa
        sigma_t_raw  = (slab_dens_raw / 300.0)**1.4 * 5.0    # kPa
        # Characteristic elastic length Λ = sqrt(E'·h·D_wl / G_wl)
        G_wl_pa      = 0.2e6                                   # Pa
        nu           = 0.3
        E_prime_raw  = E_slab_raw * 1e6 / (1 - nu**2)         # Pa
        D_wl_m       = np.where(slab_thick_raw > 0,
                                slab_thick_raw * 0.2, np.nan)  # 20% of slab as WL
        h_m          = slab_thick_raw
        Lambda_raw   = np.where(
            (h_m > 0) & (D_wl_m > 0),
            np.sqrt(E_prime_raw * h_m * D_wl_m / G_wl_pa), np.nan)
        # Gravitational driving stress τ_g (Pa)
        # Use fixed slope ψ=32° (mean release zone slope)
        psi, phi = np.radians(32), np.radians(27)
        tau_g_raw = (slab_dens_raw * 9.81 * h_m *
                     np.sin(psi) * (1 - np.tan(phi)/np.tan(psi)))

    return {
        'HS':          hs,
        'dhs_dt':      dhs_dt,
        'sk38_min':    sk38_min,
        'ssi_min':     ssi_min,
        'sn38_min':    sn38_min,
        'tg_min':      np.abs(tg_min),
        'atg_mean':    atg_mean,
        'sdr_min':     sdr_min,
        'wl_strength':  wl_str_raw,
        'wl_grain':     wl_grain_raw,
        'wl_burial':    wl_burial_raw,
        'slab_thick':   slab_thick_raw,
        'slab_density': slab_dens_raw,
        'E_slab':       E_slab_raw,
        'sigma_t':      sigma_t_raw,
        'Lambda':       Lambda_raw,
        'tau_g':        tau_g_raw,
    }, {'hs': hs, 'time': timestamp}
    
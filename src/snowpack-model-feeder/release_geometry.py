"""
release_geometry.py — Physically-motivated avalanche release polygon generation.

Provides:
    find_stauchwall()          Walk downslope from trigger until slope < threshold
    estimate_cross_slope_width() Cross-slope arrest width from Gaume (2015) / θ
    project_along_aspect()     Project a UTM point distance along slope aspect
    make_release_polygon_2d()  Full 2D release polygon from four constraints:
                                 1. Meloche A_ca (upslope)
                                 2. ~28° slope threshold (downslope / stauchwall)
                                 3. θ cross-slope gradient (flanks)
                                 4. Start zone KML (hard lateral boundary)
    rasterize_release_polygon() Burn polygon to a depth raster on the DEM grid
    depth_from_snowpack()       Per-pixel slab depth (m) from cluster HS

No CLI. Import from analysis_pipeline.py and scenario_writer.py.

References
----------
Upslope:   Meloche et al. (2025) JGR Earth Surface, doi:10.1029/2025JF008470
Downslope: Perzl (2007) JRC; Swiss ALIP/PRA; Maggioni & Gruber;
           Bühler et al.; Veitinger et al. (2016)
Flanks:    Gaume et al. (2015) The Cryosphere, doi:10.5194/tc-9-795-2015
           + Meloche (2025) cross-slope θ gradient
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

# Default physical parameters
STAUCHWALL_DEG   = 28.0   # slope threshold for downslope arrest (degrees)
FRICTION_DEG     = 27.0   # snow friction angle (degrees), Meloche Table 1
GAUME_ASPECT_CAP = 2.5    # max width/A_ca ratio (prevents runaway cross-slope)
MIN_POLYGON_AREA = 200.0  # m² — discard degenerate polygons smaller than this


# -----------------------------------------------------------------------
# Terrain helpers
# -----------------------------------------------------------------------

def compute_slope_aspect(dem: np.ndarray,
                         pixel_size: float = 1.0
                         ) -> tuple[np.ndarray, np.ndarray]:
    """
    Return slope (degrees) and aspect (degrees from N, clockwise) grids.

    Parameters
    ----------
    dem        : 2D elevation array (m), NaN for nodata
    pixel_size : pixel size in metres (assumes square pixels)
    """
    fill = np.where(np.isnan(dem), np.nanmean(dem), dem)
    dy, dx = np.gradient(fill, pixel_size)
    slope  = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    aspect = np.degrees(np.arctan2(-dx, dy)) % 360.0
    return slope, aspect


def pixel_to_utm(row: int, col: int, transform) -> tuple[float, float]:
    """Convert raster pixel (row, col) to UTM (x, y) using Affine transform."""
    x = transform.c + col * transform.a + row * transform.b
    y = transform.f + col * transform.d + row * transform.e
    return x, y


def utm_to_pixel(x: float, y: float, transform) -> tuple[int, int]:
    """Convert UTM (x, y) to nearest raster pixel (row, col)."""
    col = int(round((x - transform.c) / transform.a))
    row = int(round((y - transform.f) / transform.e))
    return row, col


# -----------------------------------------------------------------------
# Downslope: stauchwall location
# -----------------------------------------------------------------------

def find_stauchwall(trigger_row: int,
                    trigger_col: int,
                    slope_grid: np.ndarray,
                    aspect_grid: np.ndarray,
                    transform,
                    threshold_deg: float = STAUCHWALL_DEG,
                    max_steps: int = 500
                    ) -> tuple[int, int]:
    """
    Walk downslope from trigger pixel until slope drops below threshold.

    Uses the fall-line direction (aspect at each step) to advance.
    Returns (row, col) of the stauchwall pixel.

    If slope never drops below threshold within max_steps, returns
    the pixel at max_steps — a conservative (large) release estimate.

    References: Perzl (2007) JRC; Swiss ALIP/PRA; Maggioni & Gruber;
                Bühler et al.; Veitinger et al. (2016).
    """
    nrows, ncols = slope_grid.shape
    row, col     = trigger_row, trigger_col

    for _ in range(max_steps):
        if slope_grid[row, col] < threshold_deg:
            return row, col
        # Step one pixel in the downslope direction
        asp_rad = np.radians(aspect_grid[row, col])
        dr = int(np.sign( np.cos(asp_rad)))   # row advances with -y (north)
        dc = int(np.sign( np.sin(asp_rad)))   # col advances with +x (east)
        new_row = row + dr
        new_col = col + dc
        if not (0 <= new_row < nrows and 0 <= new_col < ncols):
            break
        if np.isnan(slope_grid[new_row, new_col]):
            break
        row, col = new_row, new_col

    return row, col


# -----------------------------------------------------------------------
# Cross-slope: Gaume (2015) + θ
# -----------------------------------------------------------------------

def estimate_cross_slope_width(
        trigger_cluster_id: int,
        A_ca: float,
        meloche_df: pd.DataFrame,
        cluster_map: np.ndarray,
        transform,
        n_lateral_neighbors: int = 6,
        gaume_aspect_cap: float = GAUME_ASPECT_CAP) -> float:
    """
    Estimate cross-slope release width (m).

    Implements the Gaume et al. (2015) / Meloche (2025) conceptual framework:
    cross-slope crack arrest is controlled by the WL shear-strength gradient θ
    in the lateral direction. A higher cross-slope θ means stronger heterogeneity
    and earlier arrest → narrower release.

    Approach:
      1. Find clusters laterally adjacent to the trigger cluster
      2. Compute mean cross-slope θ from τ_p differences
      3. Estimate width as A_ca × f(θ_cross / θ_downslope), capped at
         GAUME_ASPECT_CAP × A_ca

    If θ data is unavailable, falls back to width = A_ca (square release).

    References
    ----------
    Gaume et al. (2015) The Cryosphere 9(2):795–804
    Meloche et al. (2025) JGR Earth Surface, Fig.13 and Sec.4
    """
    if meloche_df.empty or 'theta' not in meloche_df.columns:
        return A_ca

    try:
        theta_down = float(meloche_df.loc[trigger_cluster_id, 'theta'])
    except (KeyError, TypeError):
        return A_ca

    if np.isnan(theta_down) or theta_down < 1e-6:
        return A_ca

    # Get trigger centroid in pixel space
    trig_pixels = np.argwhere(cluster_map == trigger_cluster_id)
    if len(trig_pixels) == 0:
        return A_ca
    t_row, t_col = trig_pixels.mean(axis=0)

    # Find candidate lateral neighbors — similar row, different col
    cids_all = np.unique(cluster_map[cluster_map > 0])
    lateral  = []
    for cid in cids_all:
        if cid == trigger_cluster_id:
            continue
        pxs = np.argwhere(cluster_map == cid)
        if len(pxs) == 0:
            continue
        c_row, c_col = pxs.mean(axis=0)
        # "Lateral" = roughly same row (within ±50 px), different column
        if abs(c_row - t_row) < 50 and abs(c_col - t_col) > 5:
            dist = abs(c_col - t_col)  # pixel distance
            lateral.append((cid, dist))

    lateral.sort(key=lambda x: x[1])
    lateral = lateral[:n_lateral_neighbors]

    if not lateral:
        return A_ca

    # Compute θ_cross from τ_p differences
    tau_trigger = float(meloche_df.loc[trigger_cluster_id, 'tau_p']
                        if 'tau_p' in meloche_df.columns
                        else meloche_df.loc[trigger_cluster_id, 'wl_shear_strength']
                        if 'wl_shear_strength' in meloche_df.columns
                        else np.nan)
    if np.isnan(tau_trigger):
        return A_ca

    theta_cross_vals = []
    for cid, dist_px in lateral:
        if cid not in meloche_df.index:
            continue
        col_name = ('tau_p' if 'tau_p' in meloche_df.columns
                    else 'wl_shear_strength')
        tau_lat = float(meloche_df.loc[cid, col_name])
        if not np.isnan(tau_lat) and dist_px > 0:
            # Convert pixel distance to metres (assumes 1m pixels)
            theta_cross_vals.append(abs(tau_trigger - tau_lat) / dist_px)

    if not theta_cross_vals:
        return A_ca

    theta_cross = float(np.mean(theta_cross_vals))

    # Width scaling: larger cross-slope θ → shorter arrest width
    # Ratio θ_down/θ_cross: if cross-slope is MORE heterogeneous than
    # downslope, crack arrests sooner laterally.
    if theta_cross > 1e-6:
        width_factor = min(theta_down / theta_cross, gaume_aspect_cap)
    else:
        width_factor = gaume_aspect_cap

    return float(A_ca * width_factor)


# -----------------------------------------------------------------------
# Upslope projection
# -----------------------------------------------------------------------

def project_along_aspect(x: float, y: float,
                          distance: float,
                          aspect_deg: float,
                          upslope: bool = True) -> tuple[float, float]:
    """
    Project a UTM point by `distance` metres along (or against) aspect.

    Parameters
    ----------
    x, y        : UTM coordinates of start point
    distance    : metres to project
    aspect_deg  : slope aspect in degrees clockwise from north
    upslope     : if True, project upslope (opposite of aspect)
    """
    asp_rad = np.radians(aspect_deg)
    dx = np.sin(asp_rad) * distance
    dy = np.cos(asp_rad) * distance
    sign = -1 if upslope else 1
    return x + sign * dx, y + sign * dy


# -----------------------------------------------------------------------
# Full 2D release polygon
# -----------------------------------------------------------------------

def make_release_polygon_2d(
        trigger_cluster_id: int,
        A_ca: float,
        meloche_df: pd.DataFrame,
        cluster_map: np.ndarray,
        dem: np.ndarray,
        transform,
        start_zone_mask: Optional[np.ndarray] = None,
        snap_features: 'Optional[pd.DataFrame]' = None,
        size_factor: float = 1.0,
        stauchwall_deg: float = STAUCHWALL_DEG,
        mode3_scale: float = 1.5,
        use_propagation: bool = True):
    """
    Build a 2D release polygon.

    Primary method (use_propagation=True):
        BFS cluster flood-fill with per-direction arrest criteria:
          - Upslope:   Meloche brittle Π₁ at candidate < Π₁ at trigger
          - Downslope: slope < stauchwall_deg (~28°)
          - Lateral:   Π₁ < Π₁_trigger × mode3_scale (mode III)
        size_factor relaxes (>1) or tightens (<1) arrest thresholds.

    Fallback (use_propagation=False, or if propagation returns None):
        Oriented rectangle from A_ca + stauchwall + Gaume cross-slope width.

    Returns shapely Polygon (UTM) or None.
    """
    slope_grid, _ = compute_slope_aspect(dem)

    # --- Build start zone polygon for post-clipping ---
    # Both the BFS and fallback paths must clip to start zone + stauchwall
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union
    import rasterio.features

    sz_polygon = None
    if start_zone_mask is not None:
        shapes = list(rasterio.features.shapes(
            start_zone_mask.astype(np.uint8),
            mask=start_zone_mask, transform=transform))
        if shapes:
            sz_polygon = unary_union(
                [Polygon(s['coordinates'][0]) for s, v in shapes if v == 1])

    # Build stauchwall mask: only terrain ≥ stauchwall_deg can be in release
    stauchwall_mask = slope_grid >= stauchwall_deg
    sw_shapes = list(rasterio.features.shapes(
        stauchwall_mask.astype(np.uint8),
        mask=stauchwall_mask, transform=transform))
    sw_polygon = None
    if sw_shapes:
        sw_polygon = unary_union(
            [Polygon(s['coordinates'][0]) for s, v in sw_shapes if v == 1])
        # Simplify to keep intersection fast — 1m tolerance at 1m grid
        sw_polygon = sw_polygon.simplify(1.0, preserve_topology=True)

    def _clip_polygon(poly):
        """Clip polygon to start zone AND stauchwall, return largest piece."""
        if poly is None or poly.is_empty:
            return None
        if sz_polygon is not None:
            poly = poly.intersection(sz_polygon)
        if sw_polygon is not None:
            poly = poly.intersection(sw_polygon)
        if poly.is_empty:
            return None
        # Intersection can produce GeometryCollection (mix of polygons,
        # lines, points at boundaries). Extract only polygon geometries.
        if poly.geom_type == 'GeometryCollection':
            polys = [g for g in poly.geoms
                     if g.geom_type in ('Polygon', 'MultiPolygon')]
            if not polys:
                return None
            poly = unary_union(polys)
        if poly.geom_type == 'MultiPolygon':
            poly = max(poly.geoms, key=lambda p: p.area)
        if poly.is_empty or poly.area < MIN_POLYGON_AREA:
            return None
        return poly

    # --- Primary: BFS propagation ---
    if use_propagation and not meloche_df.empty:
        polygon, failed = propagate_release(
            trigger_cluster_id = trigger_cluster_id,
            meloche_df         = meloche_df,
            cluster_map        = cluster_map,
            dem                = dem,
            slope_grid         = slope_grid,
            transform          = transform,
            start_zone_mask    = start_zone_mask,
            stauchwall_deg     = stauchwall_deg,
            mode3_scale        = mode3_scale,
            size_factor        = size_factor,
            A_ca               = A_ca,
            snap_features      = snap_features,
        )
        if polygon is not None:
            polygon = _clip_polygon(polygon)
            if polygon is not None:
                return polygon
        print(f"  propagate_release returned None for cluster {trigger_cluster_id}"
              f" — falling back to rectangle")

    # --- Fallback: oriented rectangle ---
    _, aspect_grid = compute_slope_aspect(dem)
    trig_px = np.argwhere(cluster_map == trigger_cluster_id)
    if len(trig_px) == 0:
        return None
    t_row, t_col = trig_px.mean(axis=0).astype(int)
    t_x, t_y    = pixel_to_utm(t_row, t_col, transform)
    t_aspect    = float(aspect_grid[t_row, t_col])

    up_x, up_y  = project_along_aspect(t_x, t_y, A_ca * size_factor,
                                        t_aspect, upslope=True)
    sw_row, sw_col = find_stauchwall(t_row, t_col, slope_grid, aspect_grid,
                                     transform, threshold_deg=stauchwall_deg)
    sw_x, sw_y  = pixel_to_utm(sw_row, sw_col, transform)
    half_width  = estimate_cross_slope_width(
        trigger_cluster_id, A_ca, meloche_df, cluster_map, transform
    ) * size_factor / 2.0

    asp_rad = np.radians(t_aspect)
    fall_x, fall_y   = np.sin(asp_rad), np.cos(asp_rad)
    cross_x, cross_y = -fall_y, fall_x
    corners = [
        (up_x - cross_x * half_width, up_y - cross_y * half_width),
        (up_x + cross_x * half_width, up_y + cross_y * half_width),
        (sw_x + cross_x * half_width, sw_y + cross_y * half_width),
        (sw_x - cross_x * half_width, sw_y - cross_y * half_width),
    ]
    polygon = Polygon(corners)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)

    return _clip_polygon(polygon)



# -----------------------------------------------------------------------
# Depth raster
# -----------------------------------------------------------------------


# -----------------------------------------------------------------------
# Cluster-level crack propagation (BFS flood-fill)
# -----------------------------------------------------------------------


# -----------------------------------------------------------------------
# Connected-region instability approach (replaces BFS crack propagation)
# -----------------------------------------------------------------------

def propagate_release(
        trigger_cluster_id: int,
        meloche_df: pd.DataFrame,
        cluster_map: np.ndarray,
        dem: np.ndarray,
        slope_grid: np.ndarray,
        transform,
        start_zone_mask: Optional[np.ndarray] = None,
        stauchwall_deg: float = STAUCHWALL_DEG,
        mode3_scale: float = 2.0,    # kept for API compat, not used
        size_factor: float = 1.0,
        A_ca: Optional[float] = None,
        snap_features: 'Optional[pd.DataFrame]' = None,
        k_neighbours: int = 8,
        max_clusters: int = 500):
    """
    Identify release zone as the connected region of clusters with
    Pi1 >= Pi1_trigger that is reachable from the trigger cluster.

    Physical basis (Gaume et al. 2015, Meloche et al. 2025):
      - Pi1 = tau_g / (theta * Lambda * sqrt(1+delta))
      - Where Pi1 >= Pi1_trigger the WL crack propagates without arrest
      - Where Pi1 < Pi1_trigger the crack arrests (stronger / more
        heterogeneous terrain)
      - The release zone is the CONNECTED region of Pi1 >= threshold
        that contains the trigger cluster, further constrained by:
          * slope >= stauchwall_deg (terrain must support release)
          * within start_zone_mask (hard operational boundary)
          * distance <= A_ca upslope (Meloche upslope arrest distance)

    size_factor scales the Pi1 threshold:
      > 1.0 → stricter threshold → smaller release (less terrain qualifies)
      < 1.0 → looser threshold  → larger release  (more terrain qualifies)
      = 1.0 → baseline: all terrain as unstable as trigger
    """
    from collections import deque
    from shapely.geometry import MultiPolygon
    from shapely.ops import unary_union
    import rasterio.features

    if meloche_df.empty or trigger_cluster_id not in meloche_df.index:
        return None, set()

    # --- Pi1 at trigger cluster ---
    loc = meloche_df.loc[trigger_cluster_id]
    row = loc.iloc[0] if isinstance(loc, pd.DataFrame) else loc

    col = 'Pi1_elastic' if 'Pi1_elastic' in meloche_df.columns else None
    if col is None:
        return None, set()

    pi1_raw = row[col]
    pi1_trigger = float(pi1_raw.iloc[0] if isinstance(pi1_raw, pd.Series)
                        else pi1_raw)
    if np.isnan(pi1_trigger) or pi1_trigger <= 0:
        return None, set()

    # Pi1 retained for reporting only
    threshold = pi1_trigger  # informational

    # Minimum tau_g for crack propagation — no tuning, same as trigger filter
    MIN_PROPAGATION_TAU_G = 50.0   # Pa

    print(f"    Pi1_trigger={pi1_trigger:.3f}  "
          f"MIN_tau_g={MIN_PROPAGATION_TAU_G:.0f}Pa  "
          f"size_factor={size_factor:.2f}")

    # --- Build per-cluster lookup: Pi1, mean slope ---
    def _scalar(cid, c):
        if cid not in meloche_df.index:
            return np.nan
        v = meloche_df.loc[cid, c]
        if isinstance(v, pd.DataFrame): v = v.iloc[0][c]
        elif isinstance(v, pd.Series):  v = v.iloc[0]
        return float(v)

    def _mean_slope(cid):
        pxs = np.argwhere(cluster_map == cid)
        if len(pxs) == 0:
            return 0.0
        return float(slope_grid[pxs[:, 0], pxs[:, 1]].mean())

    # --- Precompute trigger centroid for upslope distance cap ---
    t_pxs = np.argwhere(cluster_map == trigger_cluster_id)
    if len(t_pxs) == 0:
        return None, set()
    t_row  = int(t_pxs.mean(axis=0)[0])
    t_col  = int(t_pxs.mean(axis=0)[1])
    t_x, t_y = pixel_to_utm(t_row, t_col, transform)
    _, aspect_grid = compute_slope_aspect(dem)
    t_aspect = float(aspect_grid[t_row, t_col])

    # Upslope distance cap: A_ca (Meloche) — no upslope limit for downslope/lateral
    d_up = (max(A_ca, 20.0) if A_ca else 50.0) * size_factor

    # Stauchwall distance (downslope hard cap)
    sw_row, sw_col = find_stauchwall(
        t_row, t_col, slope_grid, aspect_grid, transform,
        threshold_deg=stauchwall_deg)
    sw_x, sw_y = pixel_to_utm(sw_row, sw_col, transform)
    d_down = max(float(np.sqrt((sw_x - t_x)**2 + (sw_y - t_y)**2)),
                 20.0) * size_factor

    print(f"    distance caps: up={d_up:.0f}m  down={d_down:.0f}m")

    # --- Build neighbour graph (k nearest centroids) ---
    from sklearn.neighbors import NearestNeighbors
    cids = np.array([c for c in np.unique(cluster_map) if c > 0])
    pxs  = np.array([np.argwhere(cluster_map == c).mean(axis=0) for c in cids])
    k_actual = min(k_neighbours + 1, len(cids))
    nbrs     = NearestNeighbors(n_neighbors=k_actual).fit(pxs)
    _, idxs  = nbrs.kneighbors(pxs)
    neighbours = {int(cids[i]): [int(cids[j]) for j in idxs[i][1:]]
                  for i in range(len(cids))}

    # Centroids in UTM
    centroids = {int(c): pixel_to_utm(int(pxs[i][0]), int(pxs[i][1]), transform)
                 for i, c in enumerate(cids)}

    # WL shear strength at trigger (Pa) — crack arrests where tau_p increases
    # tau_p lives in snap_features (release_zone_features CSV), not meloche_df
    tau_p_trigger = None
    if snap_features is not None and trigger_cluster_id in snap_features.index:
        raw = snap_features.loc[trigger_cluster_id, 'wl_shear_strength'] \
              if 'wl_shear_strength' in snap_features.columns else np.nan
        if isinstance(raw, pd.Series): raw = raw.iloc[0]
        v = float(raw) * 1000.0   # kPa → Pa
        tau_p_trigger = v if not np.isnan(v) and v > 0 else None
    if tau_p_trigger is None:
        print(f"    Warning: no tau_p for trigger {trigger_cluster_id}, "
              f"WL strength criterion disabled")

    # Lateral distance cap from Gaume cross-slope width estimate
    d_lat = estimate_cross_slope_width(
        trigger_cluster_id, A_ca if A_ca else 50.0,
        meloche_df, cluster_map, transform) * size_factor
    d_lat = max(d_lat, 15.0)

    tau_p_str = f"{tau_p_trigger:.0f} Pa" if tau_p_trigger else "N/A"

    # Elastic length Λ threshold for lateral arrest (Gaume et al. 2015)
    # Use median Λ of all clusters within the connected region rather than
    # trigger's own Λ — the trigger may be an outlier (lowest Λ in the zone)
    # and using it would block ALL lateral propagation immediately.
    # Median reflects the typical slab stiffness of the release terrain.
    lambda_trigger = None
    if not meloche_df.empty and 'Lambda' in meloche_df.columns:
        # Use median Λ of all start-zone clusters as the baseline
        lam_vals = meloche_df['Lambda'].dropna()
        if len(lam_vals):
            lam_raw = float(lam_vals.median())
            if lam_raw > 0:
                lambda_trigger = lam_raw
                print(f"    Lambda_median={lambda_trigger:.2f}m  "
                      f"(trigger own={_scalar(trigger_cluster_id, 'Lambda'):.2f}m)")

    print(f"    tau_p_trigger={tau_p_str}  d_lat={d_lat:.0f}m")

    # --- Per-cluster property lookup helpers ---
    def _get_props(cid):
        """Return dict of crack-relevant properties for a cluster."""
        props = {}
        if not meloche_df.empty and cid in meloche_df.index:
            for col in ['Lambda', 'tau_g']:
                if col in meloche_df.columns:
                    v = meloche_df.loc[cid, col]
                    if isinstance(v, pd.DataFrame): v = v.iloc[0][col]
                    elif isinstance(v, pd.Series):  v = v.iloc[0]
                    props[col] = float(v)
        if snap_features is not None and cid in snap_features.index:
            for col in ['slab_thickness', 'slab_density']:
                if col in snap_features.columns:
                    v = snap_features.loc[cid, col]
                    if isinstance(v, pd.Series): v = v.iloc[0]
                    try:
                        props[col] = float(v)
                    except (ValueError, TypeError):
                        pass
        return props

    # Pre-fetch trigger properties as baseline
    trigger_props = _get_props(trigger_cluster_id)

    def _qualifies(cid, current_props):
        """
        Does this cluster qualify for crack propagation?

        Uses LOCAL GRADIENT arrest: checks property change from current
        propagating cluster to candidate neighbor. A sharp discontinuity
        arrests the crack regardless of absolute property values.

        Arrest conditions:
          1. Outside start zone KML (hard boundary)
          2. Distance cap: upslope > A_ca, downslope > stauchwall
          3. Downslope slope < 28° (terrain)
          4. tau_g < MIN_PROPAGATION_TAU_G (slab too thin/flat)
          5. |ΔΛ/Λ| > LAMBDA_JUMP_FACTOR: sharp slab discontinuity in
             either direction arrests crack (stiffening OR softening)
          6. |Δh/h| > THICKNESS_JUMP_FACTOR: sharp slab thickness change
             in either direction arrests crack
        """
        # Hard boundary: start zone KML
        if start_zone_mask is not None:
            px = np.argwhere(cluster_map == cid)
            if len(px) == 0:
                return False
            r, c = int(px.mean(axis=0)[0]), int(px.mean(axis=0)[1])
            if not start_zone_mask[r, c]:
                return False

        # Direction and distance from trigger
        xy = centroids.get(cid)
        if xy is None:
            return False
        asp_rad        = np.radians(t_aspect)
        fall_x, fall_y = np.sin(asp_rad), np.cos(asp_rad)
        dx, dy         = xy[0] - t_x, xy[1] - t_y
        dist           = np.sqrt(dx**2 + dy**2)
        dot            = (dx * fall_x + dy * fall_y) / max(dist, 1e-6)

        is_downslope = dot >  0.707
        is_upslope   = dot < -0.707

        # Distance caps: upslope A_ca, downslope stauchwall
        if is_upslope   and dist > d_up:   return False
        if is_downslope and dist > d_down: return False

        # Downslope terrain threshold
        if is_downslope and _mean_slope(cid) < stauchwall_deg:
            return False

        # Minimum tau_g — slab must be thick/steep enough to sustain crack
        nbr_props = _get_props(cid)
        tau_g_nbr = nbr_props.get('tau_g', np.nan)
        if not np.isnan(tau_g_nbr) and tau_g_nbr < MIN_PROPAGATION_TAU_G:
            return False

        # LOCAL GRADIENT ARREST — compare neighbor to current propagating cluster
        # (not to trigger). Detects sharp slab edges and Λ discontinuities.

        # Slab property discontinuity arrest (Λ and thickness)
        # Physical basis: crack arrests at ANY sharp slab discontinuity —
        # both stiffening (higher Λ redistributes energy over longer length)
        # and softening (lower Λ reduces stress concentration at crack tip,
        # dropping K_I below K_Ic). Criterion: |ΔΛ/Λ| > threshold.
        # Same logic applies to slab thickness h.
        lam_current = current_props.get('Lambda', np.nan)
        lam_nbr     = nbr_props.get('Lambda', np.nan)
        if (not np.isnan(lam_current) and not np.isnan(lam_nbr)
                and lam_current > 0):
            rel_change = abs(lam_nbr - lam_current) / lam_current
            if rel_change > LAMBDA_JUMP_FACTOR * size_factor:
                return False

        # Slab thickness: arrest on sharp change in either direction
        h_current = current_props.get('slab_thickness', np.nan)
        h_nbr     = nbr_props.get('slab_thickness', np.nan)
        if (not np.isnan(h_current) and not np.isnan(h_nbr)
                and h_current > 0):
            rel_change = abs(h_nbr - h_current) / h_current
            if rel_change > THICKNESS_JUMP_FACTOR * size_factor:
                return False

        return True

    # Gradient arrest thresholds — relative change in either direction
    # LAMBDA_JUMP_FACTOR:    arrest if |ΔΛ/Λ| > 0.5 (50% change in either direction)
    # THICKNESS_JUMP_FACTOR: arrest if |Δh/h|  > 0.25 (25% change in either direction)
    # size_factor multiplies threshold → larger size_factor = more tolerant = larger release
    LAMBDA_JUMP_FACTOR    = 0.5
    THICKNESS_JUMP_FACTOR = 0.25

    # --- Flood-fill from trigger through qualifying clusters ---
    if not _qualifies(trigger_cluster_id, trigger_props):
        pxs_t = np.argwhere(cluster_map == trigger_cluster_id)
        if len(pxs_t) == 0:
            return None, set()

    failed         = {trigger_cluster_id}
    cluster_props  = {trigger_cluster_id: trigger_props}  # track per-cluster props
    frontier       = deque([trigger_cluster_id])
    visited        = {trigger_cluster_id}

    while frontier and len(failed) < max_clusters:
        current       = frontier.popleft()
        current_props = cluster_props.get(current, trigger_props)
        for nbr in neighbours.get(current, []):
            if nbr in visited:
                continue
            visited.add(nbr)
            if _qualifies(nbr, current_props):
                failed.add(nbr)
                cluster_props[nbr] = _get_props(nbr)
                frontier.append(nbr)

    # Debug direction breakdown
    dir_counts = {'upslope': 0, 'downslope': 0, 'lateral': 0, 'other': 0}
    for cid in failed:
        if cid == trigger_cluster_id:
            continue
        xy = centroids.get(cid)
        if xy is None:
            continue
        asp_rad = np.radians(t_aspect)
        fall_x, fall_y = np.sin(asp_rad), np.cos(asp_rad)
        dx, dy = xy[0] - t_x, xy[1] - t_y
        dist   = np.sqrt(dx**2 + dy**2)
        dot    = (dx * fall_x + dy * fall_y) / max(dist, 1e-6)
        if   dot < -0.707: dir_counts['upslope']   += 1
        elif dot >  0.707: dir_counts['downslope']  += 1
        else:              dir_counts['lateral']    += 1
    print(f"  Connected region: {len(failed)} clusters "
          f"(up={dir_counts['upslope']} "
          f"down={dir_counts['downslope']} "
          f"lat={dir_counts['lateral']})")

    if len(failed) < 2:
        return None, failed

    # Rasterize to polygon
    mask   = np.isin(cluster_map, list(failed)).astype(np.uint8)
    shapes = list(rasterio.features.shapes(
        mask, mask=mask.astype(bool), transform=transform))
    if not shapes:
        return None, failed

    polys   = [__import__('shapely.geometry', fromlist=['shape']).shape(s)
               for s, v in shapes if v == 1]
    polygon = unary_union(polys)
    if polygon.geom_type == 'MultiPolygon':
        polygon = max(polygon.geoms, key=lambda p: p.area)
    if polygon.is_empty or polygon.area < MIN_POLYGON_AREA:
        return None, failed

    return polygon, failed



def depth_from_snowpack(cluster_map: np.ndarray,
                        ds: xr.Dataset,
                        snapshot: str,
                        wl_fraction: float = 0.80) -> np.ndarray:
    """
    Per-pixel slab depth (m) from SNOWPACK HS at snapshot date.

    Parameters
    ----------
    cluster_map  : 2D array of cluster IDs
    ds           : xsnow Dataset (Zarr-backed)
    snapshot     : date string 'YYYY-MM-DD'
    wl_fraction  : fraction of HS assumed to be slab (default 0.80)
                   WL occupies the bottom (1 - wl_fraction)

    Returns
    -------
    float32 array, shape = cluster_map.shape, values in metres
    """
    snap_ts = np.datetime64(f"{snapshot}T12:00")
    times   = ds.coords['time'].values
    tidx    = int(np.argmin(np.abs(times - snap_ts)))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ds_t = ds.isel(time=tidx).compute()
        sq   = [d for d in ['slope', 'realization'] if d in ds_t.dims]
        hs   = ds_t['HS'].squeeze(sq).values    # cm, (n_locations,)

    locs    = ds.coords['location'].values
    loc_ids = np.array([int(str(l).split('_')[-1]) for l in locs])
    id_to_hs = dict(zip(loc_ids, hs))

    depth_m = np.full(cluster_map.shape, np.nan, dtype=np.float32)
    for cid, h in id_to_hs.items():
        if not np.isnan(h):
            depth_m[cluster_map == cid] = h * wl_fraction / 100.0  # cm→m

    return depth_m


def rasterize_release_polygon(polygon,
                               depth_grid: np.ndarray,
                               dem_shape: tuple,
                               transform) -> np.ndarray:
    """
    Burn release polygon onto the depth grid.

    Returns float32 raster with depth values inside the polygon,
    NaN outside.

    Parameters
    ----------
    polygon    : shapely Polygon in UTM coordinates
    depth_grid : per-pixel depth (m), same shape as DEM
    dem_shape  : (nrows, ncols)
    transform  : rasterio Affine transform
    """
    import rasterio.features

    if polygon is None or polygon.is_empty:
        return np.full(dem_shape, np.nan, dtype=np.float32)

    mask = rasterio.features.geometry_mask(
        [polygon.__geo_interface__],
        out_shape=dem_shape,
        transform=transform,
        invert=True)

    release = np.where(mask & ~np.isnan(depth_grid),
                       depth_grid.astype(np.float32), np.nan)
    return release


# -----------------------------------------------------------------------
# Comparison plot: Meloche polygons vs observed release area
# -----------------------------------------------------------------------


# -----------------------------------------------------------------------
# Comparison plot: Meloche polygons vs observed release area
# -----------------------------------------------------------------------

def plot_release_comparison(
        meloche_polygons: list,
        observed_polygon,
        dem: np.ndarray,
        transform,
        start_zone_mask=None,
        trigger_labels=None,
        out_path=None,
        title: str = "Release polygon comparison") -> None:
    """
    Plot Meloche-derived polygons vs observed release area.
    One star marker per trigger nucleation point, labels in legend.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.colors import LightSource
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(10, 10))

    fill_dem  = np.where(np.isnan(dem), np.nanmean(dem), dem)
    hillshade = LightSource(azdeg=315, altdeg=45).hillshade(fill_dem, dx=1.0, dy=1.0)
    nrows, ncols = dem.shape
    extent = [transform.c,
              transform.c + ncols * transform.a,
              transform.f + nrows * transform.e,
              transform.f]
    ax.imshow(hillshade, cmap='gray', extent=extent,
              alpha=0.6, aspect='auto', origin='upper')

    # Start zone
    if start_zone_mask is not None:
        ax.contour(start_zone_mask.astype(float), levels=[0.5],
                   colors=['limegreen'], linewidths=2.0, alpha=0.9,
                   extent=extent, origin='upper')

    # Observed release area
    obs_area = 0.0
    if observed_polygon is not None and not observed_polygon.is_empty:
        obs_area = observed_polygon.area
        xs, ys = observed_polygon.exterior.xy
        ax.fill(xs, ys, alpha=0.30, color='red', zorder=3)
        ax.plot(xs, ys, color='red', linewidth=2.5, zorder=4)

    # Meloche polygons — one colour per trigger, star at centroid
    colors    = plt.cm.tab10.colors
    mel_areas = []

    for i, (poly, size_f) in enumerate(meloche_polygons):
        if poly is None or poly.is_empty:
            continue
        color = colors[i % len(colors)]
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, alpha=0.20, color=color, zorder=2)
        ax.plot(xs, ys, color=color, linewidth=1.8, alpha=0.85, zorder=5)
        mel_areas.append(poly.area)

        # Nucleation star
        cx, cy = poly.centroid.x, poly.centroid.y
        ax.plot(cx, cy, marker='*', markersize=14, color=color,
                markeredgecolor='white', markeredgewidth=0.8, zorder=10)

        # Short map label with outline effect — no box
        short = f"T{i+1}"
        txt   = ax.text(cx + 8, cy + 8, short, fontsize=9,
                        color=color, fontweight='bold', zorder=11)
        txt.set_path_effects([pe.Stroke(linewidth=2.5, foreground='white'),
                               pe.Normal()])

    # Stats box
    if mel_areas and obs_area:
        ratio = float(np.median(mel_areas)) / obs_area
        line1 = "Observed:       %6.0f m2" % obs_area
        line2 = "Meloche P50:    %6.0f m2" % float(np.median(mel_areas))
        line3 = "Ratio Mel/Obs:    %.2f" % ratio
        stats = line1 + "\n" + line2 + "\n" + line3
    elif mel_areas:
        stats = "Meloche P50: %.0f m2" % float(np.median(mel_areas))
    else:
        stats = "No polygons generated"
    ax.text(0.02, 0.02, stats, transform=ax.transAxes, fontsize=8.5,
            verticalalignment='bottom', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white',
                      edgecolor='gray', alpha=0.85))

    # Legend
    handles = [
        Line2D([0],[0], color='red',       linewidth=2.5,
               label="Observed Jan 18  (%.0f m2)" % obs_area),
        Line2D([0],[0], color='limegreen', linewidth=2.0, label='Start zone'),
    ]
    for i, (poly, size_f) in enumerate(meloche_polygons):
        if poly is None or poly.is_empty:
            continue
        color = colors[i % len(colors)]
        lbl   = trigger_labels[i] if trigger_labels else ("T%d" % (i+1))
        handles.append(Patch(facecolor=color, alpha=0.5, edgecolor=color,
                             label="%s  (%.0f m2)" % (lbl, poly.area)))

    ax.legend(handles=handles, loc='upper right', fontsize=7.5, framealpha=0.90)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Easting (m UTM)')
    ax.set_ylabel('Northing (m UTM)')
    ax.ticklabel_format(style='plain', axis='both')
    plt.tight_layout()

    if out_path:
        fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
        print(f"Release comparison plot: {out_path}")
        plt.close(fig)
    else:
        plt.show()
        
"""
Cluster grid cells by HS evolution to reduce the number of SMET files.

Instead of writing one SMET per cell (146K files), group cells with
similar snow depth progression into clusters and write one SMET per cluster.

Also handles domain masking from KML boundary polygons.

Approach:
  1. Parse KML boundary → rasterize to grid → combine with slope threshold
  2. Build HS matrix at survey times: (n_cells × n_surveys)
  3. PCA to reduce dimensionality
  4. K-means in PCA space
  5. Optionally enforce spatial contiguity
  6. Output: cluster map + representative HS per cluster
"""

import numpy as np
import pandas as pd
import re
from pathlib import Path
from scipy.ndimage import label as connected_components
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from matplotlib.path import Path as MplPath
from typing import Optional
import warnings


# =====================================================================
# Domain mask from KML boundary
# =====================================================================

def parse_kml_polygon(kml_path: str) -> list:
    """
    Parse a KML file and return polygon coordinates as [(lon, lat), ...].
    Handles the first <coordinates> block found.
    """
    with open(kml_path) as f:
        kml = f.read()

    match = re.search(r'<coordinates>\s*(.*?)\s*</coordinates>', kml, re.DOTALL)
    if not match:
        raise ValueError(f"No <coordinates> found in {kml_path}")

    coords = []
    for pt in match.group(1).strip().split():
        parts = pt.split(',')
        lon, lat = float(parts[0]), float(parts[1])
        coords.append((lon, lat))

    return coords


def build_domain_mask(dem: np.ndarray, transform, crs,
                      kml_path: str | None = None,
                      min_slope_deg: float = 15.0,
                      resolution: float = 1.0) -> np.ndarray:
    """
    Build the analysis domain mask combining:
      - KML boundary polygon (if provided)
      - Slope threshold
      - Valid DEM cells

    Parameters
    ----------
    dem : 2D elevation array
    transform : rasterio transform
    crs : rasterio CRS
    kml_path : path to KML boundary file (optional)
    min_slope_deg : minimum slope angle
    resolution : grid cell size

    Returns
    -------
    boolean mask (True = cell is in domain)
    """
    # Start with valid DEM
    mask = ~np.isnan(dem)

    # Slope filter
    fill = np.where(np.isnan(dem), np.nanmean(dem), dem)
    dy, dx = np.gradient(fill, resolution)
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    mask = mask & (slope >= min_slope_deg)

    # KML boundary filter
    if kml_path and Path(kml_path).exists():
        coords_lonlat = parse_kml_polygon(kml_path)
        print(f"  KML boundary: {len(coords_lonlat)} vertices from {Path(kml_path).name}")

        # Convert to grid CRS (UTM)
        try:
            from pyproj import Transformer
            transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            utm_coords = [transformer.transform(lon, lat) for lon, lat in coords_lonlat]
        except ImportError:
            raise ImportError("pyproj is required for KML boundary reprojection")

        # Rasterize polygon
        poly_path = MplPath(utm_coords)
        rows, cols = np.mgrid[0:dem.shape[0], 0:dem.shape[1]]
        eastings = transform[2] + cols * transform[0]
        northings = transform[5] + rows * transform[4]
        points = np.column_stack([eastings.ravel(), northings.ravel()])
        in_poly = poly_path.contains_points(points).reshape(dem.shape)

        n_before = mask.sum()
        mask = mask & in_poly
        print(f"  Domain: {n_before} cells → {mask.sum()} after KML clip")
    else:
        if kml_path:
            print(f"  WARNING: KML file not found: {kml_path}")
        print(f"  Domain: {mask.sum()} cells (slope ≥ {min_slope_deg}°, no boundary)")

    return mask


def build_survey_hs_matrix(survey_grids: dict, valid_mask: np.ndarray) -> tuple:
    """
    Build the (n_cells × n_surveys) HS matrix from survey grids.

    Parameters
    ----------
    survey_grids : dict mapping date_str -> 2D HS array
    valid_mask : boolean mask of cells to include

    Returns
    -------
    hs_matrix : (n_cells, n_surveys) array
    cell_indices : (n_cells, 2) array of (row, col) indices
    survey_dates : list of date strings in order
    """
    dates = sorted(survey_grids.keys())
    candidate_idx = np.argwhere(valid_mask)
    n_surveys = len(dates)

    # Build full matrix (vectorized)
    hs_full = np.zeros((len(candidate_idx), n_surveys), dtype=np.float32)
    rows = candidate_idx[:, 0]
    cols = candidate_idx[:, 1]
    for j, d in enumerate(dates):
        grid = survey_grids[d]
        hs_full[:, j] = grid[rows, cols]

    # Filter out cells with any NaN across surveys
    valid_rows = ~np.isnan(hs_full).any(axis=1)
    hs_matrix = hs_full[valid_rows]
    cell_idx = candidate_idx[valid_rows]

    n_dropped = len(candidate_idx) - len(cell_idx)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} cells with NaN in one or more surveys")

    return hs_matrix, cell_idx, dates


def cluster_cells(hs_matrix: np.ndarray,
                  cell_indices: np.ndarray,
                  grid_shape: tuple,
                  n_clusters: int = 300,
                  max_cells_per_cluster: int = 20,
                  max_cluster_std_m: float = 0.05,  # 5 cm
                  n_pca_components: float | int = 0.99, # 99% variance retained
                  min_cluster_size: int = 4,
                  enforce_contiguity: bool = True) -> np.ndarray:
    """
    Cluster cells by HS evolution with recursive splitting of oversized clusters.
 
    1. PCA on full HS matrix
    2. Initial K-means to get approximate clusters
    3. Recursively split any cluster that is BOTH:
       - larger than max_cells_per_cluster AND
       - has intra-cluster HS std above max_cluster_std_m (if set)
 
    Parameters
    ----------
    hs_matrix : (n_cells, n_surveys) array
    cell_indices : (n_cells, 2) array of (row, col)
    grid_shape : (nrows, ncols) of the full grid
    n_clusters : initial target number of clusters
    max_cells_per_cluster : recursively split clusters larger than this
    max_cluster_std_m : skip splitting if cluster HS std is below this (meters).
                        If None, split by size only.
    n_pca_components : number of PCA components or confidance interval (defauls 95%)
    min_cluster_size : don't split clusters smaller than this
    enforce_contiguity : if True, split disconnected cluster regions
 
    Returns
    -------
    cluster_map : 2D array (nrows, ncols) with cluster IDs (0 = no data)
    """
    n_cells, n_surveys = hs_matrix.shape
 
    # PCA dimensionality reduction (once, for all subsequent splitting)
    n_comp = min(n_pca_components, n_surveys, n_cells)
    pca = PCA(n_components=n_comp)
    hs_pca = pca.fit_transform(hs_matrix)
    explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA: {n_comp} components explain {explained:.1%} of variance")
 
    # Initial K-means
    n_k = min(n_clusters, n_cells // max(min_cluster_size, 1))
    kmeans = MiniBatchKMeans(n_clusters=n_k, batch_size=min(10000, n_cells),
                             random_state=42, n_init=3)
    labels = kmeans.fit_predict(hs_pca)
    print(f"  Initial K-means: {n_k} clusters from {n_cells} cells")
 
    # --- Recursive splitting of oversized clusters ---
    next_label = labels.max() + 1
    max_iterations = 50
    iteration = 0
    n_skipped_by_std = 0
 
    while iteration < max_iterations:
        unique_labels, counts = np.unique(labels, return_counts=True)
        oversized = unique_labels[counts > max_cells_per_cluster]
 
        if len(oversized) == 0:
            break
 
        n_split = 0
        for cid in oversized:
            mask = labels == cid
            n_in_cluster = mask.sum()
 
            if n_in_cluster <= max_cells_per_cluster:
                continue
            if n_in_cluster < 2 * min_cluster_size:
                continue
 
            # Variance check: skip if cluster is already tight
            if max_cluster_std_m is not None:
                # Mean std across surveys for cells in this cluster
                cluster_hs = hs_matrix[mask]  # (n_cells_in_cluster, n_surveys)
                per_survey_std = np.std(cluster_hs, axis=0)  # std across cells, per survey
                #mean_std = np.mean(per_survey_std)
                mean_std = np.max(per_survey_std) # Use the maximum variance found across all surveys
                if mean_std < max_cluster_std_m:
                    n_skipped_by_std += 1
                    continue
 
            # Split this cluster into 2 using its PCA features
            sub_features = hs_pca[mask]
            sub_km = MiniBatchKMeans(n_clusters=2, random_state=42 + iteration, n_init=3)
            sub_labels = sub_km.fit_predict(sub_features)
 
            # Check split quality: both halves must be meaningful
            n_half0 = (sub_labels == 0).sum()
            n_half1 = (sub_labels == 1).sum()
            if min(n_half0, n_half1) < min_cluster_size:
                continue
 
            # Assign: keep one half as original label, other gets new label
            indices = np.where(mask)[0]
            for i, idx in enumerate(indices):
                if sub_labels[i] == 1:
                    labels[idx] = next_label
 
            next_label += 1
            n_split += 1
 
        iteration += 1
        if n_split == 0:
            break
 
    # Report
    unique_labels, counts = np.unique(labels, return_counts=True)
    n_final = len(unique_labels)
    max_size = counts.max()
    print(f"  After recursive splitting: {n_final} clusters "
          f"(max size: {max_size}, target max: {max_cells_per_cluster})")
    if n_skipped_by_std > 0:
        print(f"  Skipped {n_skipped_by_std} split(s) — cluster std < "
              f"{max_cluster_std_m*100:.0f} cm")
    if max_size > max_cells_per_cluster:
        n_over = np.sum(counts > max_cells_per_cluster)
        print(f"  {n_over} cluster(s) still exceed max size "
              f"(kept because std < threshold or unsplittable)")
 
    # Build cluster map on the grid (re-index labels to 1-based contiguous)
    label_remap = {old: new + 1 for new, old in enumerate(unique_labels)}
    cluster_map = np.zeros(grid_shape, dtype=np.int32)
    for i, (r, c) in enumerate(cell_indices):
        cluster_map[r, c] = label_remap[labels[i]]
 
    if enforce_contiguity:
        cluster_map = enforce_spatial_contiguity(cluster_map, min_cluster_size)
        n_final = len(np.unique(cluster_map[cluster_map > 0]))
        print(f"  After contiguity enforcement: {n_final} clusters")
 
    return cluster_map


def enforce_spatial_contiguity(cluster_map: np.ndarray,
                                min_size: int = 10) -> np.ndarray:
    """
    Split disconnected cluster regions into separate clusters.
    Merge tiny clusters into their nearest spatial neighbor.
    """
    max_id = cluster_map.max()
    new_map = np.zeros_like(cluster_map)
    next_id = 1

    for cid in range(1, max_id + 1):
        mask = cluster_map == cid
        if not mask.any():
            continue

        # Find connected components within this cluster
        labeled, n_components = connected_components(mask)

        for comp in range(1, n_components + 1):
            comp_mask = labeled == comp
            comp_size = comp_mask.sum()

            if comp_size >= min_size:
                new_map[comp_mask] = next_id
                next_id += 1
            else:
                # Merge small component into nearest neighboring cluster
                merge_id = find_nearest_cluster(new_map, comp_mask)
                if merge_id > 0:
                    new_map[comp_mask] = merge_id
                else:
                    # No neighbor found, keep as own cluster
                    new_map[comp_mask] = next_id
                    next_id += 1

    return new_map


def find_nearest_cluster(cluster_map: np.ndarray,
                          target_mask: np.ndarray) -> int:
    """Find the cluster ID of the nearest cell to target_mask."""
    from scipy.ndimage import distance_transform_edt

    # Distance from each cell to the target region
    # Invert: distance_transform gives distance to nearest zero
    inv = ~target_mask
    dist, indices = distance_transform_edt(inv, return_indices=True)

    # Find cells adjacent to target (within 2 pixels)
    border = (dist > 0) & (dist <= 2) & (cluster_map > 0) & ~target_mask
    if border.any():
        # Most common cluster ID among border cells
        border_ids = cluster_map[border]
        values, counts = np.unique(border_ids, return_counts=True)
        return int(values[np.argmax(counts)])

    return 0


def compute_cluster_representatives(cluster_map: np.ndarray,
                                      grid_stack: np.ndarray,
                                      timestamps: list) -> dict:
    """
    Compute representative HS time series for each cluster.

    Parameters
    ----------
    cluster_map : 2D array of cluster IDs
    grid_stack : (n_times, nrows, ncols) hourly HS grids
    timestamps : list of timestamps matching grid_stack axis 0

    Returns
    -------
    dict mapping cluster_id -> {
        'hs_series': 1D array (n_times,) mean HS in meters,
        'n_cells': int,
        'centroid_row': float,
        'centroid_col': float,
    }
    """
    cluster_ids = np.unique(cluster_map[cluster_map > 0])
    representatives = {}

    for cid in cluster_ids:
        mask = cluster_map == cid
        rows, cols = np.where(mask)

        cell_hs = grid_stack[:, rows, cols]  # (n_times, n_cells)

        # Drop cells that are NaN for every timestep — they are outside
        # the valid hourly grid extent and would pollute the cluster mean.
        valid_cells = ~np.all(np.isnan(cell_hs), axis=0)
        if not valid_cells.any():
            # Entire cluster is outside the hourly grid extent — skip it.
            # This shouldn't happen in a well-formed run but log it.
            print(f"  WARNING: cluster {cid} has no valid cells in hourly grids, skipping")
            continue

        cell_hs = cell_hs[:, valid_cells]

        with np.errstate(all='ignore'):
            hs_series = np.nanmean(cell_hs, axis=1)

        # Interpolate over any remaining all-NaN timesteps (isolated boundary
        # gaps), then fill any leading/trailing NaN with 0 (bare ground).
        nan_ts = np.isnan(hs_series)
        if nan_ts.any():
            hs_series = (pd.Series(hs_series)
                         .interpolate(method='linear', limit_direction='both')
                         .fillna(0.0)
                         .to_numpy())

        n_cells = int(valid_cells.sum())
        representatives[int(cid)] = {
            'hs_series': hs_series,
            'n_cells': n_cells,
            'centroid_row': float(np.mean(rows[valid_cells])),
            'centroid_col': float(np.mean(cols[valid_cells])),
        }

    return representatives

def auto_select_n_clusters(n_cells: int, target_cells_per_cluster: int = 50) -> int:
    """
    Heuristic: aim for ~target_cells_per_cluster cells per cluster.
    Bounded between 50 and 2000.
    """
    n = max(50, min(2000, n_cells // target_cells_per_cluster))
    return n


# =====================================================================
# Mid-season cluster quality assessment and splitting
# =====================================================================

def adaptive_split_threshold(median_hs_m: float,
                              rel_frac: float = 0.10,
                              min_abs: float = 0.03,
                              max_abs: float = 0.12) -> float:
    """
    Adaptive HS std threshold for splitting (metres).

    Scales with snow depth so shallow HS differences count more:
        threshold = clip(rel_frac × median_hs, min_abs, max_abs)

    At 30 cm HS → threshold = 3 cm (min clamp).
    At 80 cm HS → threshold = 8 cm.
    At 120+ cm HS → threshold = 12 cm (max clamp, upper snowpack still matters).
    """
    return float(np.clip(rel_frac * median_hs_m, min_abs, max_abs))


def assess_cluster_quality(cluster_map: np.ndarray,
                            survey_hs_matrix: np.ndarray,
                            cell_indices: np.ndarray,
                            rel_frac: float = 0.10,
                            min_abs: float = 0.03,
                            max_abs: float = 0.12,
                            min_cluster_size: int = 4) -> list:
    """
    Assess each cluster's HS homogeneity against its adaptive threshold.

    Parameters
    ----------
    cluster_map       : (nrows, ncols) cluster ID array
    survey_hs_matrix  : (n_cells, n_surveys) HS values in metres
    cell_indices      : (n_cells, 2) [row, col] matching survey_hs_matrix rows
    rel_frac, min_abs, max_abs : adaptive threshold parameters (metres)
    min_cluster_size  : clusters below 2× this cannot be split meaningfully

    Returns
    -------
    list of dicts — one per cluster, sorted by cid, keys:
        cid, n_cells, max_std, threshold, should_split
    """
    cid_per_cell = np.array([cluster_map[r, c] for r, c in cell_indices])
    results = []
    for cid in np.unique(cid_per_cell):
        if cid == 0:
            continue
        sel = cid_per_cell == cid
        n = int(sel.sum())
        if n < 2 * min_cluster_size:
            results.append({'cid': int(cid), 'n_cells': n,
                            'max_std': 0.0, 'threshold': 0.0,
                            'should_split': False})
            continue
        cluster_hs = survey_hs_matrix[sel]
        median_hs = float(np.nanmedian(cluster_hs))
        threshold = adaptive_split_threshold(median_hs, rel_frac, min_abs, max_abs)
        per_survey_std = np.nanstd(cluster_hs, axis=0)
        max_std = float(np.max(per_survey_std))
        results.append({
            'cid': int(cid),
            'n_cells': n,
            'max_std': round(max_std, 4),
            'threshold': round(threshold, 4),
            'should_split': bool(max_std > threshold),
        })
    return results


def bisect_cluster(cluster_map: np.ndarray,
                   cid: int,
                   survey_hs_matrix: np.ndarray,
                   cell_indices: np.ndarray,
                   min_cluster_size: int = 4) -> tuple:
    """
    Split cluster ``cid`` into two children using 2-means on HS PCA.

    Larger child (by pixel count) inherits ``cid``.
    Smaller child gets ``cluster_map.max() + 1``.

    Spatial contiguity is NOT re-enforced here to preserve existing cluster IDs.
    Run enforce_spatial_contiguity on the full season's first clustering only.

    Parameters
    ----------
    cluster_map      : (nrows, ncols) — modified in-place and returned
    cid              : cluster ID to split
    survey_hs_matrix : (n_cells, n_surveys) — rows match cell_indices
    cell_indices     : (n_cells, 2) [row, col]
    min_cluster_size : both children must have ≥ this many cells

    Returns
    -------
    (updated_cluster_map, child_cid) on success, or (cluster_map, None) on failure.
    """
    cid_per_cell = np.array([cluster_map[r, c] for r, c in cell_indices])
    sel_idx = np.where(cid_per_cell == cid)[0]
    n = len(sel_idx)

    if n < 2 * min_cluster_size:
        return cluster_map, None

    cluster_hs = survey_hs_matrix[sel_idx]  # (n_in_cluster, n_surveys)

    n_comp = min(n - 1, cluster_hs.shape[1], 8)
    if n_comp < 1:
        return cluster_map, None

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pca = PCA(n_components=n_comp)
        features = pca.fit_transform(cluster_hs)

    sub_km = MiniBatchKMeans(n_clusters=2, random_state=42, n_init=5)
    sub_labels = sub_km.fit_predict(features)

    n0 = int((sub_labels == 0).sum())
    n1 = int((sub_labels == 1).sum())
    if min(n0, n1) < min_cluster_size:
        return cluster_map, None

    child_cid = int(cluster_map.max()) + 1

    # Larger child keeps parent cid; minority group gets child_cid
    keep_label = 0 if n0 >= n1 else 1
    for i, idx in enumerate(sel_idx):
        if sub_labels[i] != keep_label:
            r, c = cell_indices[idx]
            cluster_map[r, c] = child_cid

    return cluster_map, child_cid

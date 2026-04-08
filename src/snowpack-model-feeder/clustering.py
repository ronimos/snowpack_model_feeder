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
                      kml_path: str = None,
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
                  n_pca_components: int = 6,
                  min_cluster_size: int = 10,
                  enforce_contiguity: bool = False) -> np.ndarray:
    """
    Cluster cells by HS evolution.

    Parameters
    ----------
    hs_matrix : (n_cells, n_surveys) array
    cell_indices : (n_cells, 2) array of (row, col)
    grid_shape : (nrows, ncols) of the full grid
    n_clusters : target number of clusters
    n_pca_components : number of PCA components for dimensionality reduction
    min_cluster_size : merge clusters smaller than this into nearest neighbor
    enforce_contiguity : if True, split disconnected cluster regions (slower)

    Returns
    -------
    cluster_map : 2D array (nrows, ncols) with cluster IDs (0 = no data)
    """
    n_cells, n_surveys = hs_matrix.shape

    # PCA dimensionality reduction
    n_comp = min(n_pca_components, n_surveys, n_cells)
    pca = PCA(n_components=n_comp)
    hs_pca = pca.fit_transform(hs_matrix)
    explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA: {n_comp} components explain {explained:.1%} of variance")

    # K-means clustering
    n_k = min(n_clusters, n_cells // max(min_cluster_size, 1))
    kmeans = MiniBatchKMeans(n_clusters=n_k, batch_size=min(10000, n_cells),
                             random_state=42, n_init=3)
    labels = kmeans.fit_predict(hs_pca)
    print(f"  K-means: {n_k} clusters from {n_cells} cells")

    # Build cluster map on the grid
    cluster_map = np.zeros(grid_shape, dtype=np.int32)
    for i, (r, c) in enumerate(cell_indices):
        cluster_map[r, c] = labels[i] + 1  # 1-indexed, 0 = no data

    if enforce_contiguity:
        cluster_map = enforce_spatial_contiguity(cluster_map, min_cluster_size)

    n_final = len(np.unique(cluster_map[cluster_map > 0]))
    print(f"  Final cluster count: {n_final}")

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
        n_cells = len(rows)

        # Mean HS across cluster cells at each timestep (vectorized)
        hs_series = np.nanmean(grid_stack[:, rows, cols], axis=1)

        # Cluster centroid
        centroid_r = float(np.mean(rows))
        centroid_c = float(np.mean(cols))

        representatives[int(cid)] = {
            'hs_series': hs_series,
            'n_cells': n_cells,
            'centroid_row': float(centroid_r),
            'centroid_col': float(centroid_c),
        }

    return representatives


def auto_select_n_clusters(n_cells: int, target_cells_per_cluster: int = 1000) -> int:
    """
    Heuristic: aim for ~target_cells_per_cluster cells per cluster.
    Bounded between 50 and 2000.
    """
    n = max(50, min(2000, n_cells // target_cells_per_cluster))
    return n
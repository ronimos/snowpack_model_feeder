# step_scenarios() — Design Sketch

## Function signature

```python
def step_scenarios(
    cfg: ProjectConfig,
    snapshot_date: str,          # "2026-01-17" — day before event
    ds: xr.Dataset,              # SNOWPACK Zarr dataset
    cluster_map: np.ndarray,     # cluster ID raster
    dem: np.ndarray,             # DEM raster (m)
    transform,                   # rasterio Affine transform
    groups: dict,                # release/adjacent/reference cluster IDs
                                 # from analyze_release_zone
    meloche_df: pd.DataFrame,    # per-cluster Meloche features
                                 # from compute_meloche_features()

    # Trigger location axis
    n_trigger_locations: int = 5,   # top-N weakest Sk38 clusters

    # Release size axis
    release_size_sigma: float = 0.15,  # fractional std on A_ca
    n_size_samples: int = 5,           # P10/P25/P50/P75/P90

    # Slab depth axis
    depth_percentiles: list = [10, 50, 90],  # from cluster HS distribution

    # com1DFA parameters
    rho_snow: float = None,      # kg/m³ — None = use per-cluster slab_density
    phi_deg: float = 27.0,       # friction angle (Voellmy μ derived from this)
    mu: float = 0.155,           # Voellmy dry friction coefficient
    xi: float = 1500.0,          # Voellmy turbulent friction (m/s²)
) -> Path:                       # returns path to output directory
```

## Output schema

```
outputs/scenarios/2026-01-17/
    metadata.json               # date, n_scenarios, parameter ranges,
                                # Meloche A_ca estimates, git hash

    trigger_locations.geojson   # GeoJSON FeatureCollection — one point
                                # per trigger location with properties:
                                #   cluster_id, sk38, tau_p, rank

    scenarios/
        scenario_001/           # one directory per com1DFA run
            release.geojson     # release polygon (single Feature)
                                #   properties: trigger_cluster, A_ca,
                                #               depth_percentile,
                                #               size_factor, probability
            depth.tif           # release depth raster (float32, m)
                                #   same CRS and resolution as DEM
                                #   NaN outside release polygon
            density.json        # {"mean": 285.0, "std": 12.0}  kg/m³
            params.json         # μ, ξ, rho, release_area_m2,
                                #   mean_depth_m, total_volume_m3
        scenario_002/
            ...

    scenario_weights.json       # {scenario_id: weight} — probability
                                # of each scenario summing to 1.0
                                # based on Sk38 rank + size probability

    summary.csv                 # one row per scenario:
                                #   scenario_id, trigger_cluster, sk38_rank,
                                #   A_ca_m, size_factor, depth_pct,
                                #   release_area_m2, mean_depth_m,
                                #   total_volume_m3, weight
```

## Release polygon generation logic

```python
def _make_release_polygon(
        trigger_cluster_id: int,
        A_ca: float,            # crack arrest length (m) from Meloche
        size_factor: float,     # 1.0 = median estimate
        cluster_map, dem, transform,
        start_zone_mask) -> shapely.Polygon:
    """
    1. Find trigger cluster centroid (x, y) in UTM
    2. Estimate release dimensions:
         width   = A_ca * size_factor           # upslope dimension
         breadth = A_ca * size_factor * aspect  # cross-slope
                   where aspect = start_zone_width / A_ca (capped at 2.0)
    3. Orient rectangle along slope aspect (from DEM gradient at centroid)
    4. Intersect with start_zone_mask — ensures polygon stays on real
       avalanche terrain regardless of size perturbation
    5. Return as shapely Polygon in UTM coords
    """
```

## Depth raster generation logic

```python
def _make_depth_raster(
        release_polygon: shapely.Polygon,
        cluster_map: np.ndarray,
        hs_values: dict,        # cluster_id -> HS (cm) at snapshot
        depth_percentile: int,  # 10, 50, or 90
        transform) -> np.ndarray:
    """
    For each pixel inside release_polygon:
      - Look up its cluster_id from cluster_map
      - Get that cluster's HS from SNOWPACK at snapshot_date
      - Multiply by depth_percentile factor:
          P10 factor = 0.75 (conservative thin slab)
          P50 factor = 1.00 (best estimate)
          P90 factor = 1.30 (deep slab scenario)
      - Convert cm -> m
    Pixels outside release_polygon = NaN
    """
```

## Scenario probability weights

```python
# Weight each scenario by two factors:
#   w_trigger: inversely proportional to Sk38 rank
#              (weakest location most likely trigger)
#   w_size:    from log-normal distribution on A_ca
#              (median most probable, tails less likely)
#
# w_total = w_trigger * w_size, normalized to sum to 1.0
#
# Example for 5 triggers × 5 sizes = 25 scenarios:
#   trigger weights: [0.35, 0.25, 0.20, 0.12, 0.08]  (rank-inverse)
#   size weights:    [0.05, 0.20, 0.50, 0.20, 0.05]  (log-normal P10-P90)
```

## Notes on com1DFA interface

AvaFrame com1DFA expects:
  - Release area: shapefile or GeoJSON polygon in the project CRS
  - Release depth: either uniform scalar OR a raster (same grid as DEM)
    → raster mode requires AvaFrame >= 1.5; check version before building
  - DEM: .asc or GeoTIFF in the project CRS
  - Simulation config: .ini file with μ, ξ, density, entrainment flags

The `scenarios/scenario_NNN/` output is designed so that a thin AvaFrame
wrapper can iterate over subdirectories and call com1DFA once per scenario
without needing to know about the upstream pipeline.

## Open questions before implementing

1. Road polygon — need US-6 milepost geometry for P(exceedance)
2. AvaFrame version on the cluster — raster depth input availability
3. Voellmy parameters for Little Professor — calibrated or default?
4. Whether release polygon should be constrained to the observed Jan 18
   release area boundary for validation, or free to vary
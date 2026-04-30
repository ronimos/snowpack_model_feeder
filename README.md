# Little Professor — UAS-Driven Avalanche Model Chain

**From Snow Depth to Runout Probability: A UAS-Driven Model Chain for Highway Avalanche Operations**
Ron, Ryan, Valerie, Snook et al.

Demonstrated on the 18 January 2026 D2 skier-triggered slab avalanche at Little Professor,
Loveland Pass (US-6), Colorado. End-to-end pipeline: UAS snow depth survey → spatially
distributed SNOWPACK → release area geometry (Meloche et al. 2025) → com1DFA runout
probability envelopes.

---

## Quick Start

```bash
# Full rebuild (new DEM, first setup, or major config change)
./run_full_pipeline.sh

# Operational update (after a new UAS survey)
./run_operational.sh --snapshot 2026-01-17
```

---

## Repository Layout

```
src/snowpack-model-feeder/
    forcing_pipeline.py       # Forcing pipeline: UAS surveys → SMET files
    analysis_pipeline.py      # Analysis pipeline: Zarr → release geometry → scenarios
    config.py                 # ProjectConfig dataclass — all paths and parameters
    spatial_model.py          # WindNinja transport model, RF gap-fill
    smet_writer.py            # SMET read/write, gap-fill logic
    clustering.py             # Cluster generation from DEM + transport features
    avalanche.py              # UAS dHS anomaly → release/deposit boundary detection
    snowpack_analysis.py      # Profile feature extraction, Meloche parameters, groups
    release_geometry.py       # BFS crack propagation, release polygon construction
    probabilistic_release.py  # Probabilistic boundary model (logistic regression)
    fit_boundary_model.py     # Train boundary model from observed events
    scenario_writer.py        # Write AvaFrame com1DFA scenario input files
    plots.py                  # Avalanche boundary + comparison plot functions
    visualize_snowpack.py     # Daily SNOWPACK frame generator + HTML flipbook
    analyze_release_zone.py   # Release zone vs adjacent slope comparison

scripts/
    animate_propagation.py        # BFS crack propagation animation
    animate_instability_evolution.py  # Seasonal Sk38 + Λ evolution animation
    cluster_map_viewer.py         # Interactive Folium cluster map
    plot_cluster_sizes.py         # Cluster group comparison plots

snowpack/
    little_prof/
        config/master_config.ini  # SNOWPACK configuration
        config/template.sno       # Initial conditions template
        run_snowpack.sh           # Run SNOWPACK on all clusters
        build_zarr_chunked.py     # Aggregate .pro output to Zarr
        output/                   # .pro files + Zarr (gitignored)

windninja/
    library/                  # 96-run WindNinja library (16 dir × 6 speeds, 1m)
    generate_wind_library.sh  # Generate library from DEM

data/
    boundaries/
        Little_Proff.kml                  # Survey domain boundary
        Litte_prof_start_zone.kml         # Avalanche start zone
        avalanche_release_area.geojson    # Observed Jan 18 release area
    surveys/                              # Raw UAS GeoTIFFs

outputs/
    resampled_1m/             # 1m HS grids + DEM
    smet/                     # Per-cluster SMET forcing files
    analysis/                 # Cluster map, features, Meloche parameters
    scenarios/                # AvaFrame scenario directories
    models/                   # Trained boundary model (.pkl)
    plots/                    # All generated figures + animations
    logs/                     # Pipeline run logs

run_full_pipeline.sh          # Full end-to-end rebuild
run_operational.sh            # Operational update after new survey
```

---

## Pipeline Overview

The pipeline has three phases, orchestrated by two shell scripts:

### Phase 1: Forcing generation (`forcing_pipeline.py`)

Transforms UAS snow depth surveys + weather station data into per-cluster SMET
forcing files for SNOWPACK.

```bash
python src/snowpack-model-feeder/forcing_pipeline.py <step>
```

| Step | Description |
|------|-------------|
| `resample` | Resample UAS surveys and DEM to 1m grid |
| `transport` | Build WindNinja transport features per cluster |
| `features` | Assemble ML feature matrix (terrain + transport) |
| `train` | Train Random Forest spatial distribution model |
| `avalanche` | Detect avalanche boundaries from dHS anomaly |
| `cluster` | Generate cluster map from DEM + transport |
| `gap_fill` | Fill hourly HS between survey dates |
| `smet` | Write per-cluster SMET forcing files |

Gap-fill modes:

```bash
python src/snowpack-model-feeder/forcing_pipeline.py gap_fill              # observed transport (default, r=0.83)
python src/snowpack-model-feeder/forcing_pipeline.py gap_fill --station-only  # station dHS only
python src/snowpack-model-feeder/forcing_pipeline.py gap_fill --use-model     # RF + WindNinja (r=0.10, not recommended)
```

### Phase 2: SNOWPACK simulation

```bash
bash snowpack/little_prof/run_snowpack.sh
```

Runs SNOWPACK independently at each of ~6,636 cluster locations. On first run,
uses template initial conditions; on reruns, restarts from `_res.sno` files
(incremental simulation). Output `.pro` files are aggregated to a Zarr store
by `build_zarr_chunked.py` (called automatically at end of `run_snowpack.sh`).

### Phase 3: Analysis + scenarios (`analysis_pipeline.py`)

Extracts snowpack features from the Zarr, generates release polygons, and
writes AvaFrame scenario directories.

```bash
python src/snowpack-model-feeder/analysis_pipeline.py <step> [options]
```

| Step | Description |
|------|-------------|
| `analyze` | Extract Meloche features for all start zone clusters |
| `scenarios` | Generate release polygons + depth rasters for com1DFA |

Scenario generation:

```bash
# Single trigger, 5 size factors, 3 depth percentiles = 15 scenarios
python src/snowpack-model-feeder/analysis_pipeline.py scenarios \
    --snapshot-date 2026-01-17 --n-triggers 1 \
    --size-factors 0.70 0.85 1.00 1.15 1.30 \
    --depth-pcts 10 50 90

# Multi-trigger ensemble = 75 scenarios
python src/snowpack-model-feeder/analysis_pipeline.py scenarios \
    --snapshot-date 2026-01-17 --n-triggers 5 \
    --size-factors 0.70 0.85 1.00 1.15 1.30 \
    --depth-pcts 10 50 90

# Validation against observed release area
python src/snowpack-model-feeder/analysis_pipeline.py scenarios \
    --snapshot-date 2026-01-17 --n-triggers 1 \
    --use-observed-release
```

---

## Full vs Operational Pipeline

| | `run_full_pipeline.sh` | `run_operational.sh` |
|---|---|---|
| **When** | New DEM, first setup, config change | After a new UAS survey |
| **Runs** | All forcing + SNOWPACK + analysis | resample → gap_fill → smet → SNOWPACK → analysis |
| **Skips** | — | transport, features, train, avalanche, cluster |
| **SNOWPACK** | Full season from bare ground | Incremental from restart files |
| **Runtime** | ~4–6 hours | ~1–2 hours |

---

## Release Geometry

Two complementary models generate release area polygons:

**Physics-based BFS model** — deterministic flood-fill from a trigger cluster through
the k=8 cluster neighbour graph. Arrest criteria: slab thickness discontinuity
(THICKNESS_JUMP_FACTOR=0.25), elastic length discontinuity (LAMBDA_JUMP_FACTOR=0.5),
stauchwall (slope < 28°), τ_g < 50 Pa, start zone boundary. Calibrated on 75% of
the Jan 18 release boundary, verified on 25%.

**Probabilistic boundary model** — logistic regression trained on labeled cluster-pair
transitions from observed events. Produces P(arrest) at each boundary. Improves with
each new event. Currently trained on Jan 18 only (ROC-AUC 0.64).

See `docs/release_area_geometry.md` for full method documentation.

---

## Scenario Ensemble

The scenario ensemble spans three independent axes:

| Axis | What it changes | CLI flag |
|------|----------------|----------|
| Trigger location | Where the crack nucleates (different Sk38, A_ca, position) | `--n-triggers` |
| Size factor | How far the crack propagates (scales arrest thresholds) | `--size-factors` |
| Depth percentile | How much snow in the release (scales depth raster) | `--depth-pcts` |

Each scenario produces a release polygon + depth raster + params.json for com1DFA.
Scenarios are probability-weighted (inverse Sk38 × log-normal size × P50-dominant depth).

Output structure:

```
outputs/scenarios/YYYY-MM-DD/
    metadata.json
    trigger_locations.geojson
    scenario_weights.json
    summary.csv
    scenarios/
        scenario_001/
            release.geojson       # EPSG:6342
            depth.tif             # float32, NaN outside release
            depth.asc + depth.prj # AvaFrame legacy format
            density.json
            params.json           # μ, ξ, scour_depth_m, scenario_probability
```

---

## SNOWPACK Visualization

```bash
# Fast test (simple WL split)
python src/snowpack-model-feeder/visualize_snowpack.py --end-date 2026-01-20

# Full season with grain-type WL detection
python src/snowpack-model-feeder/visualize_snowpack.py \
    --end-date 2026-03-31 --wl-method grain_type
```

Opens `outputs/plots/daily_frames/index.html` — four-tab flipbook (loading,
stability, propagation, structure). Keyboard: `←→` navigate, `space` play/pause,
`1-4` switch tabs.

---

## Animations

```bash
# BFS crack propagation from trigger cluster
python scripts/animate_propagation.py \
    --snapshot-date 2026-01-17 --trigger-cid 3178

# Seasonal instability evolution (Sk38 + Λ)
python scripts/animate_instability_evolution.py --end-date 2026-01-17

# Interactive cluster map (HTML)
python scripts/cluster_map_viewer.py \
    --cluster-map outputs/analysis/cluster_map.tif
```

---

## Key Parameters

| Parameter | Symbol | Source | Typical range |
|-----------|--------|--------|---------------|
| WL peak shear strength | τ_p | SNOWPACK `shear_strength` (kPa) | 1.5–3.5 kPa |
| WL shear strength gradient | θ | neighbor cluster pairs | 10–50 Pa/m |
| Slab elastic modulus | E | (ρ/300)^2.5 × 4 MPa | 2–6 MPa |
| Slab tensile strength | σ_t | (ρ/300)^1.4 × 5 kPa | 3–7 kPa |
| Elastic length | Λ | √(E'·h·D_wl/G_wl) | 2–5 m |
| Gravitational shear stress | τ_g | ρ·g·h·sinψ·(1−tanϕ/tanψ) | 200–600 Pa |
| Crack arrest length | A_ca | L_t·Π₁·√(σ_t/τ_g) | 50–500 m |
| Voellmy dry friction | μ | **uncalibrated default** | 0.155 |
| Voellmy turbulent friction | ξ | **uncalibrated default** | 1500 m/s² |

Fixed parameters: G_wl = 0.2 MPa, δ = 1, ν = 0.3, ϕ = 27°.

---

## Dependencies

```bash
# Python environment (uv)
cd /home/ron/snowpack_model_feeder
source .venv/bin/activate

# WindNinja (conda — separate environment)
conda activate windninja

# SNOWPACK binary
/home/caic/caic/rtsys/snowpack/exe/snowpack
```

Python packages: xsnow, xarray, numpy, pandas, rasterio, pyproj, scikit-learn,
shapely, scipy, matplotlib, dask, zarr, folium.

---

## Documentation

| Document | Description |
|----------|-------------|
| `docs/release_area_geometry.md` | Release area method: BFS model, probabilistic model, validation |
| `docs/clustering_methods.md` | Clustering algorithm, quality metrics, group comparison |
| `docs/TODO.md` | Prioritized task list |

---

## References

- Meloche et al. (2025). Modeling Crack Arrest in Snow Slab Avalanches — Toward
  Estimating Avalanche Release Sizes. *JGR Earth Surface*, 130, e2025JF008470.
  doi:10.1029/2025JF008470
- Gaume et al. (2015). Influence of weak layer heterogeneity and slab properties
  on slab tensile failure propensity and avalanche release area.
  *The Cryosphere*, 9, 795–804. doi:10.5194/tc-9-795-2015
- Guillet et al. (2023). A Depth-Averaged Material Point Method for Shallow
  Landslides: Applications to Snow Slab Avalanche Release. *JGR Earth Surface*.
- van Herwijnen et al. (2016). Estimating the effective elastic modulus and
  specific fracture energy of snowpack layers from field experiments.
  *Journal of Glaciology*, 62(236).
- Reiweger et al. (2010). Load-controlled test apparatus for snow.
  *Cold Regions Science and Technology*, 62(2-3).
- Perzl (2007). Stauchwall threshold. JRC report.
- Veitinger et al. (2016). Stauchwall / start zone delineation.

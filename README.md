# Little Professor — UAS-Driven SNOWPACK Model Chain

**From Snow Depth to Runout Probability: A UAS-Driven Model Chain for Highway Avalanche Operations**
Ron, Ryan, Valerie, Snook et al.

Demonstrated on the 18 January 2026 D2 skier-triggered slab avalanche at Little Professor,
Loveland Pass (US-6). End-to-end pipeline: UAS snow depth survey → spatially distributed
SNOWPACK → avalanche detection → snowpack structure analysis → crack arrest estimation
(Meloche et al. 2025) → com1DFA runout probability maps.

---

## Repository Layout

```
src/snowpack-model-feeder/
    pipeline.py               # Main pipeline orchestrator (run steps by name)
    config.py                 # ProjectConfig dataclass — all paths and parameters
    spatial_model.py          # WindNinja transport model, RF gap-fill
    smet_writer.py            # SMET read/write, gap-fill logic
    clustering.py             # Cluster generation from DEM + transport features
    avalanche.py              # UAS dHS anomaly → release/deposit boundary detection
    plots.py                  # Avalanche boundary plot functions
    visualize_snowpack.py     # Daily SNOWPACK frame generator + HTML flipbook
    analyze_release_zone.py   # Release zone vs adjacent slope comparison,
                              #   WL/slab decomposition, Meloche et al. (2025) features,
                              #   RF classifier, cross-date test

data/
    boundaries/
        Little_Proff.kml                  # Survey domain boundary
        Litte_prof_start_zone.kml         # Avalanche start zone (release candidates)
        avalanche_release_area.geojson    # Observed Jan 18 release area

windninja/
    library/                  # 96-run WindNinja library (16 dir × 6 speeds, 1m resolution)

outputs/
    smet/                     # Per-cluster SMET forcing files
    plots/                    # Avalanche boundary plots, comparison plots
        daily_frames/         # Per-day SNOWPACK panel PNGs + index.html flipbook
            loading/          # HS, dHS/dt
            stability/        # Sk38, SSI, Sn38
            propagation/      # TG, accumulated TG, stab deformation rate
            structure/        # WL burial, slab thickness, WL shear strength, grain size
    analysis/
        cluster_map.npy               # Cluster ID raster (same grid as DEM)
        release_zone_groups.json      # Cluster IDs per group (release/adjacent/reference)
        meloche_features_YYYY-MM-DD.csv   # Per-cluster Meloche (2025) parameters
        release_zone_features_YYYY-MM-DD.csv  # Per-cluster WL/slab feature table
```

---

## Pipeline Steps

Run steps individually:

```bash
cd /home/ron/snowpack_model_feeder
python src/snowpack-model-feeder/pipeline.py <step>
```

| Step | Description |
|------|-------------|
| `resample` | Resample UAS surveys and DEM to 1m grid |
| `transport` | Build WindNinja transport features per cluster |
| `features` | Assemble ML feature matrix (terrain + transport) |
| `train` | Train Random Forest spatial distribution model |
| `avalanche` | Detect avalanche boundaries from dHS anomaly |
| `cluster` | Generate cluster map from DEM + transport |
| `gap_fill` | Fill missing survey periods (default: observed transport) |
| `smet` | Write per-cluster SMET forcing files |

Gap-fill modes:

```bash
python pipeline.py gap_fill                  # observed transport (default, r=0.83)
python pipeline.py gap_fill --station-only   # station dHS only
python pipeline.py gap_fill --use-model      # RF + WindNinja (r=0.10, not recommended)
```

---

## SNOWPACK

SMET files are written to `outputs/smet/`. Run SNOWPACK:

```bash
cd /home/ron/snowpack/little_prof
bash run_snowpack.sh           # Runs all 1989 clusters × slopes in parallel
```

Output `.pro` files: `/home/ron/snowpack/little_prof/output/cluster_XXXX_cluster_XXXX.pro`

Build Zarr cache (one-time, ~2h, resumable):

```bash
nohup python build_zarr_chunked.py \
    > /home/ron/snowpack/little_prof/build_zarr.log 2>&1 &
```

---

## Avalanche Boundary Detection

```bash
# Default parameters (Jan 14–20 period)
python src/snowpack-model-feeder/avalanche.py

# Custom period
python src/snowpack-model-feeder/avalanche.py \
    --date-before 2026-01-06 --date-after 2026-01-14

# Key tunable parameters
#   --erosion-sigma 1.5    threshold for release zone seeds (default 1.5)
#   --canny-low 0.05       Canny edge lower threshold (default 0.05)
#   --canny-high 0.08      Canny edge upper threshold (default 0.08)
#   --no-noise-mask        disable persistent noise mask
```

Jan 18 detection result: release ~1100 m² / -600 m³, deposit ~3945 m² / 3121 m³.

---

## SNOWPACK Visualization

```bash
# Fast test (simple WL split — bottom 20% of HS)
python src/snowpack-model-feeder/visualize_snowpack.py --end-date 2026-01-20

# Full season with proper grain-type WL detection (run overnight)
python src/snowpack-model-feeder/visualize_snowpack.py \
    --end-date 2026-03-31 --wl-method grain_type
```

Opens `outputs/plots/daily_frames/index.html` — four-tab flipbook:
- **Loading** — HS, dHS/dt
- **Stability** — min Sk38, SSI, Sn38 at WL interface
- **Propagation** — TG, accumulated TG, stab deformation rate
- **Structure** — WL burial depth, slab thickness, WL shear strength, grain size, slab density

All panels show start zone boundary (green) and Jan 18 release area (red).
Keyboard: `←→` navigate, `space` play/pause, `1-4` switch tabs.

---

## Release Zone Analysis (Meloche et al. 2025)

Compares snowpack properties between: (1) release zone clusters, (2) adjacent
slope clusters within start zone, (3) terrain-matched reference clusters.

```bash
# Box plot comparison at Jan 17 snapshot
python src/snowpack-model-feeder/analyze_release_zone.py

# With RF classifier (per-cluster feature importance)
python src/snowpack-model-feeder/analyze_release_zone.py --classifier

# With cross-date test (train on Jan 17, score across season)
python src/snowpack-model-feeder/analyze_release_zone.py \
    --classifier --cross-date-test

# Different snapshot date
python src/snowpack-model-feeder/analyze_release_zone.py \
    --snapshot-date 2026-01-14 --classifier
```

**Jan 17 results (key findings):**
- Release zone has lower WL shear strength gradient θ (~20 Pa/m vs ~28 Pa/m adjacent)
  — more spatially homogeneous weak layer → crack propagates further
- Release zone has higher driving stress τ_g — steeper, thicker slab
- Dimensionless numbers Π₁, Π₂ higher in release zone → longer predicted crack arrest
  distance, consistent with observed D2 size
- θ values match Meloche et al. (2024) field-measured range (15–30 Pa/m) — independent
  validation of distributed SNOWPACK spatial variability

Output files saved to `outputs/analysis/` and `outputs/plots/`.

---

## Key Parameters (Meloche et al. 2025 Framework)

| Parameter | Symbol | Source | Typical range |
|-----------|--------|--------|---------------|
| WL peak shear strength | τ_p | SNOWPACK `shear_strength` (kPa) | 1.5–3.5 kPa |
| WL shear strength gradient | θ | neighbor cluster pairs | 10–50 Pa/m |
| Slab elastic modulus | E | (ρ/300)^2.5 × 4 MPa | 2–6 MPa |
| Slab tensile strength | σ_t | (ρ/300)^1.4 × 5 kPa | 3–7 kPa |
| Characteristic elastic length | Λ | sqrt(E'·h·D_wl/G_wl) | 2–5 m |
| Gravitational shear stress | τ_g | ρ·g·h·sinψ·(1−tanϕ/tanψ) | 200–600 Pa |
| Elastic dimensionless number | Π₁ | τ_g/(θ·Λ·√(1+δ)) | 1–10 |
| Brittle dimensionless number | Π₂ | Π₁·√(σ_t/τ_g) | 3–30 |
| Crack arrest length (brittle) | A_ca | L_t·Π₁·√(σ_t/τ_g) | 50–500 m |

Fixed parameters: G_wl = 0.2 MPa (Reiweger et al. 2010), δ = 1, ν = 0.3, ϕ = 27°.

---

## Dependencies

```bash
# Main environment
cd /home/ron/snowpack_model_feeder
source .venv/bin/activate

# WindNinja
conda activate windninja
WindNinja_cli --help

# SNOWPACK binary
/home/caic/caic/rtsys/snowpack/exe/snowpack
```

Python packages: xsnow, xarray, numpy, pandas, rasterio, pyproj, scikit-learn,
shapely, scipy, matplotlib, dask, zarr.

---

## Data Locations

| Dataset | Path |
|---------|------|
| UAS surveys (resampled) | `outputs/resampled_1m/hs_YYYY-MM-DD.npy` |
| DEM (1m) | `outputs/resampled_1m/dem_1m.tif` |
| SMET forcing files | `outputs/smet/cluster_XXXX_cluster_XXXX.smet` |
| SNOWPACK .pro output | `/home/ron/snowpack/little_prof/output/` |
| Zarr cache | `/home/ron/snowpack/little_prof/output/slope_snowpack.zarr` |
| WindNinja library | `windninja/library/` |
| Cluster map | `outputs/analysis/cluster_map.npy` |

---

## References

- Meloche et al. (2025). Modeling Crack Arrest in Snow Slab Avalanches — Toward
  Estimating Avalanche Release Sizes. *JGR Earth Surface*, 130, e2025JF008470.
  doi:10.1029/2025JF008470
- Guillet et al. (2023). A Depth-Averaged Material Point Method for Shallow
  Landslides: Applications to Snow Slab Avalanche Release. *JGR Earth Surface*.
- van Herwijnen et al. (2016). Estimating the effective elastic modulus and
  specific fracture energy of snowpack layers from field experiments.
  *Journal of Glaciology*, 62(236).
- Reiweger et al. (2010). Load-controlled test apparatus for snow.
  *Cold Regions Science and Technology*, 62(2-3).
  
# Little Professor Pipeline — TODO

**Updated:** 2026-05-15  
**Context:** UAS → distributed SNOWPACK → release geometry → com1DFA runout probability

---

## High Priority (before next operational run)

- [x] **Post-avalanche SNOWPACK reinitialization** — `reinitialize_snowpack.py` implemented with min-kernel release detection and `.sno` layer scouring. Integrated as `step_reinit` in `forcing_pipeline.py`. Tested on Jan 18 event. Remaining: validate post-reinit SNOWPACK output against post-event UAS HS observation.

- [ ] **Spatially distributed scour depth raster** — currently `scour_depth_m` is extracted only at the trigger cluster and applied uniformly across the path. com1DFA supports an entrainable-depth raster where each cell holds its own scour cap. **Raster generation done:** `step_scenarios` now calls `_build_scour_depth_map()` which runs the failure-plane-to-hard-layer scan for every cluster in the domain (using `slab_thickness` from features for start-zone clusters; full-depth scan for path clusters without a defined failure plane) and writes `scour_depth.tif` alongside `depth.tif` in the scenario output directory. **Remaining:** wire `scour_depth.tif` into the com1DFA ini file as the entrainable-depth raster input. **Blocked by:** com1DFA entrainable-depth raster input bug (may be config issue). Revisit once the raster input path is working.

- [ ] **Run observed-release validation for Val (AvaFrame)**
  ```bash
  python src/snowpack-model-feeder/analysis_pipeline.py scenarios \
      --snapshot-date 2026-01-17 --n-triggers 1 \
      --size-factors 0.85 1.00 1.15 --depth-pcts 10 50 90 \
      --use-observed-release
  ```
  Validates flow model independently of release geometry.

- [ ] **Calibrate com1DFA Voellmy μ/ξ** against Jan 18 runout observation.  
  Current values are **uncalibrated defaults**: μ=0.155 (dry friction), ξ=1500 m/s² (turbulent friction). These are flow dynamics parameters independent of SNOWPACK — SNOWPACK provides release conditions (geometry, depth, density), but μ and ξ govern the flowing avalanche's interaction with the terrain surface (path roughness, confinement, entrainment). They cannot be derived from the snowpack state and must be calibrated empirically against observed runout.  
  **Calibration approach:** run com1DFA with the Jan 18 Meloche-derived release polygon (size_factor=1.00, P50 depth) across a μ/ξ grid, e.g. μ ∈ [0.10, 0.15, 0.20, 0.25, 0.30] × ξ ∈ [1000, 1500, 2000, 2500]. Score each combination against observed deposit extent (debris toe position, lateral spread). Select the (μ, ξ) pair that minimizes runout error. Typical D2 dry slab values on open terrain: μ ≈ 0.15–0.30, ξ ≈ 1000–2500 m/s².  
  **Dependency:** requires the observed-release validation run and the US-6 road polygon to also evaluate P(exceedance) sensitivity to μ/ξ.  
  **Note for multi-path extension:** μ/ξ are path-specific. Calibration on Little Professor does not transfer directly to Seven Sisters or Muleshoe — each path needs its own calibration against observed events or must use literature defaults with wider uncertainty bounds.

- [ ] **US-6 road polygon** for P(exceedance) calculation at infrastructure.

- [ ] **Confirm AvaFrame version** supports raster depth input format.

---

## Pipeline Completeness

- [ ] **Couple `scenarios` step to run both physical and probabilistic models** — add `--probabilistic` flag to `analysis_pipeline.py scenarios` that calls `probabilistic_release.py` for each trigger cluster after physical scenarios are written. Produces `release_comparison_YYYY-MM-DD_probability_model.png` alongside `_physical_model.png`.

- [ ] **Probabilistic model → AvaFrame scenario set** — integrate probabilistic release polygons as additional scenarios in the com1DFA ensemble. Weights derived from P(arrest) distributions rather than Sk38-based weights. This propagates release zone uncertainty through to runout probability envelopes, enabling the full probabilistic decision-support chain:
  ```
  P(release boundary) × P(runout | release) = P(exceedance at road)
  ```
  Currently the probabilistic model produces comparison plots only.

- [ ] **`--restrict-to-release-area` validation run** — confirm trigger selection after all filter fixes (Sk38 < 0.5, elevation P75).

- [ ] **Tune size_factor ensemble** against Jan 18 to confirm correct Mel/Obs ratio spread across 0.70–1.30.

- [ ] **Cross-date Meloche analysis** — run scenarios for Jan 10, Jan 14, Jan 17 to confirm Π₁ separation growing toward event. Should show increasing propagation probability as buried WL weakens.

- [ ] **Step 6 cleanup** — remove remaining duplicated logic between diagnostic scripts (`analyze_release_zone.py`, `visualize_snowpack.py`) and modules.

- [ ] **Multi-event SNOWPACK reinitialization** — current shell scripts support a single `--reinit` event. For seasons with multiple avalanches (same or different paths), extend to accept an events JSON file listing all events chronologically. The pipeline would loop: run SNOWPACK to event A → reinit A → run to event B → reinit B → ... → run to end of season. Each reinit scours a different (possibly overlapping) set of clusters. Events JSON format:
  ```json
  [
      {"event_date": "2026-01-18", "date_before": "2026-01-14",
       "date_after": "2026-01-20", "label": "Little Professor D2"},
      {"event_date": "2026-02-05", "date_before": "2026-01-27",
       "date_after": "2026-02-10", "label": "Seven Sisters D1.5"}
  ]
  ```
  Each event needs its own survey pair for detection (or a pre-drawn GeoJSON) and its own snapshot date for slab thickness. Clusters affected by earlier events will already have reduced HS when subsequent reinits run.

---

## Physical Model Improvements

- [ ] **Asymmetric gradient thresholds** — LAMBDA_JUMP_FACTOR and THICKNESS_JUMP_FACTOR are currently symmetric (same threshold for stiffening and softening). Measure actual |ΔΛ/Λ| at slab-thickening boundaries (not just thinning) to determine if separate thresholds are warranted.

- [ ] **Downslope arrest: direct τ_g < τ_p criterion** — replace 28° slope proxy with direct comparison of distributed τ_g and τ_p from SNOWPACK. Makes stauchwall arrest parameter-free and physically explicit. τ_g and τ_p are both already computed.

- [x] **Critical cut length** — ~~expose SNOWPACK variable 0606 (critical crack length) via xsnow~~. Side-loader `add_critical_cut_length.py` reads 0606 directly from `.pro` files and writes `critical_cut_length` into the Zarr. Functional for downstream use. Upstream xsnow API integration (flag to Florian/SLF) still open — see new item below.

- [ ] **Integrate 0606 parsing into `build_zarr_chunked.py`** — currently the side-loader has to run after every Zarr rebuild. Add a `parse_var` pass inside each batch in the builder; concatenate `critical_cut_length` (and likely `0607 rta`) into the xsnow Dataset before write. Removes the need for the side-loader on full rebuilds.

- [ ] **Fix `build_zarr_chunked.py` resume logic** — existing filename-stem skip check (`f.stem.replace('_cluster_', '_')`) doesn't match xsnow's `StationName=` location coord, so every rerun rebuilds from scratch. Read `StationName=` from each `.pro` header and compare against the Zarr's `location` coordinate.

- [x] **Mayer et al. (2022) P_unstable evaluation on Colorado snowpack** — DONE. Pipeline implemented across two envs (main + `.venv-punstable` with sklearn 0.22.1) and documented in `punstable_evaluation_method.md`. Scripts: `extract_punstable_features.py` → `predict_punstable.py` → `ingest_punstable_to_zarr.py`. Spatial validation against Jan 18 release-area vs adjacent skied-but-not-triggered terrain via `validate_punstable_spatial.py` and `plot_punstable_distributions.py`.

  **Summary of findings:**
  - Internal validation: model picks basal FC/DH as the worst layer at 99.9% of clusters (95.6% FC, 4.3% DH) — Colorado continental signature, without being told to look there. Implicitly cross-validates `split_wl_slab()`.
  - Spatial discrimination: median P_unstable_max essentially identical (release 0.349 vs skied 0.340, Δ +0.009); marginal upper-tail separation (p95 Δ +0.083); 2.5× hit-rate lift at P≥0.77 but on tiny absolute counts (6 vs 5 clusters).
  - Mann-Whitney p=3.8e-9 reflects sample size, not operational signal.
  - Conclusion: Mayer P_unstable correctly identifies *what* the weak layer is but cannot reliably tell us *where* on the slope it will fail. Complements the gradient-arrest physics model (which captures spatial discrimination); does not replace it. ISSW story is the dual-model design empirically justified.

- [ ] **`compare_wl_methods.py` column rename** — the column currently labeled "P_unstable" in this script is `stab_deformation_rate` (S′), which saturates at 6.0. Now that actual P_unstable exists in the Zarr (`p_unstable_max`), rename the misleading column and optionally add the real P_unstable as a second comparison panel.

- [ ] **Mayer Pk crust algorithm refinement** — current implementation in `extract_punstable_features.py` uses single-layer MFcr detection rather than Mayer's contiguous-block accumulation. Effect on Colorado snow likely minimal (strong crusts rare); revisit only if a reviewer asks.

- [ ] **Per-cluster slope angle in Pk** — `extract_punstable_features.py` uses a global default (38°) used only for MFcr thickness projection. Pull `SlopeAngle=` from each `.pro` header into a sidecar or add as a Zarr coord. Bounded effect.

- [ ] **Formal smoke test of P_unstable predictions against SLF's `input_example`** — we use sklearn 0.22.1 in an isolated env but on a newer OS / joblib version. Run the published example profile through our pipeline and compare predictions against SLF's reference outputs to close the model-validity loop.

- [ ] **P_unstable follow-ups (low priority, only if ISSW reviewer asks):**
  - Time-evolution of P_unstable_max at trigger cluster vs skied clusters through the season — could reveal temporal discrimination the snapshot test misses.
  - Timestep sensitivity: re-run validation at 20:00 UTC and 22:00 UTC. If signal is stronger at a different snapshot, reconsider temporal-saturation interpretation.
  - The "0.952 out-of-crown" cluster case study — high P_unstable but Ron's SNOWPACK review suggests insufficient slab for crack propagation. Document as a case study of why high P_unstable alone doesn't imply triggering.

- [ ] **BFS propagation at 1 m grid resolution** — test the BFS crack propagation model on the native 1 m DEM grid instead of the cluster neighbor graph, restricted to the area around the observed Jan 18 release area. Currently the BFS operates on ~6,636 clusters (~3 m median diameter); running at 1 m resolution within a cropped domain (~200×200 m around the release) would evaluate arrest criteria at the pixel scale (~64K cells but only ~4K in the cropped area). This would reveal whether cluster-scale smoothing of slab properties masks sharp gradients that control arrest at finer scales, and whether the BFS boundary converges to the same location as the cluster-based result. If boundaries differ meaningfully, the cluster resolution may need tightening (smaller `max_cells_per_cluster`) or the gradient thresholds may need recalibration for the finer grid. This is a diagnostic test — 1 m resolution across the full start zone is not operationally viable (would require ~65K SNOWPACK simulations).

---

## Probabilistic Model *(low priority — need more avalanche events)*

- [ ] **Resolve Λ coefficient sign ambiguity** — `delta_lambda_rel` has unexpected negative coefficient (larger Λ gradient → interior, not boundary). Hypothesized cause: internal crown-to-stauchwall Λ gradient dominates within the release area. Collect 3–5 more events to test whether coefficient stabilizes to positive sign.

- [ ] **Add events to training dataset** — rerun `fit_boundary_model.py` with `--append-pairs` after each new observed event. Target ROC-AUC > 0.70 with 3–5 events.

- [ ] **Calibrate propagate_threshold** — current CLI default is 0.6 (Jan 18 validation used 0.5, producing 4,862 m² vs 3,902 m² observed). As training dataset grows, calibrate threshold to maximize F1 on held-out events. Recommended operational range: 0.5–0.75.

- [x] **~~Platt scaling / isotonic regression~~** — DONE. Isotonic calibration implemented via `CalibratedClassifierCV(method='isotonic', cv=5)` in `fit_boundary_model.py`. The remaining P(arrest) compression (P10=0.058, P90=0.583) is a sample-size limitation (294 boundary pairs), not a calibration defect. Distribution will widen as events accumulate.

- [ ] **Test Bayesian updating framework** — as events accumulate, switch from refitting logistic regression to Bayesian logistic regression (PyMC) to properly track posterior uncertainty on coefficients. The δτ_p coefficient (+8.30) has high uncertainty with n=1 event; credible intervals will tighten with additional data.

---

## Data / Infrastructure

- [ ] **Environment setup and data collection guide** — write a step-by-step guide covering: (1) Python environment setup (which conda/venv environments are needed and why, e.g. `.venv-punstable` for sklearn 0.22.1); (2) required input files by step (DEM, cluster map, survey HS grids, start-zone KML, weather station SMET); (3) what data to collect before running the chain (UAS survey dates and file naming conventions, station IDs, SNOWPACK installation path, AvaFrame/com1DFA version requirements); (4) directory layout expected by `run_full_pipeline.sh`; (5) a minimal worked example going from raw survey to a hazard assessment output. Target audience: a new field team member who has the data in hand but has never run the pipeline.

- [ ] **Min-kernel auto-detection of release area** — automated start zone delineation from terrain and snowpack, reducing dependence on manually drawn KML.

- [ ] **InSAR snow depth integration** — pipeline currently accepts UAS SfM and lidar. Add InSAR-derived snow depth as third input option (noted in ISSW abstract as future capability).

- [ ] **Multi-path extension** — Seven Sisters, Star Mountain, Muleshoe paths beyond Little Professor. Requires per-path start zone KMLs and terrain parameters.

---

## Visualization / Presentation

- [ ] **High-resolution avalanche photo with Google Earth overlay** — acquire a higher-quality version of the Jan 18 avalanche photo (current image is a phone photo from Loveland Ski Area). Georeference the release area, track, and deposit boundaries and export as a KMZ overlay for Google Earth. This would allow interactive comparison of the observed avalanche extent against the BFS/probabilistic model polygons in 3D terrain context, and produce presentation-quality figures showing the model chain output draped on satellite imagery.

---

## Operational Daily Mode

Design and implement a daily-run operational pipeline that provides continuous hazard assessment between UAS surveys and auto-corrects when new survey data arrives.

### Daily forward mode

- [ ] **Daily SMET extension** — append new hourly weather station records to existing SMET files instead of regenerating all ~6,636 files. Only new rows (since last run) get added. Requires tracking the last-written timestamp per file.

- [ ] **Daily SNOWPACK incremental run** — run SNOWPACK forward one day from restart files. Extract stability snapshot (Sk38, SSI, Λ, τ_g) for the current day and generate a daily hazard assessment. Between surveys, the HS spatial distribution stays frozen at the last survey — only the stratigraphy evolves (sintering, TG metamorphism, new snow loading).

- [ ] **Daily scenario refresh** — rerun `step_scenarios` on the current snapshot date to update trigger locations and release polygons as the snowpack evolves. The daily spatial variability is station-driven only (no UAS data), so the release geometry changes are from stratigraphy evolution, not from new spatial HS information.

### Survey correction mode

- [x] **Cluster splitting after every survey** — `cluster_update` (adaptive threshold bisection) now runs automatically in `run_operational.sh` after `gap_fill` and before `smet`, so every new survey triggers a quality check and splits any clusters whose internal HS std exceeds the adaptive threshold. Order: resample → gap_fill → cluster_update → smet → SNOWPACK. `smet` then regenerates SMETs for all clusters including new children.

- [ ] **Back-correction on new survey** — when a new UAS survey arrives, the gap-filled hourly HS between the last two surveys was station-only extrapolation. The new survey reveals the actual spatial transport. Pipeline must: (1) replace gap-filled HS with observed transport for the inter-survey period, (2) regenerate SMETs only from the previous survey date forward, (3) rerun SNOWPACK from that date via restart files, (4) rebuild the Zarr for affected timesteps, (5) rerun analysis/scenarios.

- [ ] **Partial DEM resampling** — only resample the new survey to the 1 m grid. Don't reprocess the existing surveys. Currently `step_resample` regenerates everything.

### Efficiency improvements

- [ ] **SMET append-only writes** — current `step_smet` rewrites all files from scratch. An append mode would read the existing file, find the last timestamp, and write only new rows. Saves I/O and ~10 min per run on 6,636 files.

- [ ] **Zarr append** — add new timesteps to the existing Zarr store instead of rebuilding from all `.pro` files (~2 hours saved). Requires tracking which timesteps are already in the store and appending new ones from the latest `.pro` output. **Important:** on full rebuilds (new DEM, new cluster map), the Zarr must be deleted before rebuilding — stale locations from previous cluster maps cause duplicate entries and silently corrupt feature extraction (doubled layer arrays, wrong slab_thickness). Add `rm -rf slope_snowpack.zarr` to the `--clean` block in `run_full_pipeline.sh`.

- [ ] **Persist Meloche features in Zarr** — currently `compute_meloche_features` runs from scratch every `analyze` call, re-reading the full Zarr snapshot and rerunning KNN + Π₁/Π₂/A_ca for all clusters. Instead, compute the features once during the Zarr build/update step and write them as time-varying Zarr variables (e.g., `theta`, `Pi1_elastic`, `Pi2_brittle`, `A_ca_brittle`, `Lambda`, `E_slab`, `sigma_t` keyed by `[location, time]`). The `analyze` step then reads them directly from the store; `step_scenarios` gets them from the meloche CSV as now, which is populated cheaply from the Zarr slice. Benefits: (1) eliminates repeated KNN computation on every run, (2) makes Meloche features available for time-evolution analysis without re-running `analyze` for each date, (3) enables the Zarr to serve as the single source of truth for all derived snowpack variables. **Dependency:** Zarr append (#above) should land first so the feature write is incremental, not a full rebuild.

- [ ] **Feature caching with change detection** — only recompute Meloche features for clusters where HS changed by more than a threshold (e.g., 5 cm) since the last extraction. Cache the previous feature values and diff against the new Zarr snapshot. For daily forward mode, only a fraction of clusters will change meaningfully. Superseded by "Persist Meloche features in Zarr" if that item lands — persisting to Zarr is the cleaner solution.

- [ ] **Partial SNOWPACK rerun** — when correcting after a new survey, only rerun clusters whose SMET files actually changed (i.e., clusters where the gap-filled HS differs from the survey-corrected HS by more than a threshold). Use the existing `CLUSTERS_FILE` filter in `run_snowpack.sh`.

### NWP forecast mode

- [ ] **3-day forecast with NWP forcing** — extend the daily pipeline to run SNOWPACK 3 days into the future using NWP (numerical weather prediction) forecast data as forcing. Generate scenario ensembles at three snapshot dates: today (T+0), tomorrow (T+1), and day-after-tomorrow (T+2). Each snapshot produces trigger locations, release polygons, and runout probability envelopes, enabling hazard teams to plan mitigation timing and resource allocation before conditions deteriorate.

  Workflow:
  1. Ingest NWP forecast (HRRR, NAM, or GFS) for the next 72 hours — extract T, precip, wind speed/direction, RH at the study site
  2. Convert NWP fields to SMET format and append to existing cluster SMET files (after the last observed timestamp)
  3. Run SNOWPACK incrementally from now to T+72h
  4. Extract stability snapshots at T+0, T+1, T+2
  5. Run `step_scenarios` at each snapshot → 3 sets of release polygons + runout envelopes
  6. Generate a forecast summary: which paths are approaching critical instability, when, and with what confidence

  Key considerations:
  - NWP forcing has spatial resolution of 1-3 km vs the 1 m pipeline grid — needs downscaling, at minimum lapse-rate correction for temperature and orographic enhancement for precipitation
  - Wind fields from NWP are coarse — WindNinja could re-downscale NWP winds to 1 m, but this adds compute time. For forecast mode, station wind + NWP trend may be sufficient
  - **Rolling forecast convergence:** each day the forecast refreshes with the latest AWS observations and newest NWP run. Today's T+0 replaces yesterday's T+1 with 24h of real data, so forecasts converge toward reality as the target date approaches. Yesterday's T+2 (48h NWP) → today's T+1 (24h observed + 24h NWP) → tomorrow's T+0 (mostly observed). Archive the daily forecast snapshots to track convergence and calibrate forecast skill over the season
  - **Uncertainty scaling with lead time:** widen size_factor range at longer lead times to reflect decreasing forecast confidence. E.g.:
    - T+0: size_factors [0.85, 1.00, 1.15] (observed forcing, tight bounds)
    - T+1: size_factors [0.70, 0.85, 1.00, 1.15, 1.30] (mixed forcing)
    - T+2: size_factors [0.55, 0.70, 0.85, 1.00, 1.15, 1.30, 1.45] (NWP-dominated, wide bounds)
    This naturally produces wider runout envelopes at longer lead times — the hazard team sees "could reach the road at T+2" narrowing to "won't reach the road at T+0" as observations replace NWP
  - When observed data arrives (next station update or next survey), the forecast branch gets pruned and replaced with observed forcing — same back-correction workflow as survey correction mode
  - NWP data source needs to be automated (cron job pulling from NOMADS/AWS)

### Cluster management

- [ ] **Initial clustering strategy: native survey points vs. 1 m resampled grid** — decide whether to cluster on the original (irregular) UAS survey point cloud or to first resample to the 1 m DEM grid and then cluster. Both paths produce the same downstream cluster-map raster, but the spatial fidelity and compute cost differ meaningfully.

  **Option A — cluster from native survey points:**
  - **Pros:**
    - Preserves native spatial variability nuances: sharp wind-scoured ridges, lee-deposit pockets, and cornice edges are represented at survey resolution without interpolation blur
    - No interpolation artifacts — avoids smearing abrupt HS gradients across under-sampled gaps
    - Fewer input points (survey spacing typically 5–10 m equivalent density) means faster PCA + MiniBatchKMeans
    - PCA captures the true measured variance structure, not the variance of an interpolated surface
  - **Cons:**
    - Irregular point density biases cluster centroids toward densely sampled flight strips; windward ridges with sparse returns can be under-represented
    - Survey-to-survey comparisons are harder if flight-line density or point spacing varies between flights — the sample the clusterer sees changes even if the snowpack did not
    - Cluster boundary rasterization back to 1 m grid still requires nearest-neighbour assignment, so the final raster is not free of interpolation
    - Cannot directly use pixel-count weighting for SMET gap-fill without an intermediate density correction

  **Option B — resample to 1 m grid first, then cluster:**
  - **Pros:**
    - Regular grid aligns directly with DEM, all downstream 1 m raster products (depth.tif, entrainable depth, BFS map), and com1DFA input layers — no re-registration step
    - Pixel-count weighting for SMET gap-fill is unambiguous (each cell = 1 m²)
    - Consistent input geometry across surveys regardless of flight-line density or SfM point density — clustering sees the same spatial structure every time
    - Cluster map is immediately usable as a labeled raster mask; no secondary rasterization needed
    - Enables BFS crack propagation at 1 m resolution on the same grid without remapping
  - **Cons:**
    - Interpolation (IDW, kriging, or nearest-neighbour) smears sharp HS gradients — wind-slab / soft-slab transitions may be blurred, weakening the cluster-feature signal that drives PCA separation
    - Resampling a 5–10 m-equivalent survey to 1 m oversamples by 25–100×, inflating spatial autocorrelation and artificially increasing the apparent cluster count needed to explain the variance; can lead to over-clustering of essentially uniform areas
    - Substantially higher compute: clustering ~65 K cells (1 m grid, 250×260 m domain) vs. clustering ~3–6 K survey-representative points
    - Interpolated values in survey gaps carry fabricated uncertainty — the cluster model treats them as real observations

  **Current status:** pipeline uses the 1 m resampled grid (Option B) via `step_resample` → `step_cluster`. The primary open question is whether interpolation blur is masking gradient features (particularly at wind-scoured ridge crests) that would drive better cluster separation under Option A. Worth a side-by-side test on the Jan 17 survey: run both options with the same `n_clusters`, compare cluster boundary locations against the observed Jan 18 release perimeter.

- [ ] **Mid-season cluster splitting with inheritance** — when a new survey reveals that a cluster's internal HS variability exceeds the adaptive threshold (see below), split via MiniBatchKMeans bisection. One child inherits the parent's identity: same cluster ID, `.sno` restart file, `.smet`, `.pro` history, and Zarr entries — continues seamlessly. The other child is new and needs:
  1. A copy of the parent's `.sno` at the split timestamp (same stratigraphy — they were one cluster until the survey revealed heterogeneity)
  2. A new SMET generated from gap-filled hourly HS for the new child's pixel membership
  3. SNOWPACK run from the split date forward (incremental from the copied `.sno`)

  **Adaptive split threshold** — fixed `max_cluster_std_m` is too aggressive in early season (when HS is shallow, small absolute std matters more) and too permissive in deep snow. Use a relative threshold with floor and ceiling:
  ```
  threshold = clip(rel_frac × median_HS, min=min_abs, max=max_abs)
  ```
  Suggested starting values: `rel_frac=0.10`, `min_abs=0.03 m` (noise floor — UAV SfM has ~2–3 cm registration error), `max_abs=0.12 m`. At HS=30 cm → 3 cm threshold; HS=80 cm → 8 cm (same as current fixed); HS=200 cm → 12 cm (saturates). These are uncalibrated — tune against actual survey variability.

  Implementation considerations:
  - Child identity rule: larger pixel-count child inherits parent ID; smaller child gets `max_existing_id + 1`. If equal size, closest-centroid inherits.
  - Cluster map raster updated in-place; neighbour graph rebuilt only for affected clusters
  - Downstream consumers (release geometry, boundary model) pick up new cluster map automatically
  - Track split lineage in `cluster_splits.json`: `{new_id: {parent_id, split_date, split_survey_path, parent_pixels, child_pixels}}`
  - Guard against infinite splitting: if a child would fall below `min_cluster_size` (currently 4 pixels), abort the split for that cluster

- [ ] **Cluster quality monitoring** — at each new survey, compute per-cluster HS std using the adaptive threshold above. Report: how many clusters exceed threshold (split candidates), current total cluster count, mean/median cluster size, size distribution histogram. Log to `outputs/analysis/cluster_quality_YYYY-MM-DD.json`. Flag if total cluster count has grown by >50% since season start (signals aggressive splitting or poor initial clustering — tighten initial parameters next season).

- [ ] **Mid-season cluster merging** — when two neighboring clusters have converged to similar HS trajectories across all surveys to date, merge them into one. This is the counterpart to splitting and can reduce cluster count bloat from aggressive early-season splitting or from surveys that revealed heterogeneity that later homogenized (e.g., large uniform snowfall event).

  **Physical justification for mid-season merging:** HS is the *only* cluster-varying SNOWPACK input (temperature, wind, radiation, new snow all come from the shared weather station). Therefore HS trajectory similarity across the full seasonal vector genuinely implies SNOWPACK stratigraphy similarity — not just current-snapshot similarity. Two clusters that are merge candidates have been receiving nearly identical forcing throughout the season, so their `.sno` stratigraphies should be close.

  **Merge trigger — what to compare:**
  - Primary: similarity of full seasonal HS vectors in PCA space (same feature space used for initial clustering). Require distance below a merge threshold in the *original* high-dimensional space (not just the reduced PCA space — PCA compression can mask early-season differences that still matter).
  - Secondary / validation: compare `.sno` files directly. Key variables to diff: layer count, `grain_type` sequence, `shear_strength` profile, `density` profile, `temperature` profile at the WL depth. If `.sno` diff exceeds a tolerance on any of these, don't merge even if HS vectors look similar. This guards against cases where PCA compressed away a meaningful early-season divergence. Exact tolerances TBD — need to examine actual `.sno` output format and decide which variables are load-bearing for stability metrics.
  - Both criteria must pass: HS trajectory AND `.sno` profile similarity.

  **Merge procedure:**
  1. Pick the larger cluster as the "survivor" (inherits its ID)
  2. Reassign all pixels of the smaller cluster to the survivor's cluster map entry
  3. Generate merged SMET: pixel-membership-weighted average of both clusters' HS histories for all surveys to date, then rerun gap-fill for the merged pixel set
  4. Rerun SNOWPACK from season start to current date using the merged SMET (in background — see below). Since HS trajectories were similar, the merged stratigraphy should closely match either original. The survivor's `.sno` can be used as a reference to validate the rerun output before swapping.
  5. Atomic swap when background rerun completes: replace survivor's `.sno` with merged output, update cluster map, rebuild neighbour graph, update `cluster_splits.json` lineage.
  6. The smaller cluster's ID is retired (mark in lineage log, never reuse).

  **Conservative merge criteria:** merge only if the merge would reduce cluster count (don't merge if it would trigger an immediate re-split). Require similarity across the full seasonal vector, not just recent surveys — early-season divergences that later converged are still reflected in the stratigraphy. A cluster pair that diverged in December and converged in February should not be merged: their WL stratigraphy from December is different, and that's exactly what stability metrics depend on.

- [ ] **Background reclustering** — periodic full recluster + SNOWPACK rerun from scratch, run as a background process so it doesn't block the operational forecast pipeline. When complete, swap the cluster map and all associated files atomically. Forecasts continue running on the old cluster map until the swap.

  **Physical justification:** SNOWPACK is deterministic given the same SMET input — a from-scratch run from season start to the current date produces the same output as incremental updates to the same date, at higher compute cost. So a background full recluster sacrifices no accuracy vs. the existing incremental approach, and can fix problems that splits and merges can't: (1) initial clustering that was poorly positioned relative to the season's actual HS gradients, (2) accumulated cluster count bloat from many splits.

  **Trigger conditions (any one suffices):**
  - Total cluster count has grown by >50% since season start
  - Mean cluster size has fallen below `min_cluster_size × 2` (clusters too small for reliable SNOWPACK)
  - Cluster quality monitoring shows >20% of clusters exceeding the adaptive split threshold for two consecutive surveys (systematic degradation, not isolated splits)
  - Complete melt-out event (HS ≈ 0 across the domain) — stratigraphy is reset, nothing to lose
  - Explicit operator trigger (e.g., `run_full_pipeline.sh --recluster`)

  **What the background process does:**
  1. Re-run initial clustering on the full HS matrix (all surveys to date) with the same PCA + MiniBatchKMeans pipeline
  2. Generate new SMETs for all new clusters (pixel-membership-weighted HS histories)
  3. Run SNOWPACK from season start to current date for all new clusters (this is the expensive step — same wall time as initial pipeline, ~2 hrs)
  4. Validate output: check that new cluster map stability metrics (median Sk38, Pi1) are consistent with old map within some tolerance. If validation fails, abort swap and alert.
  5. Atomic swap: replace `cluster_map.npy`, all `.sno`/`.smet`/`.pro` files, Zarr store. Archive old files with a timestamp suffix.
  6. Update `cluster_splits.json` to mark the recluster event (all lineage history from before the recluster is archived but no longer active).

  **Timing:** prefer low-hazard periods or overnight. If the server is idle (no active SNOWPACK runs, no forecast in progress), the background recluster can use all available cores. Implement a lock file so the operational pipeline and background recluster don't run SNOWPACK simultaneously on the same slope directory.
# Little Professor Pipeline — TODO

**Updated:** 2026-04-20  
**Context:** UAS → distributed SNOWPACK → release geometry → com1DFA runout probability

---

## High Priority (before next operational run)

- [x] **Post-avalanche SNOWPACK reinitialization** — `reinitialize_snowpack.py` implemented with min-kernel release detection and `.sno` layer scouring. Integrated as `step_reinit` in `forcing_pipeline.py`. Tested on Jan 18 event. Remaining: validate post-reinit SNOWPACK output against post-event UAS HS observation.

- [ ] **Spatially distributed scour depth raster** — currently `scour_depth_m` is extracted only at the trigger cluster and applied uniformly across the path. com1DFA supports an entrainable-depth raster where each cell holds its own scour cap. To build this: run the failure-plane-to-hard-layer scan (same logic as the trigger extraction in `step_scenarios`) for every cluster in the domain, paint the result onto the 1 m grid, and write as a GeoTIFF alongside `depth.tif`. This would capture spatial variability in how much old snow the avalanche can entrain — e.g., thinner entrainable layer on wind-scoured ridges, thicker in sheltered terrain. **Blocked by:** com1DFA entrainable-depth raster input bug (may be config issue). Revisit once the raster input path is working.

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

- [ ] **Critical cut length** — expose SNOWPACK variable 0606 (critical crack length) via xsnow. Flag to Florian at SLF for xsnow API.

- [ ] **P_unstable-based weak layer detection** — add SNOWPACK's `P_unstable` (profile instability index) as an alternative method for identifying the layer of concern, alongside the current grain-type-based basal WL detection (`split_wl_slab()` scanning upward through FC/DH codes). The current approach always targets the basal persistent WL, which is correct for the Jan 18 event but may miss mid-pack WLs that become the primary failure plane in different loading scenarios. `P_unstable` identifies the layer with the highest instability at each timestep regardless of position in the profile, which could capture: (1) mid-pack facet layers that become critical after burial by a storm slab, (2) transitions between basal and near-surface WL dominance as the season progresses, (3) cases where the most unstable interface is not the deepest FC/DH layer. Implementation: extract the layer index and depth of the max `P_unstable` value per cluster, use that as the WL/slab interface for `profile_features()` and `compute_meloche_features()`, compare Meloche parameters (Λ, A_ca, Π₁) between the two WL detection methods.

- [ ] **BFS propagation at 1 m grid resolution** — test the BFS crack propagation model on the native 1 m DEM grid instead of the cluster neighbor graph, restricted to the area around the observed Jan 18 release area. Currently the BFS operates on ~6,636 clusters (~3 m median diameter); running at 1 m resolution within a cropped domain (~200×200 m around the release) would evaluate arrest criteria at the pixel scale (~64K cells but only ~4K in the cropped area). This would reveal whether cluster-scale smoothing of slab properties masks sharp gradients that control arrest at finer scales, and whether the BFS boundary converges to the same location as the cluster-based result. If boundaries differ meaningfully, the cluster resolution may need tightening (smaller `max_cells_per_cluster`) or the gradient thresholds may need recalibration for the finer grid. This is a diagnostic test — 1 m resolution across the full start zone is not operationally viable (would require ~65K SNOWPACK simulations).

---

## Probabilistic Model

- [ ] **Resolve Λ coefficient sign ambiguity** — `delta_lambda_rel` has unexpected negative coefficient (larger Λ gradient → interior, not boundary). Hypothesized cause: internal crown-to-stauchwall Λ gradient dominates within the release area. Collect 3–5 more events to test whether coefficient stabilizes to positive sign.

- [ ] **Add events to training dataset** — rerun `fit_boundary_model.py` with `--append-pairs` after each new observed event. Target ROC-AUC > 0.70 with 3–5 events.

- [ ] **Calibrate propagate_threshold** — current CLI default is 0.6 (Jan 18 validation used 0.5, producing 4,862 m² vs 3,902 m² observed). As training dataset grows, calibrate threshold to maximize F1 on held-out events. Recommended operational range: 0.5–0.75.

- [x] **~~Platt scaling / isotonic regression~~** — DONE. Isotonic calibration implemented via `CalibratedClassifierCV(method='isotonic', cv=5)` in `fit_boundary_model.py`. The remaining P(arrest) compression (P10=0.058, P90=0.583) is a sample-size limitation (294 boundary pairs), not a calibration defect. Distribution will widen as events accumulate.

- [ ] **Test Bayesian updating framework** — as events accumulate, switch from refitting logistic regression to Bayesian logistic regression (PyMC) to properly track posterior uncertainty on coefficients. The δτ_p coefficient (+8.30) has high uncertainty with n=1 event; credible intervals will tighten with additional data.

---

## Data / Infrastructure

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

- [ ] **Back-correction on new survey** — when a new UAS survey arrives, the gap-filled hourly HS between the last two surveys was station-only extrapolation. The new survey reveals the actual spatial transport. Pipeline must: (1) replace gap-filled HS with observed transport for the inter-survey period, (2) regenerate SMETs only from the previous survey date forward, (3) rerun SNOWPACK from that date via restart files, (4) rebuild the Zarr for affected timesteps, (5) rerun analysis/scenarios.

- [ ] **Partial DEM resampling** — only resample the new survey to the 1 m grid. Don't reprocess the existing surveys. Currently `step_resample` regenerates everything.

### Efficiency improvements

- [ ] **SMET append-only writes** — current `step_smet` rewrites all files from scratch. An append mode would read the existing file, find the last timestamp, and write only new rows. Saves I/O and ~10 min per run on 6,636 files.

- [ ] **Zarr append** — add new timesteps to the existing Zarr store instead of rebuilding from all `.pro` files (~2 hours saved). Requires tracking which timesteps are already in the store and appending new ones from the latest `.pro` output.

- [ ] **Feature caching with change detection** — only recompute Meloche features for clusters where HS changed by more than a threshold (e.g., 5 cm) since the last extraction. Cache the previous feature values and diff against the new Zarr snapshot. For daily forward mode, only a fraction of clusters will change meaningfully.

- [ ] **Partial SNOWPACK rerun** — when correcting after a new survey, only rerun clusters whose SMET files actually changed (i.e., clusters where the gap-filled HS differs from the survey-corrected HS by more than a threshold). Use the existing `CLUSTERS_FILE` filter in `run_snowpack.sh`.

### Cluster management

- [ ] **Mid-season cluster splitting with inheritance** — when a new survey reveals that a cluster's internal HS variability exceeds `max_cluster_std_m`, the cluster is split via MiniBatchKMeans bisection (same as initial clustering). One child inherits the parent's identity: same cluster ID, `.sno` restart file, `.smet`, `.pro` history, and Zarr entries. Nothing changes for this child — it continues seamlessly. The other child is new and needs:
  1. A copy of the parent's `.sno` at the split timestamp (same stratigraphy — they were one cluster until the survey showed they're different)
  2. A new SMET generated from the gap-filled hourly HS for the new child's pixel membership
  3. SNOWPACK run from the split date forward (incremental from the copied `.sno`)
  
  The secondary child starts with the correct stratigraphy and only diverges from the split point. This is cheaper than rerunning from season start and physically justified — the snowpack was identical until the survey revealed heterogeneity.
  
  Implementation considerations:
  - Cluster IDs for new children must not collide with existing IDs (use max_existing_id + 1)
  - The cluster map raster needs updating (remap parent pixels to two children)
  - The neighbour graph (k=8) needs rebuilding for affected clusters
  - Downstream consumers (release geometry, boundary model) pick up the new cluster map automatically
  - Track split lineage in a `cluster_splits.json` for provenance

- [ ] **Cluster quality monitoring** — track intra-cluster HS variability at each new survey. Flag clusters exceeding `max_cluster_std_m` (8 cm) as split candidates. Report statistics: how many clusters need splitting, where on the slope, and how much the splitting would change the cluster count. This provides a data-driven trigger for when splitting is needed rather than arbitrary re-clustering.

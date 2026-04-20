# Little Professor Pipeline — TODO

**Updated:** 2026-04-20  
**Context:** UAS → distributed SNOWPACK → release geometry → com1DFA runout probability

---

## High Priority (before next operational run)

- [ ] **Post-avalanche SNOWPACK reinitialization** — reset release clusters (cid=3178 area) to bare ground for dates after Jan 18. Currently all forecast dates treat the release zone as undisturbed.

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

---

## Physical Model Improvements

- [ ] **Asymmetric gradient thresholds** — LAMBDA_JUMP_FACTOR and THICKNESS_JUMP_FACTOR are currently symmetric (same threshold for stiffening and softening). Measure actual |ΔΛ/Λ| at slab-thickening boundaries (not just thinning) to determine if separate thresholds are warranted.

- [ ] **Downslope arrest: direct τ_g < τ_p criterion** — replace 28° slope proxy with direct comparison of distributed τ_g and τ_p from SNOWPACK. Makes stauchwall arrest parameter-free and physically explicit. τ_g and τ_p are both already computed.

- [ ] **Critical cut length** — expose SNOWPACK variable 0606 (critical crack length) via xsnow. Flag to Florian at SLF for xsnow API.

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

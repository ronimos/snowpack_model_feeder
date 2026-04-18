# Little Professor / Highway Avalanche Operations — Project TODO

## Project Context

End-to-end model chain: UAS snow depth -> distributed SNOWPACK -> com1DFA
runout probability maps for Colorado highway corridors. Demonstrated on the
18 January 2026 D2 skier-triggered avalanche at Little Professor, Loveland
Pass (US-6). Target: fully automated daily pipeline requiring no manual
intervention.

Reference: "From Snow Depth to Runout Probability: A UAS-Driven Model Chain
for Highway Avalanche Operations" — Ron, Ryan, Valerie, Snook et al.

---

## Priority 1 — SNOWPACK -> Release Scenarios -> com1DFA Interface

### P(release) and release size from SNOWPACK

**A. Stability index approach**

- `Sk38` (<1.0), `SSI` (<1.5), `Sn38` (<1.0) — point stability indices.
- Critical cut length `rc`: SNOWPACK variable 0606 (`nElems`). Not yet mapped
  by xsnow. Add `'0606': 'critical_cut_length'` and flag to Florian.

- [ ] **Jan 18 retrospective** — time series of Sk38/SSI/Sn38 in crown-zone
      clusters Nov 26 through Jan 18. Which combination first crossed threshold?
      Was spatial coherence growing before the event?

- [ ] **Critical cut length from SNOWPACK** — add to xsnow variable mapping,
      add to `visualize_snowpack.py` propagation panel.

**B. Monte Carlo / MCMC approach**
- Sample slab density ±20%, WL shear strength ±1 sigma, slab thickness from
  cluster HS distribution.

**C. DA-MPM / Meloche et al. (2025) — IN PROGRESS**
- Meloche et al. (2025, JGR Earth Surface, doi:10.1029/2025JF008470)
- [x] **Per-cluster Meloche parameters computed** in `analyze_release_zone.py`:
      E_slab (van Herwijnen 2016 fit), σ_t, Λ (characteristic elastic length),
      K_wl (WL stiffness). Power law: E=(ρ/300)^2.5 × 4 MPa, σ_t=(ρ/300)^1.4 × 5 kPa.
- [x] **θ (WL shear strength gradient) computed** from k=6 nearest neighbor
      cluster centroids. Units: Pa/m (wl_shear_strength is kPa, converted ×1000).
- [x] **Dimensionless numbers Π₁, Π₂ and crack arrest length A_ca computed**
      (Eqs. 19, 20 from paper). Saved to `outputs/analysis/meloche_features_YYYY-MM-DD.csv`.
- [x] **Meloche comparison plot** — box plots of θ, τ_g, Λ, Π₁, Π₂, A_ca, L_t,
      slope angle by group (release / adjacent / reference).
- [x] **Jan 17 results** — release zone shows lower θ (~20 Pa/m vs ~28 Pa/m
      adjacent), higher τ_g, higher Π₁/Π₂ — predicting longer crack / larger
      release. Physically consistent with observed D2. θ range matches
      Meloche et al. (2024) field-measured values (15-30 Pa/m) — validation.

- [ ] **A_ca magnitude validation** — brittle A_ca estimate for release zone
      ~250m after unit fix. Compare to actual Jan 18 release zone dimensions.
      If within factor 2, this is a strong paper result.

- [ ] **Cross-date Meloche analysis** — run at Jan 10, Jan 14, Jan 17 and
      compare whether Π₁ separation between release/adjacent grows toward
      event. Would show the instability signal developing spatially.

- [ ] **Release geometry: cross-slope propagation from Gaume et al. (2015)**
      Gaume et al. (2015, The Cryosphere, doi:10.5194/tc-9-795-2015):
      'Influence of weak layer heterogeneity and slab properties on slab
      tensile failure propensity and avalanche release area.' Use their
      framework alongside DEM-derived slope angle to estimate cross-slope
      release width. The slope angle map from the DEM constrains lateral
      extent — terrain that steepens cross-slope acts as a natural arrest
      boundary. Combine with Meloche A_ca (upslope) to define a 2D release
      polygon rather than relying solely on the observed Jan 18 boundary.

- [ ] **Stauchwall location from slope angle threshold (~28°)**
      Multiple sources converge on ~28° as the lower bound for slab release
      terrain (Perzl 2007 JRC; Swiss ALIP/PRA; Maggioni & Gruber; Bühler
      et al.; Veitinger et al. 2016). Use the DEM slope raster to find where
      slope angle drops below 28° downslope of the trigger point — this
      approximates the stauchwall location and sets the downslope release
      polygon boundary. Intersection of: (a) Meloche A_ca upslope limit,
      (b) 28° slope threshold downslope limit, (c) start zone KML lateral
      bounds, (d) cross-slope Gaume heterogeneity constraint → defines a
      physically-motivated release polygon without requiring the observed
      event boundary as input.

- [ ] **`step_scenarios()` in `pipeline.py`** — standardized output schema
      feeding com1DFA: release_zones.geojson, scenario depth rasters (P10-P90),
      scenario_weights.json, weak_layer_params.json, mcmc_bounds.json.

- [ ] **`identify_release_zones()`** — candidate zones from stability fields
      alone (no survey pair), usable any day.

- [ ] **Validation against Jan 18** — run scenario generation for Jan 17,
      compare P50 release area and depth against field observations.

---

## Priority 2 — Multi-Slope / Off-the-Shelf Architecture

- [x] **`start_zone_kml` and `avalanche_release_area.geojson` in `ProjectConfig`**.

- [ ] **`slopes.yaml` slope registry** — DEM, boundary KML, start zone KML,
      AWS station IDs, highway corridor, milepost range per slope.

- [ ] **Remove hardcoded station IDs** from `sql_util.py` and `smet_writer.py`.

- [ ] **Multi-slope runner** and standardized GeoJSON output schema.

- [ ] **Config validation** — `cfg.validate()` checks paths and SQL connectivity.

---

## Priority 3 — Daily Survey Operational Mode

- [ ] **Daily mode gap-fill** — `--daily-mode` flag skips RF when consecutive
      surveys are <36h apart.

- [ ] **Incremental SNOWPACK updates** — append to SMET, restart from `_res.sno`.

- [ ] **Cron-compatible pipeline wrapper**.

- [ ] **Post-avalanche SNOWPACK reinitialization** — after a detected
      avalanche event, reset the snowpack within the release area to
      bare ground (bed surface height). SNOWPACK restarts from a `.sno`
      file; clusters inside the detected release polygon get their `.sno`
      rewritten with HS set to the bed surface elevation and all layers
      cleared. This is critical for operational accuracy: without it,
      SNOWPACK accumulates a phantom slab in cells that actually released,
      producing incorrect stability indices in subsequent timesteps.
      Implementation: detect event from `avalanche.py`, identify affected
      cluster IDs via release polygon overlap with `cluster_map.npy`,
      rewrite their `.sno` files, rerun SNOWPACK forward from event date.

---

## Priority 4 — Transport Model

- [x] **WindNinja v3.13 installed**, 96-run library at 1m resolution.
- [x] **Sx replaced with WindNinja** throughout features/train/gap_fill.
- [x] **Gap-fill default = observed transport** (r=0.83 vs r=0.10 for RF).

- [ ] **WindNinja RF retrain** — compare LOO-CV against `cv_results_sx.json`.
- [ ] **Clip `max_valid_hs_m` to 5.0m** (currently 12.0m, P99=3.78m).

---

## Priority 5 — Avalanche Detection

- [x] **Image-processing boundary detection** (`avalanche.py`). Seeded watershed
      with start zone KML + Canny edges. Defaults: canny_low=0.05, hi=0.08,
      erosion_sigma=1.5. Jan 18: release ~1100 m² / -600 m³, deposit ~3945 m².
- [x] **Multi-period runner** — `--date-before`/`--date-after` args, auto-detects
      pre-event survey, noise mask support.
- [x] **Release area GeoJSON** saved at `data/boundaries/avalanche_release_area.geojson`.

- [ ] **Improve eastern half detection** — shallower dHS (-0.2 to -0.3m) not
      fully captured. Try erosion_sigma=1.2 or secondary fill pass.

- [ ] **Run on all survey periods** — validate no false positives in clean periods.

- [ ] **Min-kernel auto-detection of release area** — investigate whether
      a morphological minimum kernel applied to the dHS anomaly raster can
      auto-detect release zone boundaries without relying on Canny edges.
      The idea: a minimum filter with a kernel sized to the expected crown
      width highlights the local minima (maximum erosion) that correspond
      to the crown and flanks of a slab. Advantages over Canny: no edge
      threshold tuning, naturally connected regions, less sensitive to
      local noise. Compare detected boundaries against Jan 18 observed
      release area and against current Canny+watershed results.
      Reference: scipy.ndimage.minimum_filter, skimage.morphology.erosion.

- [ ] **Jan 18 retrospective** — SNOWPACK precursor signals in crown clusters.

---

## Priority 6 — SNOWPACK Indices & Visualization

- [x] **`visualize_snowpack.py`** — four-tab HTML flipbook:
      - **Loading**: HS, dHS/dt
      - **Stability**: min Sk38, min SSI, min Sn38 (at WL interface)
      - **Propagation**: min TG, accumulated TG, stab deformation rate
      - **Structure**: HS, WL burial depth, slab thickness, WL shear strength,
        WL grain size, slab density
      - **Boundary overlays** on all panels: start zone (green), release area (red)
      - **`--wl-method` flag**: `simple` (bottom 20%, fast) or `grain_type`
        (FC/DH detection from `split_wl_slab`, proper but slow — run overnight)
      - Jan 18 event banner, keyboard navigation (arrows, space, 1-4).

- [x] **`analyze_release_zone.py`** — snapshot comparison (Jan 17):
      - WL/slab boundary via grain_type (FC/DH = 4xx/5xx at base)
      - Slab: density, hardness, grain size, thickness, dominant grain class
      - WL: shear strength, grain size, density, burial depth, thickness
      - Stability: min Sk38/SSI/Sn38/SDR at ±5cm of WL-slab interface
      - Meloche et al. (2025) parameters: E, σ_t, Λ, θ, Π₁, Π₂, A_ca
      - RF classifier: release vs adjacent, per-cluster samples, SHAP if available
      - Cross-date test: train on snapshot, score across season

- [ ] **Critical cut length panel** — add once xsnow maps variable 0606.

- [ ] **Weak layer tracking** — persistent early-season facets as time series.

- [ ] **`grain_type` and `lwc` panels** — facet extent and wet slab monitoring.

---

## Priority 7 — HTML / Project Report

- [x] **Four-tab flipbook** with boundary overlays and structure panel.

- [ ] **Upgrade to project report** — study site, methods, results, Jan 18 case
      study, embed flipbook inline.

---

## Lower Priority / Future

- [ ] **Classifier generalization — design issue** — DO NOT train on non-event
      periods as negatives. Jan 18 is skier-triggered: no avalanche ≠ stable.
      Needs multiple events or independent stability obs (PST, shooting cracks).
      RF model saved to `outputs/plots/rf_model_YYYY-MM-DD.pkl` for future use.

- [ ] **Wet slab integration** — `lwc` from SNOWPACK -> wet slab hazard map,
      connecting to `wet_snow_tracker`.

- [ ] **Zarr cache** — `build_zarr_chunked.py` batched build in progress.
      Batches layer-pad to 338, squeeze slope/realization singletons, append
      along location dim. Used by `visualize_snowpack.py` and `analyze_release_zone.py`.

- [ ] **xsnow contribution** — Zarr pattern and cluster->grid reduction are
      generic. Contribute upstream. Flag pyproject.toml version pin to Florian.

- [ ] **Validation dataset** — other observed avalanches on Little Professor
      this season beyond Jan 18?

- [ ] **PST methodology connection** [LOW] — can SNOWPACK reproduce propagation
      propensity trend from Berthoud Pass PST field tests?

- [ ] **Cluster quality feedback loop** — does high intra-cluster HS std produce
      meaningfully different stability outputs?
      
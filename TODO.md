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

com1DFA will be implemented later (possibly by a different person), but the
scenario generation step and output format need to be designed now. The
distributed SNOWPACK output gives us exactly what's needed to drive both a
probabilistic scenario framework and a physics-based release area model.

### P(release) and release size from SNOWPACK

Three complementary approaches, from simple to mechanistic:

**A. Stability index approach (implement first, validate on Jan 18)**

Individual indices and their limitations:
- `Sk38` (<1.0): skier triggering on 38deg slope.
- `SSI` (<1.5): combines Sk38 with weak layer structure.
- `Sn38` (<1.0): natural stability analog of Sk38.
- Critical cut length `rc`: SNOWPACK variable 0606 (`nElems`). Short rc +
  low Sk38 is the strongest combined signal. See critical cut length item below.

Candidate composite criteria:
1. AND combination: `Sk38 < 1.0 AND Sn38 < 1.5 AND SSI < 1.5`
2. Weighted score: `score = w1*(1/Sk38) + w2*(1/SSI) + w3*(1/Sn38) + w4*(rc_crit/rc)`
3. Mechanistic gate: `rc < threshold` then `min(Sk38, Sn38)` for magnitude
4. Spatial coherence: require minimum contiguous unstable area (>500 m2)

- [ ] **Jan 18 retrospective** — plot index time series in crown-zone clusters
      Nov 26 through Jan 18. Which combination first crossed threshold?
      Was spatial coherence growing before the event?

- [ ] **Critical cut length from SNOWPACK** — variable 0606 / `nElems` (m).
      Not yet mapped by xsnow. Add `'0606': 'critical_cut_length'` to xsnow
      variable mapping and flag to Florian. Then add to `visualize_snowpack.py`
      propagation panel and retrospective analysis.

- [ ] **Release zone vs adjacent slope SNOWPACK comparison** — partition
      clusters into: (1) release zone, (2) adjacent slope within start zone,
      (3) reference outside start zone. Compare time series of Sk38, SSI,
      Sn38, rc, temperature gradient, HS before Jan 18. Expected result:
      release zone shows lower stability earlier and more persistently.
      This is direct validation that distributed SNOWPACK captures spatial
      variability relevant to release — a key paper claim.
      Implementation: use `cluster_map.npy` + detected release mask to assign
      cluster IDs to each group, load via xsnow, plot with event date marked.

**B. Monte Carlo / MCMC approach**
- Sample slab density ±20%, weak layer shear strength ±1 sigma, slab thickness
  from cluster HS distribution. Bounds from SNOWPACK output directly.

**C. DA-MPM release area estimation (mechanistic, later)**
- Meloche et al. (2025, JGR Earth Surface, doi:10.1029/2025JF008470)
- Key input: shear strength gradient theta from distributed SNOWPACK grid
- DA-MPM code: C++ on Zenodo (Guillet et al. 2023), Python post-processing
  on GitHub (Meloche 2025). Well-suited for D3+; may overestimate small slabs.

- [ ] **`step_scenarios()` in `pipeline.py`** — standardized output directory:
      `outputs/scenarios/YYYYMMDD/` containing release_zones.geojson,
      scenario depth rasters (P10-P90), scenario_weights.json,
      weak_layer_params.json, mcmc_bounds.json, metadata.json.

- [ ] **`identify_release_zones()`** — identify candidate zones from current
      stability fields alone (no survey pair). Works on any given day.

- [ ] **Validation against Jan 18** — run scenario generation for Jan 17,
      compare P50 release area and depth against field observations.

---

## Priority 2 — Multi-Slope / Off-the-Shelf Architecture

- [x] **`start_zone_kml` added to `ProjectConfig`** —
      `data/boundaries/Litte_prof_start_zone.kml` used by `detect_boundaries()`
      to restrict release detection to known release terrain.

- [ ] **`slopes.yaml` slope registry** — one entry per slope with DEM path,
      boundary KML, start zone KML, AWS station IDs + roles, highway corridor,
      milepost range. `ProjectConfig` instantiated from registry entry.

- [ ] **Remove hardcoded station IDs** — `sql_util.py` (`WEATHER_STATIONS`
      dict) and `smet_writer.py` (`SUMMIT_STATION`/`BASE_STATION`). Replace
      with config-driven `StationConfig` from slope registry.

- [ ] **Multi-slope runner** — loops over `slopes.yaml`, runs full pipeline
      per slope, writes to common output schema.

- [ ] **Standardized output schema** — GeoJSON per slope per day: release
      zones, stability index summary, scenario weights.

- [ ] **Config validation** — `cfg.validate()` checks paths and SQL
      connectivity before any step runs.

---

## Priority 3 — Daily Survey Operational Mode

- [ ] **Daily mode gap-fill** — `--daily-mode` flag skips RF transport when
      consecutive surveys are <36h apart, uses direct differencing instead.

- [ ] **Incremental SNOWPACK updates** — append to SMET, restart from
      `_res.sno`, append to `.pro` rather than full season rerun.

- [ ] **Cron-compatible pipeline wrapper** — automate: fetch weather, process
      new survey, update SMETs, run SNOWPACK forward one day, regenerate maps.

---

## Priority 4 — Transport Model

- [x] **WindNinja installed** — v3.13 via conda-forge, 64 OpenMP threads,
      running at `/home/ron/miniforge/envs/windninja/bin/WindNinja_cli`.

- [x] **Wind library generated** — 96 runs (16 directions x 6 speeds:
      3/8/15/25/40/60 m/s) at 1m resolution. Library at
      `/home/ron/snowpack_model_feeder/windninja/library/`.

- [x] **Sx replaced with WindNinja** — `spatial_model.py` has
      `load_wind_library()`, `interpolate_wind_field()`,
      `resample_wind_to_dem()`, `build_windninja_feature_array()`.
      `pipeline.py` features/train/gap_fill steps updated.

- [x] **Gap-fill default changed to observed transport** (`--no-model`).
      LOO-CV: median r=0.83, RMSE=65cm vs RF median r=0.10, RMSE=174cm.
      Modes: (default) observed transport, `--station-only`, `--use-model` (RF).

- [ ] **WindNinja RF retrain** — run `pipeline.py train`, compare LOO-CV
      against `cv_results_sx.json`. Document whether WindNinja improves RF.

- [ ] **Clip `max_valid_hs_m` to 5.0m** — currently 12.0m. P99=3.78m,
      max=10.62m. Change in `config.py` and rerun from `gap_fill`.

- [ ] **Transport model options** — maintain as swappable subclasses:
      `EmpiricalTransportModel` (RF + WindNinja), stub for physics-based.

---

## Priority 5 — Avalanche Detection

- [x] **Image-processing boundary detection implemented** (`avalanche.py`).
      Pipeline: KML domain mask -> Canny edges on dHS anomaly -> seeded
      watershed with positive-anomaly background markers -> morphological
      cleanup -> contours. Start zone KML restricts release candidates.
      Statistics recomputed within start zone for threshold sensitivity.
      Tuned defaults: canny_low=0.05, canny_high=0.08, erosion_sigma=1.5.
      Jan 18 result: release ~1100 m² / -600 m³, deposit ~3945 m² / 3121 m³.
      Standalone tuning runner: `python avalanche.py --date-before YYYY-MM-DD`

- [ ] **Improve release zone completeness** — eastern half of Jan 18 slab
      (shallower dHS loss, -0.2 to -0.3m) not fully captured. Options:
      lower erosion_sigma to 1.2, or secondary fill pass with lower threshold.

- [ ] **Run detection on all survey periods** — validate no false positives
      in clean periods. Flag periods with detected events for review.

- [ ] **Jan 18 retrospective** — look for SNOWPACK precursor signals:
      Sk38/SSI evolution, weak layer persistence, TG >10 C/m, crown vs
      reference cluster comparison. Potentially the strongest paper result.

---

## Priority 6 — SNOWPACK Indices & Visualization

- [x] **Three-panel `visualize_snowpack.py`** with HTML flipbook (3 tabs):
      - Loading: HS, dHS/dt
      - Stability: min Sk38, min SSI, min Sn38 (buried >30cm)
      - Propagation: min TG, accumulated TG, stab deformation rate
      Jan 18 event window highlighted in red banner.

- [ ] **Critical cut length panel** — add to propagation tab once xsnow
      maps variable 0606. Use `min(critical_cut_length)` within search depth.

- [ ] **Weak layer tracking** — identify persistent early-season weak layer
      (Nov-Dec facets) and track depth, extent, stability as single time series.

- [ ] **dHS/dt animation** — fourth panel or separate animation from hourly
      grids with 24h smoothing.

- [ ] **`grain_type` and `lwc` panels** — map faceted/depth hoar extent and
      liquid water content (wet slab monitoring).

---

## Priority 7 — HTML / Project Report

- [x] **Basic HTML flipbook exists** — 3-tab layout, keyboard navigation,
      Jan 18 event banner, auto-generated at `plots/daily_frames/index.html`.

- [ ] **Upgrade to project report** — add: study site description, methods
      summary (UAS workflow, spatial model, SNOWPACK, AvaFrame), results
      section with HS evolution + stability animations + Jan 18 case study,
      embed flipbook inline.

---

## Lower Priority / Future

- [ ] **Wet slab integration** — `lwc` from SNOWPACK -> spatial wet slab
      hazard map using cluster->grid mapping, connecting to `wet_snow_tracker`.

- [ ] **xsnow contribution** — Zarr caching pattern and cluster->grid
      reduction are generic. Contribute upstream. Flag `pyproject.toml`
      version pin issue to Florian.

- [ ] **Validation dataset** — other observed avalanches on Little Professor
      this season beyond Jan 18?

- [ ] **PST methodology connection** [LOW PRIORITY] — can SNOWPACK reproduce
      propagation propensity trend from PST field tests at Berthoud Pass?

- [ ] **Cluster quality feedback loop** — check whether high intra-cluster
      HS std produces meaningfully different stability outputs.
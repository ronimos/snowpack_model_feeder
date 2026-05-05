# Operational Mode: Design Document

**Project:** UAV HS surveys to Avalanche Rounout Propability Envelope Model Chain  
**Purpose:** Continuous, automated hazard assessment for highway avalanche operations  
**Last updated:** 2026-05-01

---

## 1. Operational Concept

The pipeline operates in three modes depending on what data is available:

**Daily forward mode** — between UAS surveys, the system runs daily using weather station data to evolve the snowpack stratigraphy. Spatial HS distribution stays frozen at the last survey; only the internal structure (sintering, temperature gradient metamorphism, new snow loading) evolves. This is the baseline operational state.

**NWP forecast mode** — extends the daily pipeline 72 hours into the future using numerical weather prediction data. Produces rolling hazard forecasts at T+0, T+1, and T+2 for mitigation planning. Forecasts converge toward reality as the target date approaches and NWP is replaced by observed data.

**Survey correction mode** — when a new UAS survey arrives, the system back-corrects the gap-filled HS with observed spatial transport, reruns SNOWPACK from the previous survey date, and regenerates all downstream products.

These three modes interact in a continuous cycle:

```
   Survey N arrives
        │
        ├─► Survey correction: backfill observed HS, rerun SNOWPACK
        │
        ├─► Daily forward: extend one day at a time with station data
        │       │
        │       └─► NWP forecast: extend 3 days with NWP, generate scenarios
        │               │
        │               └─► Next day: replace T+0 NWP with observed,
        │                   refresh T+1/T+2 with latest NWP
        │
        ├─► (Avalanche detected? → reinit → rerun)
        │
   Survey N+1 arrives
        │
        └─► Survey correction: correct everything since Survey N
```

---

## 2. Daily Forward Mode

### 2.1 What runs daily

Between UAS surveys, the pipeline executes these steps each morning (or on a cron schedule):

1. **Ingest new weather data** — pull the latest hourly records from a weather stations. Append to a local weather cache.

2. **Extend SMET files** — append new hourly records to existing per-cluster SMET files. Only new rows since the last run are added (append-only, no rewrite). The spatial HS distribution within each SMET remains frozen at the last survey — only the meteorological forcing columns (TA, RH, VW, DW, ISWR, ILWR, PSUM) update.

3. **Run SNOWPACK incrementally** — advance the simulation by one day from the restart files. Each cluster's `.sno` picks up where it left off. Runtime: ~5 minutes for 6,636 clusters.

4. **Extract stability snapshot** — read the Zarr (or directly from the new `.pro` output) at today's timestamp. Extract Sk38, SSI, Λ, τ_g, slab thickness for all start zone clusters.

5. **Generate scenarios** — run `step_scenarios` at today's snapshot date. Produces trigger locations, release polygons, and depth rasters for com1DFA.

6. **Run com1DFA** — execute the flow simulation on today's scenarios. Produce runout probability envelopes.

7. **Post hazard summary** — report which paths are at what hazard level, whether any scenarios reach infrastructure, and how conditions compare to yesterday.

### 2.2 What does NOT change daily

Between surveys, these remain frozen:

- **HS spatial distribution** — each cluster's HS trajectory was set by the last survey and the gap-fill model. Wind transport between surveys is approximated by station dHS, not resolved spatially.
- **Cluster map** — cluster boundaries are fixed for the season.
- **DEM / terrain features** — only change if a new ground DEM is acquired.
- **Transport model / RF features** — only rebuilt when the wind library or DEM changes.

### 2.3 Limitations of daily mode

The daily forward mode captures stratigraphy evolution (a buried WL weakening over time, new storm slabs forming) but misses spatial redistribution events. If a wind event moves significant snow between surveys, the HS pattern in the model is wrong until the next survey corrects it. The NWP forecast mode (§3) partially addresses this by incorporating forecast wind, but true spatial correction requires a UAS survey.

---

## 3. NWP Forecast Mode

### 3.1 Purpose

Extend the pipeline 72 hours into the future to provide lead time for mitigation decisions. The hazard team sees "instability building at Little Professor, Sk38 crosses critical threshold tomorrow afternoon, D2 release probability peaks at T+2 with incoming storm" — actionable information for staging crews, closing roads, or pre-positioning equipment.

### 3.2 Workflow

Each day (after the daily forward mode completes):

1. **Ingest NWP forecast** — pull the latest 72-hour forecast from WRF (2 km, hourly) or NBM (2 km, hourly) Extract T, precipitation, wind speed/direction, RH, radiation at the study site coordinates.

2. **Downscale NWP to site** — at minimum, apply lapse-rate correction for temperature and orographic enhancement for precipitation. Wind fields from NWP are coarse (1–3 km) — for forecast mode, station wind + NWP trend may be sufficient. Full WindNinja re-downscaling adds accuracy but also compute time.

3. **Convert to SMET format** — append NWP-derived hourly records to cluster SMET files after the last observed timestamp. These rows are flagged as forecast (not observed) so they can be replaced when real data arrives.

4. **Run SNOWPACK to T+72h** — incremental from the current restart files through the forecast period.

5. **Extract snapshots at T+0, T+1, T+2** — three stability states, each representing the predicted snowpack at 18:00 UTC on that day.

6. **Generate scenarios at each snapshot** — trigger selection, BFS release polygons, depth rasters. Size factor ranges widen with lead time to reflect forecast uncertainty:

   | Lead time | Size factors | Rationale |
   |-----------|-------------|-----------|
   | T+0 | 0.85, 1.00, 1.15 | Mostly observed forcing, tight bounds |
   | T+1 | 0.70, 0.85, 1.00, 1.15, 1.30 | Mixed observed + NWP |
   | T+2 | 0.55, 0.70, 0.85, 1.00, 1.15, 1.30, 1.45 | NWP-dominated, wide bounds |

7. **Run com1DFA** at each snapshot → three sets of runout envelopes with progressively wider uncertainty.

8. **Archive forecast** — save all three snapshots for later verification of forecast skill.

### 3.3 Rolling convergence

Each day, the forecast refreshes:

| Target date | Yesterday's forecast | Today's forecast | Improvement |
|-------------|---------------------|-----------------|-------------|
| Tomorrow | T+2 (48h NWP) | T+1 (24h observed + 24h NWP) | 24h of real data replaces NWP |
| Today | T+1 (24h NWP) | T+0 (mostly observed) | NWP replaced by station data |

The forecasters see runout envelopes narrowing as the target date approaches. "Could reach the road" (T+2, wide envelope) tightens to "won't reach the road" or "close it now" (T+0, tight envelope). This convergence is the operationally useful signal — it tells you how much to trust the forecast.

### 3.4 Forecast pruning

When new observed data arrives (hourly station update), the NWP-derived rows in the SMET files are replaced with real observations. The SNOWPACK restart files are updated, and the forecast branch from that timestamp forward is regenerated with the latest NWP. This is the same back-correction pattern as the survey correction mode (§4), but at hourly granularity.

---

## 4. Survey Correction Mode

### 4.1 Trigger

A new UAS survey arrives. This reveals the actual spatial snow depth change since the last survey — the gap-filled HS that the pipeline has been using was an approximation (station dHS extrapolated spatially). The correction updates everything from the last survey forward.

### 4.2 Workflow

1. **Resample new survey** — add the new survey to the 1 m grid. Only process the new survey, not the existing ones.

2. **Compute observed transport** — dHS between the previous and new surveys, corrected for station dHS. This is the real spatial redistribution pattern.

3. **Replace gap-filled HS** — for the inter-survey period, replace the station-only gap-fill with hourly HS disaggregated from the observed transport pattern. The wind-energy disaggregation weights hourly station dHS by the observed spatial transport field.

4. **Regenerate SMETs** — update SMET files from the previous survey date forward. Only rows after the last survey timestamp change — earlier rows are preserved.

5. **Rerun SNOWPACK** — incremental from the previous survey date. Uses the corrected SMET forcing. Only clusters whose HS changed meaningfully (> threshold) need rerunning — use the `CLUSTERS_FILE` filter in `run_snowpack.sh`.

6. **Update Zarr** — append new timesteps or overwrite the corrected period.

7. **Rerun analysis + scenarios** — extract features and generate scenarios at the latest snapshot.

### 4.3 What changes vs. what stays

| Component | Changes? | Notes |
|-----------|----------|-------|
| DEM, cluster map | No | Only change with new ground DEM |
| HS grids (new survey) | New file added | Only the new survey is resampled |
| HS grids (existing) | No | Already processed |
| Transport field | Yes | Observed replaces gap-filled for the latest inter-survey period |
| SMET files | Partially | Only rows from previous survey onward |
| SNOWPACK .sno | Yes | Restart from previous survey date |
| Zarr | Partially | Overwrite corrected timesteps, append new ones |
| Features / scenarios | Yes | Regenerated from corrected Zarr |

---

## 5. Avalanche Event Handling

### 5.1 Detection

When a new survey reveals an avalanche occurred (significant negative dHS in the start zone), the min-kernel detection method identifies the release boundary automatically. Alternatively, a hand-drawn GeoJSON can be used for known events.

### 5.2 SNOWPACK reinitialization

The reinit step scours release cluster `.sno` files by removing layers from the top down to the slab/WL interface depth. The two-pass workflow:

1. **Pass 1:** SNOWPACK from season start (or last restart) to event date
2. **Reinit:** scour release cluster `.sno` files
3. **Pass 2:** SNOWPACK from event date to current date (scoured clusters only)

### 5.3 Multiple events per season

For seasons with multiple avalanches, the pipeline chains reinit steps chronologically. Each event scours a different (possibly overlapping) set of clusters:

```
SNOWPACK: start → Event A → reinit A → Event B → reinit B → ... → now
```

A future improvement (documented in TODO) supports this via an events JSON file processed as a loop.

### 5.4 Interaction with forecast mode

If an avalanche is detected during the current forecast period, the NWP forecast branch from the event date forward is invalidated for the affected clusters. The reinit step scours those clusters, and the forecast regenerates from the post-event snowpack state. Unaffected clusters continue with their existing forecast.

---

## 6. Efficiency

### 6.1 SMET append-only writes

Current `step_smet` rewrites all ~6,636 files from scratch. An append mode reads each existing file, finds the last timestamp, and writes only new rows. Saves I/O and ~10 minutes per daily run.

### 6.2 Incremental SNOWPACK

Already implemented via restart files. SNOWPACK reads the `.sno` from the last run, advances to the new end date, and writes an updated `.sno`. Only the new timesteps are simulated. For daily mode, this is ~24 hours of simulation per cluster — seconds of runtime.

### 6.3 Partial SNOWPACK rerun

After survey correction or reinit, only clusters whose forcing actually changed need rerunning. The `CLUSTERS_FILE` parameter in `run_snowpack.sh` accepts a list of cluster IDs. For reinit, this is the ~400 release clusters (from `reinit_stats.json`). For survey correction, this is clusters where the corrected HS differs from the gap-filled HS by more than a threshold.

### 6.4 Zarr append

Current `build_zarr_chunked.py` rebuilds the entire Zarr from all `.pro` files (~15 min). An append mode would track which timesteps are already in the store and add only new ones. For daily mode, this adds ~24 timesteps instead of rebuilding ~3,000.

### 6.5 Feature caching with change detection

Only recompute Meloche features for clusters where HS changed by more than a threshold (e.g., 5 cm) since the last extraction. Cache previous feature values and diff against the new snapshot. For daily forward mode, only clusters receiving significant new snow loading will change meaningfully.

### 6.6 Partial DEM resampling

Only resample the new survey to the 1 m grid. Don't reprocess existing surveys. Currently `step_resample` regenerates everything. For a season with 23 surveys, this saves 22 unnecessary resample operations on each new survey.

---

## 7. Cluster Management

### 7.1 Season-stable clustering

Clusters are generated at the start of the season from the HS evolution matrix (PCA + MiniBatch K-means + recursive splitting + contiguity enforcement). The resulting ~6,636 clusters remain fixed for the entire season. This provides stable SNOWPACK simulation identity — each cluster accumulates a continuous `.pro` history.

### 7.2 Mid-season splitting

When a new survey reveals that a cluster's internal HS variability exceeds the quality threshold (`max_cluster_std_m` = 8 cm), it is split via MiniBatch K-means bisection. The inheritance model avoids rerunning the full season:

**Primary child** — inherits the parent's cluster ID, `.sno` restart file, SMET, `.pro` history, and Zarr entries. Nothing changes for this child.

**Secondary child** — receives a new cluster ID (max_existing + 1) and:
1. A copy of the parent's `.sno` at the split timestamp (same stratigraphy)
2. A new SMET generated from the gap-filled hourly HS for the child's pixel membership
3. SNOWPACK run from the split date forward (incremental from the copied `.sno`)

This costs one SNOWPACK run per split, from the split date to the current date. For 10–20 splits at mid-season, ~30 minutes of compute.

### 7.3 Quality monitoring

At each new survey, compute intra-cluster HS variability for all clusters. Flag clusters exceeding `max_cluster_std_m` as split candidates. Report: how many need splitting, where on the slope, and how much the split would change the total cluster count. This provides a data-driven trigger rather than arbitrary re-clustering.

### 7.4 What triggers a full re-cluster

A complete re-clustering (discarding all SNOWPACK history, starting fresh) is needed when:

- **New ground DEM** — terrain changes → slope, aspect, and all derived features change
- **New wind library** — transport model changes → HS evolution patterns change
- **Major config change** — cluster parameters (max_cells, n_pca_components) change
- **Start of new season** — fresh HS evolution matrix from the first few surveys

In all other cases, incremental splitting is preferred over full re-clustering.

---

## 8. File Management

### 8.1 File lifecycle

| File type | Created by | Updated by | Frequency |
|-----------|-----------|-----------|-----------|
| `hs_YYYY-MM-DD.npy` | `step_resample` | Never | One-time per survey |
| `cluster_map.npy/tif` | `step_cluster` | Cluster splitting | Start of season + splits |
| `cluster_XXXX.smet` | `step_smet` | Daily append | Daily |
| `cluster_XXXX_cluster_XXXX.sno` | SNOWPACK | Daily increment / reinit scour | Daily + events |
| `cluster_XXXX_cluster_XXXX.pro` | SNOWPACK | Daily append | Daily |
| `slope_snowpack.zarr` | `build_zarr` | Daily append | Daily |
| `*_features_*.csv` | `step_analyze` | Daily refresh | Daily |
| `scenarios/` | `step_scenarios` | Daily refresh | Daily |

### 8.2 Storage management

The `.pro` files are the largest output (~10–50 MB each, ~6,636 files). For a full season:

- `.pro` total: ~100–300 GB
- Zarr: ~20–50 GB (compressed)
- SMET: ~5 GB
- Everything else: < 1 GB

For multi-season operation, archive previous seasons' `.pro` files to cold storage and keep only the Zarr. The Zarr contains everything needed for analysis; the `.pro` files are only needed for Zarr rebuilds.

### 8.3 Backup strategy

- **Daily:** `.sno` files (restart state — losing these means rerunning the full season)
- **Per survey:** `hs_YYYY-MM-DD.npy` + `cluster_map` + feature CSVs
- **Per season:** Zarr store + scenario archives + reinit stats

---

## 9. Automation

### 9.1 Cron schedule

```
# Daily pipeline (after station data is available for the previous day)
0 8 * * *  /home/ron/snowpack_model_feeder/run_daily.sh >> /var/log/snowpack_daily.log 2>&1

# NWP forecast refresh (after latest NWP cycle is available)
0 10 * * *  /home/ron/snowpack_model_feeder/run_forecast.sh >> /var/log/snowpack_forecast.log 2>&1
```

### 9.2 Monitoring

The smoke test (`scripts/smoke_test.py`) runs after each pipeline execution. On failure, send an alert (email, Slack) with the failed checks. Key conditions to monitor:

- SMET temporal gaps > 6 hours (station outage)
- SNOWPACK cluster failures > 1% of total
- Sk38 values all NaN (feature extraction bug)
- Scenario release area outside expected range
- Zarr missing expected timesteps

### 9.3 Manual intervention triggers

The pipeline should alert a human (not auto-remediate) when:

- Avalanche detected (requires reinit decision: auto-detect or hand-draw boundary?)
- Cluster quality monitoring flags > 10 clusters for splitting
- NWP data source unavailable for > 24 hours
- Station data gap > 48 hours (beyond refill capacity)
- DEM co-registration drift detected (survey vs ground DEM origin shift)

---

## 10. Uncertainty Management

### 10.1 Sources of uncertainty by mode

| Source | Daily forward | NWP forecast | Survey correction |
|--------|--------------|-------------|-------------------|
| HS spatial distribution | Frozen at last survey | Frozen + NWP precip | Corrected by new survey |
| Stratigraphy | SNOWPACK physics | SNOWPACK + NWP forcing | SNOWPACK + corrected forcing |
| Wind transport | Not resolved | NWP wind (coarse) | Observed from dHS |
| Trigger location | From current Sk38 | From forecast Sk38 | From corrected Sk38 |
| Release area | size_factor ensemble | Wider size_factor range | size_factor ensemble |
| Runout | μ/ξ uncertainty | μ/ξ + release uncertainty | μ/ξ uncertainty |

### 10.2 Uncertainty communication

The hazard team needs clear, actionable uncertainty signals:

- **Traffic light:** green (no scenarios reach road), yellow (some scenarios reach road), red (median scenario reaches road)
- **Confidence indicator:** high (T+0, recent survey), medium (T+1, or >5 days since survey), low (T+2, or >10 days since survey)
- **Trend arrow:** instability increasing / stable / decreasing compared to yesterday
- **Key driver:** what is causing the change (new snow loading, wind event, warming, WL weakening)

### 10.3 Forecast verification

Archive daily T+0/T+1/T+2 forecasts. Post-season, evaluate:

- How often did T+1 correctly predict T+0 outcome? (hit rate)
- How much did the runout envelope shrink from T+2 → T+0? (convergence rate)
- For the days roads were closed, what did T+1 say? (decision-support value)
- False alarm rate: scenarios predicted road impact but no avalanche occurred

This calibration data justifies the system's operational value to stakeholders (CDOT, avalanche programs) and identifies which forecast components need improvement.

---

## 11. System Performance Monitoring

Operational performance tracking to evaluate how well the system is doing, identify degradation, and build the calibration dataset needed to improve accuracy over time.

### 11.1 Daily forward accuracy

**Metric: survey correction magnitude.** When a new survey arrives, compare the gap-filled HS (station-only extrapolation) against the observed HS from the survey. The per-cluster RMSE between predicted and observed HS at the survey date is the primary measure of daily forward mode accuracy.

| Metric | Good | Acceptable | Needs attention |
|--------|------|-----------|-----------------|
| Median cluster HS RMSE | < 10 cm | 10–25 cm | > 25 cm |
| P90 cluster HS RMSE | < 25 cm | 25–50 cm | > 50 cm |
| Spatial correlation (r) | > 0.8 | 0.5–0.8 | < 0.5 |

Track these per inter-survey period. Systematic degradation (RMSE growing over the season) indicates the gap-fill model is losing accuracy — possibly because wind patterns changed or the station became less representative of the domain. Sudden spikes indicate a wind event that the station-only model couldn't capture.

**Diagnostic:** plot observed vs predicted HS maps side-by-side at each survey. Residual maps (predicted − observed) show where the gap-fill is biased — persistent positive residuals mean the model under-predicts wind erosion there; negative residuals mean it misses loading.

### 11.2 Avalanche prediction accuracy

**Metric: release area overlap.** When an avalanche is detected (min-kernel or field observation), compare the system's predicted release area (from the most recent `step_scenarios` run before the event) against the detected/observed release area.

| Metric | Calculation | Target |
|--------|-------------|--------|
| Area ratio | predicted / observed | 0.7–1.3 |
| IoU (intersection over union) | overlap / total | > 0.5 |
| Trigger location error | distance from predicted trigger to observed crown | < 30 m |
| Crown position error | distance from predicted crown to observed crown | < 20 m |
| Flank position error | distance from predicted flanks to observed flanks | < 15 m |

**Metric: release volume comparison.** Compare the scenario's `total_volume_m3` against the dHS-derived volume from the min-kernel detection. Large discrepancies indicate either the depth raster (slab_thickness) or the release polygon is wrong.

**Metric: size classification.** Did the system correctly predict the avalanche size class? Map the predicted release area and volume to the D-scale (D1–D5) and compare against the observed classification.

### 11.3 Runout prediction accuracy

**Metric: runout distance.** Compare predicted runout (from com1DFA ensemble) against observed debris extent.

| Metric | Calculation | Target |
|--------|-------------|--------|
| Runout distance ratio | predicted / observed | 0.8–1.2 |
| Lateral spread ratio | predicted / observed | 0.8–1.5 |
| P(exceedance at road) vs actual | did debris reach road? | correct sign |

**Metric: envelope capture.** Did the observed runout fall within the predicted probability envelope? If the system said "P(road) = 0.15" and debris did reach the road, that's either bad luck (15% events do happen) or systematic underestimation. Track over multiple events: the fraction of times the observed outcome falls within the predicted confidence interval should match the stated confidence level.

### 11.4 Forecast skill

**Metric: forecast convergence.** For each target date, compare T+2, T+1, and T+0 predictions:

| Metric | Calculation |
|--------|-------------|
| Sk38 RMSE (T+2 vs T+0) | how much does stability prediction change? |
| Sk38 RMSE (T+1 vs T+0) | should be smaller than T+2 |
| Release area change (T+2 → T+0) | did the polygon shrink/grow/move? |
| Runout envelope width (T+2 vs T+0) | should narrow |

Track the convergence rate: if T+1 isn't consistently better than T+2, either the NWP isn't adding value or the back-correction is introducing noise. If T+0 is often surprised by events that T+2 predicted (e.g., T+2 showed instability that T+0 missed because a warming event didn't materialize), that's a different problem — the NWP was adding false alarms.

**Metric: forecast decision value.** For each day the road was closed or mitigation was deployed:
- What did T+1 predict? (Was there lead time?)
- What did T+2 predict? (Was the warning earlier?)
- Would the decision have been different without the forecast? (Counterfactual value)

### 11.5 SNOWPACK model performance

**Metric: simulated vs observed HS.** At each survey, compare SNOWPACK's simulated HS per cluster against the observed HS from the UAS survey. This evaluates the full chain: SMET forcing → SNOWPACK physics → HS output.

**Metric: stratigraphy validation.** When manual snowpit observations are available (e.g., from CAIC field teams), compare SNOWPACK's predicted layer structure against observed:
- WL depth: predicted vs observed (cm)
- WL grain type: predicted vs observed (FC/DH classification)
- Slab density: predicted vs observed (kg/m³)
- Hand hardness profile: predicted vs observed (SNOWPACK scale)

These are opportunistic — snowpits aren't available on demand, but when they are, they provide the deepest validation of the model chain.

### 11.6 Cluster quality evolution

**Metric: intra-cluster HS variability over time.** At each survey, compute the per-cluster HS standard deviation. Track whether cluster quality degrades as the season progresses (wind events creating heterogeneity that the initial clustering didn't capture).

| Metric | Start of season | Mid-season | End of season |
|--------|----------------|------------|---------------|
| Tight clusters (std < 10 cm) | target > 60% | monitor | accept > 40% |
| Loose clusters (std > 30 cm) | target < 2% | flag for splitting | accept < 5% |

If the fraction of loose clusters exceeds 5%, mid-season splitting should be triggered.

### 11.7 Station data quality

**Metric: station uptime.** Percentage of hours with valid data from each station (CAABT, CAABM).

**Metric: gap frequency and duration.** Number and length of data gaps requiring refill. Track whether gaps are increasing (station hardware degradation) or seasonal (rime icing in December).

**Metric: station representativeness.** Correlation between station dHS and domain-mean dHS from UAS surveys. If this drops below 0.5, the station is losing its value as a spatial proxy — the gap-fill model relies on station dHS being representative of domain-wide accumulation patterns.

### 11.8 Performance dashboard

Aggregate the above metrics into a daily dashboard:

```
=== System Performance — 2026-02-15 ===

  Daily forward (since last survey 6 days ago):
    Estimated RMSE: ~18 cm (based on historical correction magnitude)
    Station uptime: 100% (CAABT), 98% (CAABM)

  NWP forecast:
    T+0 trigger: cid=3178 Sk38=0.12 (▼ from 0.18 yesterday)
    T+1 trigger: cid=3178 Sk38=0.08 (storm loading predicted)
    T+2 trigger: cid=3178 Sk38=0.05 (wind event, P(road)=0.22)
    Confidence: MEDIUM (6 days since survey, NWP wind uncertain)

  Last event verification (Jan 18):
    Release area ratio: 0.90 (IoU=0.64)
    Runout envelope captured observed: YES
    Trigger location error: 8 m

  Cluster quality:
    Tight: 54%  Medium: 43%  Loose: 2%
    Splits recommended: 0
```

### 11.9 Seasonal performance report

At end of season, compile:

1. **Survey correction magnitudes** — time series of RMSE per inter-survey period. Shows where the gap-fill was weakest and how it correlates with weather events.
2. **Avalanche prediction scorecard** — for each detected event, the predicted vs observed release area, volume, trigger location, and runout.
3. **Forecast skill summary** — mean convergence rates, false alarm rate, hit rate at each lead time.
4. **Station data summary** — uptime, gap statistics, representativeness trend.
5. **Cluster quality evolution** — fraction of tight/medium/loose clusters per survey.
6. **Recommendations** — which components need improvement for next season (better NWP downscaling? More frequent surveys? Different clustering parameters? Additional stations?)

This report is the deliverable that justifies continued operation to stakeholders and guides R&D priorities for the next season.


# **A Decision-Support Model Chain For Operational Avalanche Hazard Assessment: Project Description**

## **1\. Introduction:**

On March 7, 2019, avalanche mitigation work above Interstate 70 near Herman Gulch, Colorado, triggered a historically large avalanche on the Bathel avalanche path. Following two consecutive major storm cycles, Colorado Department of Transportation (CDOT) and Colorado Avalanche Information Center (CAIC) crews conducted control work above the interstate using explosive charges. The first shot released enough snow, trees, and rock debris to bury both directions of I-70, closing one of Colorado’s primary transportation corridors for approximately 11 hours while crews removed debris from the roadway.

The Bathel event highlighted one of the central operational challenges in highway avalanche forecasting: determining not only whether a slope is unstable, but how much snow is likely to release and whether the resulting avalanche will reach the road. Prior to March 7, uncertainty in the spatial distribution of snow loading and instability within the Bathel starting zone often resulted in mitigation missions that produced little or no avalanche activity. While these “no-result” missions are operationally preferable to large uncontrolled or oversized releases, they still require road closures, personnel, explosives, aircraft support, and significant operational coordination. The challenge for avalanche programs is therefore not simply identifying instability, but optimizing mitigation timing to reduce both unnecessary operational impacts and the likelihood of unusually large releases.

Colorado’s mountain highways cross dozens of avalanche paths managed by CAIC and CDOT highway avalanche programs. These programs rely on experienced forecasters who integrate weather observations, snowpack assessments, field observations, and years of institutional knowledge to make mitigation decisions. Yet even highly experienced teams face a fundamental limitation: avalanche forecasting at the scale of individual starting zones remains constrained by long return periods and sparse spatial information. Snowpit observations and weather stations provide critical point measurements, but they cannot fully describe how snow depth, loading, and instability vary across complex terrain.

This limitation becomes particularly important during infrequent high-consequence storm cycles. Many avalanche paths capable of producing major roadway impacts have return periods long enough that even experienced forecasters may have observed only a limited number of comparable events on a specific path. As a result, operationally important questions often remain difficult to answer quantitatively: Where is the snowpack most heavily loaded? Where are the most likely failure locations? How much snow is available for release? What avalanche size is plausible under current conditions? And most importantly for operational avalanche programs, what is the probability that debris reaches a stracture?

This project addresses those questions through a spatially distributed avalanche assessment framework that links remote sensing, distributed snowpack modeling, release-area identification, and avalanche dynamics simulations into a unified operational workflow. We used High-resolution UAV surveys to measure snow depth distribution across avalanche terrain. We force this data into distributed SNOWPACK simulations to estimate spatial variability in snow stratigraphy and instability, and to drive potential release areas and avalanche sizes from the distributed SNOWPACK model, which are then fed as input to com1Frame avalanche flow simulations. The resulting ensemble of avalanche trajectories and runout distances to generate a probabilistic avalanche runout envelope that estimates the likelihood of roadway impact under current conditions.

The intent of this framework is not to replace operational avalanche forecasters, but to augment decision-making with quantitative, spatially explicit information at the scale of individual avalanche paths. By integrating snow distribution, modeled snowpack structure, release likelihood, avalanche size potential, and runout probability into a single workflow, the system provides operational teams with additional tools to support mitigation timing decisions during both routine storm cycles and rare high-consequence events.

Ultimately, the operational question is straightforward: will the avalanche reach the road, and when should mitigation occur to minimize both risk and disruption? Answering that question requires moving beyond isolated point observations toward a distributed understanding of snowpack conditions and avalanche behavior across the entire avalanche path.

## **2\. Path Forward Approach**

We are developing an end-to-end model chain that transforms UAS snow depth surveys into spatially explicit runout probability envelopes for highway avalanche paths. The chain couples four components:

1. **Spatially distributed snowpack simulation** — UAS-derived snow depth maps provide the spatial forcing for SNOWPACK, a physics-based snow cover model, at \~6,600 locations across the avalanche path. Each location receives its own hourly forcing and evolves an independent stratigraphic profile.

2. **Release area determination** — A physics-based crack propagation model (after Meloche et al., 2025\) identifies where the snowpack is most likely to fail and how far the fracture propagates. The model uses distributed slab and weak-layer properties from SNOWPACK to delineate release polygons.

3. **Flow dynamics simulation** — The release polygon and depth raster feed com1DFA (AvaFrame), a depth-averaged flow model that simulates the avalanche as it descends the track and deposits on the runout zone.

4. **Probabilistic hazard envelope** — Multiple scenarios (varying trigger location, release area size, slab depth, and flow parameters) produce an ensemble of runout extents. The envelope is the probability that debris reaches any given point on the road.

The chain is designed for both retrospective validation (did the model predict the Jan 18, 2026 event?) and operational forecasting (what will conditions look like tomorrow?). The operational mode extends the pipeline with NWP forecast data to provide 72-hour rolling hazard forecasts with quantified uncertainty. See operational\_design.md for the full operational architecture.

![Fig. 1: Model chain overview](assets/fig_10_model_chain.svg)*Figure 1: End-to-end model chain. UAS snow depth surveys and weather station data feed the snowpack-model-feeder pipeline, which generates per-cluster SMET forcing for distributed SNOWPACK simulation. SNOWPACK stratigraphy feeds the Meloche et al. (2025) release area model, which produces release polygons and depth rasters for com1DFA flow simulation. The ensemble of runout extents produces a probability envelope for infrastructure impact assessment.* 

## **3\. Study Site**

![Fig. 2: Little Professor study site](assets/Little_Professor_image.jpg)  
*Figure 2: Little Professor avalanche path on the south side of Loveland Pass (satellite imagery, summer). The starting zone is the open alpine terrain in the upper-center. US-6 is visible as the switchback road at the bottom and right. The A-Basin Administration Building is at the lower right. The avalanche path runs from the starting zone (elev. \~3,750 m) through the gully to US-6 (\~3,350 m), a vertical drop of \~400 m.*

Little Professor is a southeast-facing (aspect \~138°) avalanche path on the north side of Loveland Pass, directly above US-6. The starting zone spans elevations of approximately 3,650–3,750 m with slope angles of 25–45°. A flat northwest-facing fetch area above the starting zone provides a wind-loading source that builds deep wind slabs on the upper slope. The path terminates at US-6 (Loveland Pass road) and at the Arapaho Basin Ski area parking lot and beginner ski lift, posing a direct infrastructure hazard and a threat to the road itself. 

## **4\. Data**

### **4.1 UAS Snow Depth Surveys**

We collected 23 UAS surveys between November 2025 and March 2026 using a DJI Matrice 4 drone with PPK georeferencing, processed in SiteScan using Structure-from-Motion photogrammetry. Each survey produces a Digital Surface Model (DSM) at approximately 5 cm resolution. Snow depth is computed as HS \= snow-on DSM − bare-ground DSM.

**Bare-ground reference:** The November 10, 2025 bare-ground DSM serves as the reference surface. It was acquired before the first snowfall of the season. 

![Fig. 3: UAS snow depth map](assets/fig_03_snow_depth_map.png)   
*Figure 3: Snow depth map from the January 14, 2026 UAS survey (4 days before the avalanche event). Depth ranges from 0 m (bare ground, dark blue) to \>3 m (wind-loaded areas, red). The wind pillow at the upper-left of the starting zone is clearly visible.* 

**Co-registration:** The bare-ground DSM required co-registration to the survey coordinate frame (2.776 m Y-axis shift). All surveys and the bare-ground DSM share NAD83(2011) / UTM Zone 13N \+ NAVD88 height (EPSG:6342 \+ 5703). Co-registration was verified by comparing origins: dx \= 0.000 m, dy \= 0.000 m after correction.

**Survey quality:** A season-wide audit ([see 25\_26\_Season\_Survey\_Data\_Quality\_Review.pdf](https://docs.google.com/document/d/1DFycZZoSKWsxT0hdTxou3pjii2yyaE8fn0TB8G-yJTg/edit?usp=sharing)) identified several data quality characteristics relevant to downstream modeling:

* Absolute snow depth RMSE against ground-probed depths: approximately 0.6 m (based on 13 probe locations from Jan 14, 2026).

* Flight-to-flight differential RMSE: approximately 0.36 m (median), roughly 1.6× better than absolute RMSE. Snow depth *change* between flights is more reliable than absolute depth on any single flight.

* A persistent stake-positive, rock-negative residual pattern at virtual ground control points (vGCPs) accounts for 67% of post-correction variance. This pattern cancels in flight-to-flight differencing.

* Five flights (Dec 12, Dec 31, Feb 12, Mar 2, Mar 8\) carry elevated RMSE (1.35–1.64 m) and should be treated with caution.

* All flights resolved as PPK float-only (Q=2) over a 47.9 km baseline. No fixed-ambiguity solutions were achieved.

The fundamental limitation is the absence of surveyed ground control points (GCPs). Each flight is georeferenced independently, producing meter-scale relative positioning errors between flights. The planned fix, RTK-surveyed ground control for the 2026/27 season, addresses alignment at the source rather than through post-processing corrections.

**Implications for the model chain:** The 0.6 m absolute RMSE propagates into SNOWPACK as HS forcing uncertainty. At the cluster scale (\~5 m diameter, \~7 pixels), spatial averaging reduces per-cluster error, but systematic biases (e.g., aspect-correlated residual structure) are not removed by averaging. The operational performance monitoring system (§11 in [operational\_design.md](https://github.com/ronimos/snowpack_model_feeder/blob/main/docs/operational_mode_design_concept.md)) tracks survey correction magnitude as a measure of gap-fill model accuracy.

### **4.2 Weather Station Data**

Hourly meteorological data from two CAIC A-Basin stations:

* **CAABT** (summit/weather): temperature, humidity, wind speed/direction, solar radiation, precipitation

* **CAABM** (base): snow depth (in tenths of inches, converted via ÷10 × 2.54 to cm)

All timestamps are UTC. Station data gaps are filled using the refill algorithm in step\_smet. The longest gap observed was 34 hours (Dec 18, 2025). Station depth data provides the hourly HS evolution between UAS surveys; the gap-fill model distributes this station-scale signal spatially using the transport patterns from the surveys.

### **4.3 Avalanche Observations**

CAIC avalanche observations are fetched via the CAIC API (paginated, last 100 records, observed\_at as primary date field). The Jan 18, 2026, D2 skier-triggered event on Little Professor is the primary validation case. The observed release area (3,902 m²) was delineated from UAS survey differencing using the min-kernel detection method (see [avalanche\_handling.md](https://github.com/ronimos/snowpack_model_feeder/blob/main/docs/avalanche_handling.md)).

## **5\. Model Chain**

### **5.1 Snow Depth Distribution (snowpack-model-feeder)**

The snowpack-model-feeder pipeline transforms periodic UAS surveys and continuous station weather data into hourly per-cluster SMET forcing files for SNOWPACK. The pipeline has eight steps:

**resample** → **transport** → **features** → **train** → **avalanche** → **cluster** → **gap\_fill** → **smet**

The core challenge is temporal: UAS surveys provide high-resolution spatial snapshots every 1–2 weeks, but SNOWPACK needs hourly forcing. The gap-fill model disaggregates station-scale hourly dHS using the spatial transport patterns observed between consecutive surveys. See [TRANSPORT\_MODELS.md](https://github.com/ronimos/snowpack_model_feeder/blob/main/docs/TRANSPORT_MODELS.md) for details on the transport models.

**Clustering:** The 1 m DEM is partitioned into \~6,600 clusters using PCA (95% variance) \+ MiniBatch K-means with recursive splitting. The stopping criterion is max\_cluster\_std\_m \= 0.08 m (8 cm intra-cluster HS standard deviation) with a minimum cluster size of 4 pixels. Each cluster receives a single SMET file and a single SNOWPACK simulation. See clustering.md for the full algorithm.

![Fig. 4: Cluster map](assets/fig_04_cluster_map_6565.png)   
*Figure 4: Cluster partitioning for the Little Professor domain. Left: 6,565 clusters (64,105 cells) overlaid on hillshade, each colour representing a distinct cluster with its own SNOWPACK simulation. Centre: mean HS trajectories for 30 representative clusters showing the diversity of accumulation patterns — upper-slope wind pillows accumulate 3× faster than wind-scoured areas. Right: cluster size distribution (median \= 7 pixels, range 1–103).*

### **5.2 Snowpack Simulation (SNOWPACK)**

SNOWPACK is run independently at each cluster location. The simulation evolves the full stratigraphic profile (layer-by-layer density, grain type, grain size, temperature, stability indices) from bare ground through the season. Key configuration: ENFORCE\_MEASURED\_SNOW\_HEIGHTS=TRUE — SNOWPACK adjusts its internal HS to match the SMET forcing at each timestep, adding or removing surface layers as needed.

Output is stored as .pro files (profile time series) and aggregated into a Zarr store for efficient analysis. The Zarr contains \~30 per-layer variables (density, grain\_type, sk38, ssi, sn38, shear\_strength, Lambda, etc.) at all cluster locations and all timesteps.

**Avalanche reinitialization:** When an avalanche is detected (from UAS survey differencing), the affected cluster .sno files are scoured by removing layers from the top down to the slab/WL interface depth. A two-pass SNOWPACK workflow (run to event → scour → rerun from event) handles this. See avalanche\_handling.md for the detection and reinitialization workflow.

### **5.3 Weak Layer Detection**

The failure plane is identified by split\_wl\_slab(), which scans upward from the snowpack base through FC (grain type 4xx) and DH (grain type 5xx) codes until the first transition to slab grain types (RG, DF, PP, etc.). This targets the persistent basal weak layer — the failure plane for Colorado’s continental slab avalanches.

Stability indices (Sk38, SSI, Sn38) are extracted within a ±5 cm window of the WL/slab interface. The interface-specific Sk38 is the primary trigger selection metric: clusters with Sk38 \< 1.0, tau\_g ≥ 50 Pa, slope ≥ 30°, and slab thickness between 0.5–2.0 m are trigger candidates for skier-triggered scenarios.

**Comparison with SNOWPACK’s built-in stability indices:** SNOWPACK’s structural stability index S’ (stab\_deformation\_rate) was evaluated as an alternative WL detection method. S’ saturates at 6.0 for all clusters with zero spatial discrimination between the release area and adjacent stable terrain. The grain-type method finds the WL at a median 1.62 m depth (basal), while profile-wide minimum Sk38 is at a median 0.17 m (near-surface). For persistent slab avalanches in Colorado, depth matters more than index value — the grain-type method correctly targets the failure plane. See compare\_wl\_methods.py for the full comparison.

![Fig. 5: WL detection method comparison](assets/fig_05_wl_method_comparison.png)  
*Figure 5: Weak layer detection method comparison at the Jan 18 snapshot. Left: grain-type WL depth vs S’ depth — correlation is high (r \= 0.94) but S’ is biased 1.12 m toward the surface. Centre: depth difference histogram showing S’ consistently finding shallower layers. Right: boxplot of WL depth by method. The grain-type method targets the basal persistent WL at \~1.6 m; S’ and profile-wide min Sk38 find near-surface instabilities.* 

### **5.4 Release Area Geometry**

The release area is determined using a physics-based BFS crack-propagation model after Meloche et al. (2025). Starting from the trigger cluster, the model propagates the crack outward through the cluster neighborhood graph, arresting where slab properties change sufficiently to stop fracture propagation.

**Arrest criteria (9 conditions, in order):**

1. Outside start zone KML boundary

2. Distance caps (upslope ≤ A\_ca, downslope ≤ stauchwall distance, lateral ≤ Gaume width)

3. Downslope slope \< 28° (stauchwall)

4. tau\_g \< 50 Pa (absolute floor)

5. tau\_g \< 40% of trigger’s tau\_g (relative drop, downslope/lateral only)

6. Slab thickness \< 0.3 m (too thin for fracture)

7. Lambda \< 0.5 m (slab too weak/compliant)

8. |ΔΛ/Λ| \> 50% (Lambda discontinuity)

9. |Δh/h| \> 25% (slab thickness discontinuity)

We applied relative tau\_g arrest (\#5) only to downslope and laterally weak-layer crack propagation; upslope propagation is driven by stored elastic energy and naturally has lower local tau\_g as terrain flattens toward the ridge.

We also developed a probabilistic boundary model (logistic regression on cluster-pair transitions) in parallel. However, it currently has insufficient discriminative power (ROC-AUC \= 0.586, 0% boundary recall) with a single training event. It requires ≥3–5 events to produce meaningful predictions.

See [release\_area\_geometry.md](https://github.com/ronimos/snowpack_model_feeder/blob/main/docs/release_area_geometry.md) for the complete method description, calibrated parameters, and validation.

![Fig. 6: Meloche feature fields](assets/fig_06_meloche_fields_2026-01-17.png)  
*Figure 6: Distributed Meloche et al. (2025) feature fields at the cluster scale. Panels show slab thickness, Lambda (elastic length), tau\_g (driving stress), theta (WL shear strength gradient), A\_ca (crack arrest length), and slope angle. The observed release boundary (red) overlays the strongest tau\_g contrast (2.31× ratio at the boundary).*

![Fig. 6: Crack propagation animation](assets/fig_07_crack_propagation.gif)    
*Figure 7: BFS crack propagation from trigger cluster, showing arrest reasons at each boundary segment. Blue \= propagated clusters; coloured boundaries show arrest type (yellow \= tau\_g \< 50 Pa, orange \= distance cap, teal \= thickness discontinuity, purple \= Lambda discontinuity, red \= stauchwall). Note: regenerate with current arrest criteria (includes relative tau\_g, min Lambda, lateral cap) after BFS recalibration.*

### **5.5 Scenario Ensemble**

The system produces multiple daily scenarios by varying three axes:

* **Trigger location** (n\_triggers, typically 1–5): uses the top-ranked trigger candidates by Sk38

* **Size factor** (0.70–1.30): scales the BFS arrest thresholds, producing smaller/larger release polygons

* **Depth percentile** (P10, P50, P90): scales the depth raster within the release polygon


Each scenario produces a release polygon (GeoJSON), a depth raster (ASCII grid \+ PRJ sidecar), and a parameter file (JSON) for com1DFA. Scenario weights are assigned based on Sk38 ranking and depth percentile probability. See scenario\_writer.py for the output format.

### **5.6 Flow Dynamics (com1DFA / AvaFrame)**

The release polygons and depth rasters feed com1DFA (AvaFrame), which simulates the flowing avalanche using a depth-averaged flow model. An initial 75-scenario ensemble run (May 11, 2026\) produced the following results:

| Metric | Value |
| :---- | :---- |
| Scenarios | 75 |
| Friction model | samosAT Medium |
| Mesh resolution | 1.0 m |
| Entrainment | Disabled |
| Max runout distance | 806 m |
| Scenarios reaching road | 1 / 75 |
| P(reach road) | 0.01 |
| Envelope area (P ≥ 0.05) | 60,550 m² |
| Max impact velocity | 31.6 m/s |
| Max flow depth | 5.1 m |
| Max pressure | 315 kPa |

Voellmy parameters μ and ξ are currently set by the samosAT Medium friction model defaults — not yet calibrated to Little Professor. Calibration against the Jan 18 observed runout is planned. Entrainment is disabled pending resolution of a com1DFA raster-depth input issue.

![Fig. 8: AvaFrame runout probability envelope](assets/fig_08_Jan_18_AvaFrame.jpg) 
*Figure 8: Runout probability envelope from the 75-scenario com1DFA ensemble (preliminary, samosAT Medium friction). The probability heatmap (yellow \= high, blue \= low) shows the most likely flow path from the release zone to US-6 (orange line, lower right). Red contours show individual scenario extents. One scenario (1/75) reaches the road. The interactive viewer is available at: https://nwp.mtnweather.info/val/SNOWPACK\_runout\_260511\_newViewer.html*

The ensemble of runout extents produces a probability envelope:

**P(exceedance at road) \= P(release at this location) × P(runout reaches road | release)**

This is the operationally actionable output: the probability that an avalanche starting from this release area deposits debris on the road.

## **6\. Validation: January 18, 2026 Event**

### **6.1 Event Description**

![Fig. 9: Jan 18 avalanche photo](assets/fig_09_jan18_avalanche_photo.png)  
*Figure 9: The January 18, 2026 D2 skier-triggered slab avalanche on Little Professor, viewed from Loveland Ski Area. The release area, track, and deposit are visible. Five ski tracks are present within 10–20 m of the right flank.*

On January 18, 2026, a skier triggered a D2 slab avalanche on the Little Professor path. The avalanche failed on basal faceted layers (FC/DH) at approximately 1.6 m depth, with a release area of 3,902 m² measured from UAS survey differencing (Jan 14 → Jan 20). Five ski tracks were visible within 10–20 m of the right flank boundary.

### **6.2 Release Area Prediction**

Table 1: BFS output for the simulated Jan 18, 2026 snowpack. 

| Metric | Value |
| :---- | :---- |
| Observed release area | 3,902 m² |
| Modeled release area (BFS, sf=1.0) | \~1,500–3,200 m²  |
| Area ratio (model/observed) | \~0.39–0.83  |
| Trigger location | Inside observed release area (upper slope) |
| Crown position | Consistent with the observed crown |

![Fig. 10: Release polygon validation](assets/fig_10_simulated_avalanche_boundries.png) 
*Figure 10: Release polygon comparison for the Jan 18 event. Blue \= Meloche-derived BFS polygon; red \= observed release area (3,902 m²); green \= start zone boundary. The star marks the trigger cluster centroid. Current result with corrected grain-type extraction and Nov 10 bare-ground DEM.*

The BFS model correctly places the trigger in the upper portion of the observed release area. In fact, none of the selected five likely trigger points were in the area where we observed ski stacks outside the avalanche area. The polygon overlaps the observed release well in the upper and left flanks. The key boundary discriminator is tau\_g contrast (2.31× between release and adjacent median values).

### **6.3 Avalanche Detection**

![Fig. 11: Min-kernel avalanche detection](assets/fig_11_minkernel_detection.png)  
*Figure 11: Avalanche release boundary detected from dHS between Jan 14 and Jan 20 surveys using the min-kernel method (kernel\_size=7, threshold\_sigma=1.2). IoU \= 0.55–0.64 against a hand-drawn boundary.*

The min-kernel detection method identifies the release boundary from dHS between pre-event (Jan 14\) and post-event (Jan 20\) surveys with IoU \= 0.55–0.64 against a hand-drawn boundary. See avalanche\_handling.md for the method comparison (min-kernel vs Canny+watershed).

### **6.4 Stability Comparison**

Table 2: stability indices that were evaluated for their ability to discriminate between the release area and adjacent stable terrain:

| Index | Release area | Adjacent | Discrimination |
| :---- | :---- | :---- | :---- |
| Sk38 at basal WL (grain-type interface) | median 0.89 | median 0.97 | Moderate |
| tau\_g (driving stress) | median 536 Pa | median 232 Pa | Strong (2.31×) |
| Lambda (elastic length) | median 1.68 m | median 1.27 m | Moderate (1.32×) |
| S’ (stab\_deformation\_rate) | 6.0 (saturated) | 6.0 (saturated) | None |
| Profile-wide min Sk38 | 0.07 | 0.07 | None |

tau\_g provides the strongest spatial discrimination. S’ and profile-wide min Sk38 are useless for this snowpack type — they saturate or find near-surface instabilities everywhere. The grain-type method correctly targets the basal persistent WL at 1.6 m depth.

---

## **7\. Limitations**

### **7.1 Input Data**

* **Survey accuracy:** 0.6 m absolute RMSE (single validation point). The 0.36 m differential RMSE is more relevant for the transport model but still represents significant uncertainty at the slab-thickness scale.

* **No surveyed GCPs:** Flight-to-flight positioning is not reliable at sub-meter level. Meter-scale horizontal shifts produce aspect-correlated snow depth biases that propagate into SNOWPACK.

* **Single validation event:** All calibration (arrest thresholds, trigger selection filters) is based on one D2 event on one path. Multi-event, multi-path validation is required before operational deployment.

* **Station representativeness:** The gap-fill model assumes station dHS is representative of domain-wide accumulation patterns. Wind events can invalidate this assumption between surveys.

### **7.2 Model Chain**

* **Cluster resolution (\~5 m):** Smooths slab property gradients relative to SMP-scale measurements. Gradient-based arrest criteria reflect this averaging.

* **Grain-type WL detection:** Assumes the deepest FC/DH layer is always the failure plane. Mid-pack WLs from buried surface hoar or crust-facet combinations are not targeted.

* **Stability window (±5 cm):** The Sk38 extraction window size affects which layers contribute to the stability assessment. Sensitivity to this parameter has not been fully characterized.

* **BFS arrest thresholds:** Under recalibration. The LAMBDA\_JUMP\_FACTOR and THICKNESS\_JUMP\_FACTOR values were originally calibrated with a grain-type classification bug (\>= 4 instead of \== 4 or \== 5). The corrected Meloche parameters produce smoother spatial gradients at the boundary.

* **Probabilistic boundary model:** ROC-AUC \= 0.586 with zero boundary recall. Requires more training events.

* **Voellmy parameters uncalibrated:** μ \= 0.155 and ξ \= 1500 m/s² are literature defaults, not calibrated to Little Professor.

* **S’ mislabeled as P\_unstable:** The initial comparison used SNOWPACK’s stab\_deformation\_rate (S’) and mislabeled it as P\_unstable. The Mayer et al. (2022) P\_unstable is a separate random forest model not currently implemented. S’ saturation at 6.0 is a valid finding; the label has been corrected.

### **7.3 Operational**

* **No NWP integration yet:** The 72-hour forecast mode is designed but not implemented. Daily forward mode and survey correction mode are implemented in shell scripts.

* **Single path:** Currently validated on Little Professor only. Extension to Widowmaker, Muleshoe, and other US-6 paths is planned.

## **8\. Future Work**

### **Near-term (ISSW 2026\)**

* Calibrate Voellmy μ/ξ against Jan 18 observed runout (Val)

* Produce runout probability envelope for US-6 road polygon

* Cross-date Meloche analysis (Jan 10, 14, 17, 18\) showing instability evolution toward the event

* Evaluate Mayer et al. (2022) P\_unstable RF model with published Alpine weights on Colorado data

### **Medium-term (2026/27 season)**

* RTK-surveyed ground control for UAS surveys (summer 2026 field plan)

* Automated daily operational pipeline with NWP 72-hour forecasts

* Multi-event calibration (accumulate 3–5 events for probabilistic boundary model training)

* WindNinja wind field library for improved RF transport model

* Extension to the Seven Sisters, Widowmaker, and Bethel paths

* Real-time hazard dashboard for operations teams

* 

### **Long-term**

* Colorado-specific instability model trained on accumulated events

* Multi-path corridor-scale hazard assessment

* Integration with CDOT road management systems

## **9\. Related Documents**

| Document | Contents |
| :---- | :---- |
| release\_area\_geometry.md | Complete release area method: trigger selection, BFS arrest criteria, probabilistic model, calibrated parameters, validation |
| operational\_design.md | Operational architecture: daily forward mode, NWP forecasts, survey correction, efficiency, cluster management, monitoring, performance reporting |
| avalanche\_handling.md | Avalanche detection (min-kernel vs Canny+watershed), SNOWPACK reinitialization workflow, transport correction |
| TRANSPORT\_MODELS.md | Snow transport model: wind-energy disaggregation, RF regression, gap filling |
| clustering.md | Cluster algorithm: PCA \+ K-means \+ recursive splitting \+ contiguity enforcement |
| TODO.md | Prioritized task list with operational mode, NWP forecast, cluster management, P\_unstable evaluation |
| 25\_26\_Season\_Survey\_Data\_Quality\_Review.pdf | Season-wide UAS survey audit: co-registration, cross-flight repeatability, probe comparison, outlier analysis, summer 2026 field plan |

---

## **10\. References**

Gaume, J., van Herwijnen, A., Chambon, G., Wever, N., and Schweizer, J. (2017). Snow fracture in relation to slab avalanche release: critical state for the onset of crack propagation. *The Cryosphere*, 11(1), 217–228.

Mayer, S., van Herwijnen, A., Techel, F., and Schweizer, J. (2022). A random forest model to assess snow instability from simulated snow stratigraphy. *The Cryosphere*, 16(11), 4593–4615.

Meloche, F., Bhatt, A., Gauthier, F., and Hébert-Houle, P.-É. (2025). Spatial variability of slab properties as a control on avalanche release area geometry. *Journal of Glaciology*.

Schweizer, J. and Jamieson, J. B. (2007). A threshold sum approach to stability evaluation of manual snow profiles. *Cold Regions Science and Technology*, 47(1–2), 50–59.

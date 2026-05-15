# P_unstable Evaluation Method

Evaluation of the Mayer et al. (2022) random-forest P_unstable model on the
Little Professor avalanche path, with focus on whether it spatially
discriminates between the Jan 18, 2026 D2 release area and adjacent
skied-but-not-triggered terrain.

## Goal

Determine whether the Alpine-trained Mayer P_unstable random forest provides
useful spatial discrimination on Colorado continental snowpack. Specifically:
does it flag clusters within the Jan 18 release area as more unstable than
clusters in adjacent terrain that skiers traversed without triggering?

This is positioned as a *complement* to the gradient-arrest physics model
for release-area geometry, not a replacement. The physics model captures
spatial discrimination via slab-property gradients; this evaluation tests
whether a layer-resolved profile classifier adds an independent line of
evidence.

## Pipeline

Three-stage pipeline running across two Python environments. Decoupled by
serialized intermediate files so version constraints don't cascade.

| Stage | Script | Env | Output |
|---|---|---|---|
| 0. Side-load rc | `add_critical_cut_length.py` | main | new Zarr var `critical_cut_length` |
| 1. Extract features | `extract_punstable_features.py` | main | `punstable_features.npz` |
| 2. Predict | `predict_punstable.py` | `.venv-punstable` | `punstable_predictions.npz` |
| 3. Ingest back | `ingest_punstable_to_zarr.py` | main | new Zarr vars `p_unstable_max*` |
| 4a. Validate (spatial) | `validate_punstable_spatial.py` | main | maps + metrics |
| 4b. Validate (dist.) | `plot_punstable_distributions.py` | main | violin + ECDF figure |

### Why two environments

The published Mayer model (`RF_instability_model.sav`) was trained with
scikit-learn 0.22.1 and cannot be loaded by modern sklearn — the tree
internals (`sklearn.tree._tree.Tree.__setstate__`) changed dtype/layout
between 0.22 and 1.x. Loading with sklearn 1.8 throws
`UnpicklingError: invalid load key, '\x00'` after the outer
RandomForestClassifier wrapper parses correctly. Verified by walking the
pickle with `pickletools.dis` (dies at position 1065, inside the first
tree's numpy array).

Created an isolated env `~/code/external/random_forest_snow_instability_model/.venv-punstable`
with Python 3.8 + sklearn 0.22.1 + pandas 1.1.0 + numpy 1.19.5, matching
SLF's `environment.yml`. The model loads cleanly there with
`joblib.load` (NOT `pickle.load` — the file is a joblib pickle with
NumpyArrayWrappers; bare pickle hits raw float bytes after the metadata
wrapper and fails).

## Feature computation

Six features in the order required by Mayer's trained RF
(`extract_punstable_features.py`):

1. **viscdefrate** — viscous deformation rate of WL layer, raw `1e-6/s`
   scale, direct from Zarr var `viscous_deformation_rate` (SNOWPACK 0523).
   Negative values represent compressive settling and are valid; the model
   was trained with this sign convention.

2. **rcflat** — critical crack length on the flat (Richter 2019), computed
   in Python from layer density, grain size, and shear strength. *NOT* the
   `critical_cut_length` we side-loaded from SNOWPACK 0606. SNOWPACK
   computes rc on the actual slope angle; Mayer's RF was trained on the
   flat-surface form. Our side-loaded 0606 is preserved in the Zarr as an
   independent reference value.

3. **sphericity** — direct from Zarr (SNOWPACK 0509).
4. **grainsize** — direct from Zarr in mm (SNOWPACK 0511).
5. **pendepth** — skier penetration depth via Jamieson & Johnston (1998)
   with the Bellaire (2006) pre-factor of 0.8: `Pk = min(0.8 * 43.3 /
   rho_top30, (HS - top_crust) / 100)`. Top-30cm slab density is
   thickness-weighted. Crust detection is per-layer (single MFcr layer
   with grain type 772, rho > 500, slope-projected thickness > 3 cm)
   rather than Mayer's exact contiguous-block accumulation — simplification
   noted in script header; unlikely to bite on Colorado snow where strong
   crusts are rare.
6. **slab_rhogs** — thickness-weighted mean of `rho / gs` over the slab
   *above* the candidate WL layer: `sum(rho_i * thick_i / gs_i) /
   sum(thick_i)`. This is *not* equivalent to `mean(rho) / mean(gs)`.

The topmost layer of each profile is naturally excluded (no slab above for
features 2 and 6).

### Unit and convention notes

- **height** is in cm despite the Zarr's `units: 'm'` CF attribute — xsnow's
  metadata is wrong but the underlying values are SNOWPACK-native cm
  (verified by comparing `HS` to the top-of-profile height).
- **viscous_deformation_rate** is in `1e-6/s`, matching what Mayer trained
  on. xsnow did not normalize to SI.
- **grain_size** in mm, **density** in kg/m³, **shear_strength** in kPa,
  all as Mayer expects.
- Layers indexed bottom-up (first = closest to ground), matching SNOWPACK
  and Mayer's convention.

### Per-cluster summary derivation

`ingest_punstable_to_zarr.py` derives five 2D fields from the full
`(location, time, layer)` P_unstable array:

- `p_unstable_max` — max across layers per profile.
- `p_unstable_max_layer` — index of the worst layer.
- `p_unstable_max_depth_cm` — depth below surface of worst layer.
- `p_unstable_max_height_cm` — height above ground of worst layer.
- `p_unstable_max_grain_type` — SNOWPACK 3-digit code at worst layer.

Per Mayer's convention, `P_unstable = predict_proba(...)[:, 0]` since the
RF was trained with class 0 = unstable, class 1 = stable (verified via
`model.classes_ == [0., 1.]`).

## Validation methodology

The honest counterfactual is **release-area clusters vs skied-but-not-
triggered clusters**, not release-area vs entire-slope. The original
in/out-of-release test diluted the "out" set with terrain that nobody
ever skied and that lacked sufficient loading — essentially guaranteed
overlap regardless of model quality.

Two polygons, hand-digitized from the post-event UAS orthomosaic:

- `data/boundaries/avalanche_release_area.geojson` — Jan 18 release area.
- `data/boundaries/skies-non-release.geojson` — adjacent terrain showing
  visible ski tracks but no release.

Cluster classification: a cluster belongs to a polygon if ≥50% of its
pixels in `cluster_map.tif` fall inside the polygon. Clusters appearing
in both sets are removed from the skied set.

Snapshot: 2026-01-18T18:00:00 UTC (11 AM MST, ~3-6h before the trigger).

## Findings

### Internal validation: grain type at the worst layer

At Jan 18 18:00 UTC across all 6565 clusters in the domain:

| Grain type | Clusters | % |
|---|---|---|
| FC (faceted crystals) | 6274 | 95.6% |
| DH (depth hoar) | 283 | 4.3% |
| Other | 8 | 0.1% |

The Alpine-trained RF picks basal faceted/depth-hoar grains as the failure
plane at 99.9% of locations — the Colorado continental signature, without
being told to look there. This is strong evidence that the model is doing
physics (responding to the layer's stratigraphic properties) rather than
pattern-matching to Swiss-specific surface hoar features that don't exist
in this snowpack.

This implicitly cross-validates `split_wl_slab()`: the basal-WL search and
Mayer's RF converge on the same layer 99.9% of the time.

### Spatial discrimination: release vs skied

At Jan 18 18:00 UTC:

| Stat | Release (n=401) | Skied (n=775) | Δ |
|---|---|---|---|
| median | 0.349 | 0.340 | +0.009 |
| p75 | 0.385 | 0.398 | **-0.013** |
| p95 | 0.587 | 0.504 | +0.083 |
| max | 0.885 | 0.846 | +0.040 |

Hit rates at Mayer thresholds:

| Threshold | Release | Skied | Lift |
|---|---|---|---|
| P ≥ 0.50 | 7.2% (29/401) | 5.4% (42/775) | 1.3× |
| P ≥ 0.77 | 1.5% (6/401) | 0.6% (5/775) | 2.5× |
| P ≥ 0.92 | 0.0% (0/401) | 0.0% (0/775) | — |

Mann-Whitney U (release > skied): p = 3.78e-9. Statistically significant
but operationally meaningless — the median difference is 0.009 and the
distributions overlap heavily in the bulk. At p75, skied is *higher* than
release (a small inversion). The Mann-Whitney p-value reflects the large
sample size, not a usable effect.

The signal lives in the upper tail (p95 differs by 0.083, ~10× the median
gap), and the 2.5× hit-rate lift at the P≥0.77 threshold favors release.
But absolute counts are small (6 vs 5) and the threshold-crossing
distribution is noisy at this scale.

### Honest conclusion

Mayer P_unstable correctly identifies *what* the weak layer is (99.9%
basal FC/DH) but cannot reliably tell us *where* on the slope it will
fail. There is marginal upper-tail discrimination between release and
skied terrain but the bulk of the two distributions is indistinguishable.

This is consistent with the model's design: Mayer's RF was trained on
point profiles to answer "is this profile stable" — not on spatial fields
to answer "which profile on this slope will trigger." It implicitly
assumes that profile-internal properties are what matter, which is true
for the layer identification but inadequate for spatial trigger location.

The gradient-arrest physics model captures the spatial discrimination
Mayer's RF lacks (area ratio 0.85 on Jan 18). Together the two methods
answer different questions and the dual-model design is empirically
justified.

## Known limitations

- **Pk crust detection** simplified from Mayer's contiguous-block
  accumulation. Unlikely to bite on Colorado snow but worth flagging.
- **Slope angle** is a global default (38°) used only in the Pk crust
  thickness projection. Per-cluster slope would be a refinement; effect
  is small because MFcr layers are rare.
- **Single-timestep snapshot.** Time-evolution analysis (does P_unstable
  climb harder pre-event at the trigger cluster than at adjacent clusters?)
  not pursued in this round — judged lower-priority than getting the
  spatial counterfactual right.
- **sklearn version warning** when loading the published model is
  suppressed but not eliminated. Predictions on test inputs were monotonic
  in the expected direction during smoke testing; not formally validated
  against SLF reference outputs.
- **Side-loaded `critical_cut_length`** lives in the Zarr as `0606` but
  not as a derived xsnow variable. `build_zarr_chunked.py` should
  eventually parse 0606 in-line; until then, `add_critical_cut_length.py`
  has to run after any Zarr rebuild.

## References

- Mayer, S., van Herwijnen, A., Techel, F., Schweizer, J. (2022). A random
  forest model to assess snow instability from simulated snow stratigraphy.
  *The Cryosphere* 16, 4593–4615.
  <https://doi.org/10.5194/tc-16-4593-2022>
- SLF model repo: <https://gitlabext.wsl.ch/mayers/random_forest_snow_instability_model>
- Jamieson, J. & Johnston, C. (1998). Refinements to the stability index for
  skier-triggered dry-slab avalanches. *Annals of Glaciology* 26, 296–302.
- Bellaire, S. (2006). Modeling skier-triggered avalanches. Diploma thesis.
- Richter, B. et al. (2019). Modeling the critical crack length using SNOWPACK.

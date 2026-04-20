# Future: Physics-Based Transport Model Integration

## Current Approach (v1): Empirical Transport + RF Regression

The current pipeline learns transport patterns empirically from survey pairs
and generalizes with a terrain regression. This works well when:
- Wind patterns during gap-fill are similar to training periods
- The site has a dominant wind direction (W-NW at this site)
- Settlement is roughly uniform (tracked by the station)

It fails or degrades when:
- Novel wind directions occur (outside training distribution)
- Snow surface erodibility varies (fresh vs. sintered)
- Sublimation losses differ between periods (temp/humidity dependent)
- Transport occurs at sub-hourly scales that matter for stratigraphy

## Priority Improvements (ranked by impact-to-effort ratio)

### 1. WindNinja Wind Field Downscaling (HIGH IMPACT, MODERATE EFFORT)

Replace Sx as the spatial wind predictor with actual modeled wind fields.

**Why:** Sx is a geometric proxy. WindNinja solves mass-consistent flow over
the DEM and produces actual wind speed/direction at each cell. This would:
- Give physically meaningful wind speeds for threshold calculations
- Handle flow separation, speed-up over ridges, etc.
- Enable direct computation of transport flux

**Integration point:** `spatial_model.py → compute_wind_weighted_sx()`
Replace with a function that:
1. Precomputes WindNinja wind fields for N discrete directions (e.g., 16)
   at a few reference speeds (e.g., 5, 10, 15 m/s)
2. At runtime, interpolates to actual hourly station wind → distributed field
3. Uses distributed wind speed in place of Sx for RF features AND
   for the hourly disaggregation weighting

**Resources:**
- WindNinja is open source: https://github.com/firelab/windninja
- Can run from CLI with DEM + station wind as input
- Precomputed library approach: Marsh et al. (2023)
  "The Canadian Hydrological Model" Section 4.4

**Estimated effort:** 2-3 days to build the wind library + interpolation.
WindNinja itself runs in seconds for a 500×500m domain at 1m.


### 2. Transport Threshold from Snow Surface Age (MODERATE IMPACT, LOW EFFORT)

Add a simple erodibility factor based on time since last snowfall.

**Why:** Fresh snow erodes at ~5 m/s threshold. After 24-48h of sintering,
threshold rises to 10-15+ m/s. Currently we apply transport proportional to
wind² regardless of surface state.

**Integration point:** `spatial_model.py → gap_fill_period()`
Add erodibility scaling:
```python
hours_since_snowfall = ...  # from station HN24 or ΔHS
erodibility = max(0, 1 - hours_since_snowfall / 48)  # linear decay
hourly_transport = transport_field * frac * erodibility
```

This is a crude parameterization but captures the first-order effect.
The station's HN24/snow24h field provides the needed information.

**Estimated effort:** 0.5 days.


### 3. Sublimation Loss Parameterization (MODERATE IMPACT, LOW EFFORT)

Account for mass lost during blowing snow suspension.

**Why:** 10-30% of transported mass sublimates. Our transport fields are
calibrated to net deposition (surveys), so sublimation is implicitly included
in the training data. But when applying a transport field to a new period
with different temp/RH, the sublimation fraction should change.

**Integration point:** `spatial_model.py → gap_fill_period()`
Scale transport by a sublimation efficiency factor:
```python
# Pomeroy & Li (2000) sublimation rate parameterization
# Fraction of transport that deposits (vs sublimates)
deposition_fraction = f(temperature, humidity, wind_speed, fetch_distance)
hourly_transport *= deposition_fraction
```

**Reference:**
- Pomeroy & Li (2000), J. Geophys. Res., 105(D21), 26619-26634
- Liston & Sturm (1998), J. Hydrometeorol.

**Estimated effort:** 1 day.


### 4. FSM2oshd / Quéno et al. Approach (HIGH IMPACT, HIGH EFFORT)

Full intermediate-complexity transport model.

**Why:** Combines WindNinja wind fields + simplified blowing snow scheme +
FSM2 snowpack model. This is the "right" way to do it for operational
applications. Quéno et al. (2024) showed it works at 25-250m resolution
for Swiss operational hydrology.

**Key insight from their approach:**
- Precompute WindNinja wind fields for 12 sectors × 3 speed classes
- At each timestep, look up the appropriate wind field
- Apply Pomeroy (1993) equilibrium saltation transport formula
- Transport is function of (local wind speed - threshold)^3
- Sublimation parameterized following Pomeroy & Li (2000)

**Integration architecture:**
The cleanest integration would be to add a `TransportModel` abstract class
that both our empirical approach and a physics-based approach implement:

```python
class TransportModel:
    def compute_hourly_transport(self, dem, hs, wind_speed, wind_dir,
                                  temperature, ...) -> np.ndarray:
        '''Return transport flux [m/timestep] at each grid cell.'''
        raise NotImplementedError

class EmpiricalTransport(TransportModel):
    '''Current approach: learned from surveys.'''
    ...

class PhysicsTransport(TransportModel):
    '''WindNinja + saltation/suspension/sublimation.'''
    ...
```

**References:**
- Quéno et al. (2024), The Cryosphere, 18, 3533-3557
  doi:10.5194/tc-18-3533-2024
- FSM2: Essery (2015), https://github.com/RichardEssery/FSM2
- SnowTran-3D: Liston & Sturm (1998), J. Hydrometeorol.

**Estimated effort:** 2-3 weeks for full implementation + validation.


### 5. SnowTran-3D (HIGH IMPACT, HIGH EFFORT)

Full physics-based transport in the Liston SnowModel framework.

**Why:** Most complete treatment of saltation + suspension + sublimation.
Handles variable fetch, two-way coupling with snowpack.

**Challenges for our use case:**
- Written in Fortran, would need Python wrapper or reimplementation
- Requires full distributed meteorological forcing (not just wind)
- The parallel version (Mower et al. 2023) targets large domains
- Our 1m resolution × 500m domain is unusual for SnowTran-3D
  (typically used at 30-100m over km-scale domains)

**Recommendation:** Unless you need the full SnowModel framework,
the FSM2oshd approach is more practical for this application.


### 6. Data Assimilation: Kalman Smoother (MODERATE IMPACT, MODERATE EFFORT)

Use any of the above as the forecast model, surveys as observations.

**Why:** Currently we apply the transport field as a fixed correction.
A Kalman smoother would:
- Use the transport model as the forecast step
- Use surveys as the analysis step (correct state)
- Propagate information backward from future surveys (smoother vs filter)
- Provide uncertainty estimates at every cell and timestep

**Integration point:** Wraps around `gap_fill_period()`.
The forecast step is the current gap-fill; the analysis step adjusts
layer properties (density) rather than resetting HS, preserving
SNOWPACK stratigraphic consistency.

**Reference:**
- Alonso-González et al. (2023), HESS, 27, 4637-4659
  Hyper-resolution ensemble-based snow data assimilation with
  topography-aware covariance matrices


## Architecture Notes

The current `spatial_model.py` is structured around these abstractions:

1. **Spatial predictor:** Sx + terrain features → where does transport happen?
   Replace with: WindNinja distributed wind speed

2. **Temporal disaggregation:** wind² weighting → when does transport happen?
   Replace with: threshold-based transport flux at each timestep

3. **Transport magnitude:** learned from survey pairs → how much transport?
   Replace with: physics (saltation flux equation) calibrated against surveys

4. **Validation:** leave-one-out CV on survey pairs
   Keep this regardless of model complexity

The RF regression remains useful even with physics-based transport —
use it to predict the model's bias (residual from physics prediction)
rather than predicting transport directly. This hybrid approach captures
what the physics misses while maintaining physical consistency.

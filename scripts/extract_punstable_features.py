"""
extract_punstable_features.py

Compute Mayer et al. (2022) P_unstable input features per
(location, time, layer) from the SNOWPACK Zarr, save flat npz for the
punstable env prediction step.

Six features in the order required by Mayer's trained RF:
  1. viscdefrate  — viscous deformation rate, raw 1e-6/s scale (var 0523)
  2. rcflat       — Richter (2019) critical crack length on the FLAT (NOT
                    SNOWPACK var 0606; computed from rho, gs, shear strength)
  3. sphericity   — WL sphericity (var 0509)
  4. grainsize    — WL grain size in mm (var 0511)
  5. pendepth     — skier penetration depth in m (Jamieson-Johnston 1998
                    with Bellaire 2006 pre-factor; simplified crust logic)
  6. slab_rhogs   — thickness-weighted mean of (rho/gs) over the slab

Caveats:
  - Pk crust detection is simplified vs. Mayer's exact algorithm: we require
    a SINGLE MFcr layer (graintype 772, rho > 500) with slope-projected
    thickness > 3 cm, rather than accumulating contiguous MFcr layers. For
    Colorado snowpacks where strong crusts are rare this is unlikely to bite,
    but it's a known deviation.
  - Slope is a CLI default (used only for crust thickness projection). If
    per-cluster slope matters, expose via Zarr attrs or a sidecar.
  - The topmost valid layer in each profile naturally gets NaN for rcflat
    and slab_rhogs (no slab above) and is excluded from the output.

Usage:
  python extract_punstable_features.py --sample-loc 100   # smoke test
  python extract_punstable_features.py                    # full run
"""
import argparse
import time
from pathlib import Path

import numpy as np
import xarray as xr


MAX_LAYERS = 338
RHO_ICE = 917.0            # kg m-3
GS_0 = 0.00125             # m, reference grain size in Richter formula
RC_A = 4.6e-9              # Richter empirical constant
PK_PREFACTOR = 0.8 * 43.3  # m * kg/m3; gives Pk in m when rho in kg/m3
PK_WINDOW_CM = 30.0        # top-of-slab averaging window for Pk
PK_MIN_CRUST_CM = 3.0      # min slope-projected MFcr thickness
MFCR_CODE = 772            # SNOWPACK grain type for melt-freeze crust
MFCR_RHO_MIN = 500.0       # kg m-3

REQUIRED_VARS = [
    'height', 'density', 'grain_size', 'sphericity',
    'shear_strength', 'viscous_deformation_rate', 'grain_type',
]
FEATURE_COLS = ['viscdefrate', 'rcflat', 'sphericity',
                'grainsize', 'pendepth', 'slab_rhogs']


def compute_features(h, rho, gs, sph, strs, vdr, gtype, slope_deg):
    """
    All inputs shape (n_loc, n_time, MAX_LAYERS), bottom-up, NaN-padded above
    the top valid layer. Heights in cm, density kg/m^3, grain size mm, shear
    strength kPa, viscous deformation rate 1e-6/s, grain type SNOWPACK 3-digit
    code.

    Returns dict of (n_loc, n_time, MAX_LAYERS) arrays for the 6 features plus
    a boolean valid_pred mask of rows fit for the RF.
    """
    valid = ~np.isnan(h)

    # Layer thickness: thick_i = height_i - height_{i-1}, thick_0 = height_0
    h_below = np.zeros_like(h)
    h_below[..., 1:] = h[..., :-1]
    thick = np.where(valid, h - h_below, 0.0)

    # Slab integrals from layer i+1 to top.
    # cumsum-from-top[i] = sum over j >= i (after reverse, cumsum, reverse).
    # slab_above[i] = cumsum-from-top[i+1] (excludes layer i itself).
    pad = np.zeros(h.shape[:-1] + (1,))
    rho_thick = np.where(valid, rho * thick, 0.0)
    rt_cumtop = np.cumsum(rho_thick[..., ::-1], axis=-1)[..., ::-1]
    th_cumtop = np.cumsum(thick[..., ::-1], axis=-1)[..., ::-1]
    rt_above = np.concatenate([rt_cumtop[..., 1:], pad], axis=-1)
    th_above = np.concatenate([th_cumtop[..., 1:], pad], axis=-1)

    with np.errstate(invalid='ignore', divide='ignore'):
        rho_sl = rt_above / th_above  # mean slab density above each layer

    # Feature 2 — rc_flat (Richter 2019).
    # Mayer's form: sqrt(a*(rho_wl/rho_ice * gs_wl/gs_0)^(-2))
    #               * sqrt(2 tau_p E' D/sigma_n)
    # Rewritten as sqrt(a)/wl_term * sqrt(2 tau_p E' / (g rho_sl))   (D/sigma=1/(g rho))
    gs_m = gs * 1e-3       # mm -> m
    tau_p = strs * 1e3     # kPa -> Pa
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        eprime = 5.07e9 * (rho_sl / RHO_ICE) ** 5.13 / (1.0 - 0.2 ** 2)
        wl_term = (rho / RHO_ICE) * (gs_m / GS_0)
        rcflat = (np.sqrt(RC_A) / wl_term) * np.sqrt(
            2.0 * tau_p * eprime / (9.81 * rho_sl)
        )

    # Feature 6 — slab_rhogs (thickness-weighted <rho/gs> over slab).
    with np.errstate(invalid='ignore', divide='ignore'):
        rhogs = np.where((gs > 0) & valid, rho * thick / gs, 0.0)
    rg_cumtop = np.cumsum(rhogs[..., ::-1], axis=-1)[..., ::-1]
    rg_above = np.concatenate([rg_cumtop[..., 1:], pad], axis=-1)
    with np.errstate(invalid='ignore', divide='ignore'):
        slab_rhogs = rg_above / th_above

    # Feature 5 — Pk per (loc, time), broadcast to layer dim.
    HS = np.where(valid, h, -1.0).max(axis=-1)         # cm
    depth_from_top = HS[..., np.newaxis] - h
    in_window = valid & (depth_from_top >= 0) & (depth_from_top < PK_WINDOW_CM)

    sum_rt = np.where(in_window, rho * thick, 0.0).sum(axis=-1)
    sum_th = np.where(in_window, thick, 0.0).sum(axis=-1)
    with np.errstate(invalid='ignore', divide='ignore'):
        rho_window = sum_rt / np.where(sum_th > 0, sum_th, np.nan)
    pk_jj = PK_PREFACTOR / rho_window                  # m

    cos_slope = np.cos(np.deg2rad(slope_deg))
    is_thick_mfcr = (
        (gtype == MFCR_CODE) & (rho > MFCR_RHO_MIN) &
        in_window & (thick * cos_slope > PK_MIN_CRUST_CM)
    )
    crust_h = np.where(is_thick_mfcr, h, -1.0)
    top_crust_h = crust_h.max(axis=-1)
    has_crust = top_crust_h > 0
    pk_bound_m = np.where(has_crust,
                          (HS - top_crust_h) / 100.0,
                          HS / 100.0)
    pendepth_2d = np.minimum(pk_jj, pk_bound_m)
    pendepth = np.broadcast_to(
        pendepth_2d[..., np.newaxis], rcflat.shape
    ).copy()

    valid_pred = (
        valid & np.isfinite(vdr) & np.isfinite(rcflat) & np.isfinite(sph) &
        np.isfinite(gs) & np.isfinite(pendepth) & np.isfinite(slab_rhogs)
    )

    return dict(
        viscdefrate=vdr.astype(np.float32),
        rcflat=rcflat.astype(np.float32),
        sphericity=sph.astype(np.float32),
        grainsize=gs.astype(np.float32),
        pendepth=pendepth.astype(np.float32),
        slab_rhogs=slab_rhogs.astype(np.float32),
        valid_pred=valid_pred,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--zarr',
        default='/data/snowpack/little_prof/output/'
                'slope_snowpack.zarr')
    p.add_argument('--out',
        default='/data/snowpack/little_prof/output/'
                'punstable_features.npz')
    p.add_argument('--chunk-loc', type=int, default=500,
                   help='Number of locations to load+compute per chunk')
    p.add_argument('--slope', type=float, default=38.0,
                   help='Slope angle (deg) for Pk crust thickness check')
    p.add_argument('--sample-loc', type=int, default=0,
                   help='If >0, process only this many locations')
    args = p.parse_args()

    ds = xr.open_zarr(args.zarr)
    missing = [v for v in REQUIRED_VARS if v not in ds.data_vars]
    if missing:
        raise RuntimeError(f"Zarr missing required variables: {missing}")

    locations = ds.coords['location'].values.astype(str)
    times = ds.coords['time'].values
    n_loc_total, n_time = len(locations), len(times)
    n_loc = min(args.sample_loc, n_loc_total) if args.sample_loc else n_loc_total

    print(f"Zarr: {n_loc_total} locations  {n_time} times  "
          f"{ds.sizes['layer']} layers")
    print(f"Processing {n_loc} locations (chunk={args.chunk_loc}, "
          f"slope={args.slope}°)")

    chunks_feat, chunks_idx = [], []
    t0 = time.time()

    for cs in range(0, n_loc, args.chunk_loc):
        ce = min(cs + args.chunk_loc, n_loc)
        chunk = ds[REQUIRED_VARS].isel(location=slice(cs, ce)).load()

        feats = compute_features(
            chunk.height.values,
            chunk.density.values,
            chunk.grain_size.values,
            chunk.sphericity.values,
            chunk.shear_strength.values,
            chunk.viscous_deformation_rate.values,
            chunk.grain_type.values,
            args.slope,
        )

        m = feats['valid_pred']
        li, ti, layi = np.where(m)
        flat = np.stack([feats[c][m] for c in FEATURE_COLS], axis=1)
        idx = np.stack([li + cs, ti, layi], axis=1).astype(np.int32)

        chunks_feat.append(flat)
        chunks_idx.append(idx)

        elapsed = time.time() - t0
        pct = ce / n_loc
        eta = elapsed / pct * (1 - pct) / 60 if pct > 0 else 0
        print(f"  loc {cs:5d}:{ce:5d}  +{len(li):>9,} rows  "
              f"[{elapsed:5.0f}s, ETA {eta:4.0f} min]")

    features = np.concatenate(chunks_feat, axis=0)
    indices = np.concatenate(chunks_idx, axis=0)
    print(f"\nTotal: {features.shape[0]:,} valid rows")

    print("\nFeature ranges (finite values only):")
    for i, c in enumerate(FEATURE_COLS):
        col = features[:, i]
        ff = col[np.isfinite(col)]
        if ff.size:
            print(f"  {c:14s} min={ff.min():>11.4g}  "
                  f"median={np.median(ff):>11.4g}  "
                  f"max={ff.max():>11.4g}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to {out} ...")
    np.savez_compressed(
        out,
        features=features,
        indices=indices,
        locations=locations[:n_loc],
        times=times,
        feature_cols=np.array(FEATURE_COLS),
    )
    size_mb = out.stat().st_size / 1e6
    print(f"Done. {size_mb:.0f} MB. Total {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()


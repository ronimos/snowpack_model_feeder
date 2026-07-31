"""
ingest_punstable_to_zarr.py

Read predictions from predict_punstable.py and write them into the SNOWPACK
Zarr as per-profile summary variables (and optionally the full 3D array).

Runs in main env — only needs xarray/zarr/numpy.

Derived 2D variables added to Zarr (dims: location, time):
  p_unstable_max               max P_unstable across layers in profile
  p_unstable_max_layer         layer index of the max (int16, -1 if none)
  p_unstable_max_depth_cm      depth below snow surface of the max layer (cm)
  p_unstable_max_height_cm     height above ground of the max layer (cm)
  p_unstable_max_grain_type    SNOWPACK grain type code at the max layer

With --write-3d, also adds the full (location, time, layer) p_unstable field.

Usage:
  python ingest_punstable_to_zarr.py            # 2D summary only
  python ingest_punstable_to_zarr.py --write-3d # also full 3D
"""
import argparse
import time
from pathlib import Path

import numpy as np
import xarray as xr


SUMMARY_VARS = [
    'p_unstable_max',
    'p_unstable_max_layer',
    'p_unstable_max_depth_cm',
    'p_unstable_max_height_cm',
    'p_unstable_max_grain_type',
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--zarr',
        default='/data/snowpack/little_prof/output/'
                'slope_snowpack.zarr')
    p.add_argument('--predictions',
        default='/data/snowpack/little_prof/output/'
                'punstable_predictions.npz')
    p.add_argument('--write-3d', action='store_true',
                   help='Also write full (location, time, layer) P_unstable')
    p.add_argument('--overwrite', action='store_true',
                   help='Allow overwriting existing variables')
    args = p.parse_args()

    print(f"Loading predictions from {args.predictions}")
    npz = np.load(args.predictions, allow_pickle=False)
    p_unstable = npz['p_unstable']        # (N,) float32
    indices = npz['indices']              # (N, 3) int32: loc, time, layer
    print(f"  {len(p_unstable):,} prediction rows")

    print(f"Opening Zarr: {args.zarr}")
    ds = xr.open_zarr(args.zarr)
    n_loc = ds.sizes['location']
    n_time = ds.sizes['time']
    n_layer = ds.sizes['layer']
    print(f"  {n_loc} locations  {n_time} times  {n_layer} layers")

    # Check for collisions
    target_vars = list(SUMMARY_VARS)
    if args.write_3d:
        target_vars.append('p_unstable')
    existing = [v for v in target_vars if v in ds.data_vars]
    if existing and not args.overwrite:
        raise RuntimeError(
            f"Variables already in Zarr: {existing}. Use --overwrite to "
            f"replace them, or delete them manually: "
            f"rm -r {Path(args.zarr) / existing[0]}/"
        )

    # Build full 3D array from flat predictions
    print("Building 3D P_unstable array...")
    t0 = time.time()
    p_full = np.full((n_loc, n_time, n_layer), np.nan, dtype=np.float32)
    p_full[indices[:, 0], indices[:, 1], indices[:, 2]] = p_unstable
    print(f"  built in {time.time()-t0:.1f}s  "
          f"memory={p_full.nbytes/1e9:.2f} GB")

    # Per-profile summary stats — argmax over layer axis, ignoring NaN
    print("Computing per-profile summary...")
    t0 = time.time()

    # For argmax, replace NaN with -1 so the index lands on a real-but-low
    # cell when a profile has any finite value, and on layer 0 (which we'll
    # mask out) when it doesn't.
    p_for_argmax = np.where(np.isnan(p_full), -1.0, p_full)
    max_layer = np.argmax(p_for_argmax, axis=-1).astype(np.int16)

    # Gather max values along the argmax axis
    loc_idx, time_idx = np.meshgrid(
        np.arange(n_loc), np.arange(n_time), indexing='ij'
    )
    max_val = p_full[loc_idx, time_idx, max_layer]
    has_any = np.isfinite(max_val)
    # Mask invalid profiles
    max_val = np.where(has_any, max_val, np.nan).astype(np.float32)
    max_layer_out = np.where(has_any, max_layer, -1).astype(np.int16)

    # Pull height + grain type at the max layer (heights in cm per earlier
    # check, regardless of CF attrs claim).
    print("  fetching height and grain_type at max-P_unstable layer...")
    height = ds.height.values
    grain_type = ds.grain_type.values
    max_height_cm = height[loc_idx, time_idx, max_layer]

    # HS = max valid height per (loc, time)
    h_for_max = np.where(np.isnan(height), -np.inf, height)
    hs_cm = h_for_max.max(axis=-1)
    depth_from_surface_cm = hs_cm - max_height_cm

    max_grain_type = grain_type[loc_idx, time_idx, max_layer]

    # Mask invalid
    max_height_cm = np.where(has_any, max_height_cm,
                             np.nan).astype(np.float32)
    depth_from_surface_cm = np.where(
        has_any, depth_from_surface_cm, np.nan
    ).astype(np.float32)
    max_grain_type = np.where(has_any, max_grain_type,
                              np.nan).astype(np.float32)
    print(f"  summary built in {time.time()-t0:.1f}s")

    # Build DataArrays matching the dim order of an existing 2D-ish variable
    ref_var = next(v for v in ds.data_vars if ds[v].dims == ('location', 'time'))
    ref_chunks = {d: c[0] for d, c in zip(ds[ref_var].dims, ds[ref_var].chunks)}
    print(f"Using chunking {ref_chunks} from reference 2D var '{ref_var}'")

    coords_2d = {'location': ds.coords['location'].values,
                 'time': ds.coords['time'].values}

    def make_da(arr, name, units, long_name, dtype=None):
        if dtype is not None:
            arr = arr.astype(dtype)
        da = xr.DataArray(
            arr, dims=('location', 'time'), coords=coords_2d, name=name,
            attrs={'units': units, 'long_name': long_name,
                   'source': 'Mayer et al. 2022 RF P_unstable model'},
        )
        return da.chunk(ref_chunks)

    summaries = xr.Dataset({
        'p_unstable_max': make_da(
            max_val, 'p_unstable_max', '1',
            'max P_unstable across profile'),
        'p_unstable_max_layer': make_da(
            max_layer_out, 'p_unstable_max_layer', '1',
            'layer index of max P_unstable', dtype=np.int16),
        'p_unstable_max_depth_cm': make_da(
            depth_from_surface_cm, 'p_unstable_max_depth_cm', 'cm',
            'depth below snow surface of max-P_unstable layer'),
        'p_unstable_max_height_cm': make_da(
            max_height_cm, 'p_unstable_max_height_cm', 'cm',
            'height above ground of max-P_unstable layer'),
        'p_unstable_max_grain_type': make_da(
            max_grain_type, 'p_unstable_max_grain_type', '1',
            'SNOWPACK grain type code at max-P_unstable layer'),
    })

    print("\nWriting summary variables...")
    t0 = time.time()
    if args.overwrite:
        # Drop existing copies first so to_zarr(mode='a') works cleanly
        for v in target_vars:
            varpath = Path(args.zarr) / v
            if varpath.exists():
                import shutil
                shutil.rmtree(varpath)
                print(f"  removed existing {v}")
    summaries.to_zarr(args.zarr, mode='a')
    print(f"  summaries written in {time.time()-t0:.1f}s")

    if args.write_3d:
        print("\nWriting full 3D P_unstable...")
        t0 = time.time()
        ref3d = next(v for v in ds.data_vars
                     if ds[v].dims == ('location', 'time', 'layer'))
        ref3d_chunks = {d: c[0] for d, c
                        in zip(ds[ref3d].dims, ds[ref3d].chunks)}
        print(f"  using chunking {ref3d_chunks} from '{ref3d}'")
        da3d = xr.DataArray(
            p_full,
            dims=('location', 'time', 'layer'),
            coords={'location': ds.coords['location'].values,
                    'time': ds.coords['time'].values},
            name='p_unstable',
            attrs={'units': '1',
                   'long_name': 'probability of instability per layer',
                   'source': 'Mayer et al. 2022 RF model'},
        ).chunk(ref3d_chunks)
        da3d.to_dataset().to_zarr(args.zarr, mode='a')
        print(f"  3D written in {time.time()-t0:.1f}s")

    ds.close()
    print("\nDone. Verify with:")
    print(f"  python -c \"import xarray as xr; "
          f"ds=xr.open_zarr('{args.zarr}'); "
          f"print(ds.p_unstable_max.attrs); "
          f"print('max-of-max:', float(ds.p_unstable_max.max()))\"")


if __name__ == '__main__':
    main()


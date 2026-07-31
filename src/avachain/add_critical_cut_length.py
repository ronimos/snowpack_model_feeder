"""
Side-loader for SNOWPACK var 0606 (critical cut length, rc) into
slope_snowpack.zarr.

xsnow drops 0606 during .pro -> xarray conversion. This script parses it
directly from the .pro files and appends it as a new data variable to the
existing Zarr without rebuilding.

Usage:
  python add_critical_cut_length.py --dry-run        # inspect only
  python add_critical_cut_length.py                  # parse and write
  python add_critical_cut_length.py --var-code 0607 --var-name rta  # other vars
"""
import argparse
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import xarray as xr


MAX_LAYERS = 338  # matches build_zarr_chunked.py


def parse_var(pro_path, var_code='0606', max_layers=MAX_LAYERS):
    """Extract one layered variable per timestep from a .pro file.

    Returns
    -------
    station_name : str or None  (StationName= from header; matches xsnow's
                                 location coord)
    times        : list[datetime]
    arr          : (n_times, max_layers) float32, NaN-padded
    """
    station_name = None
    times = []
    rows = []
    in_data = False
    cur_time = None
    cur_row = None
    prefix_500 = '0500,'
    prefix_var = f'{var_code},'

    with open(pro_path, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            if not in_data:
                if line.startswith('StationName='):
                    station_name = line.split('=', 1)[1].strip()
                elif line.strip() == '[DATA]':
                    in_data = True
                continue

            if line.startswith(prefix_500):
                # Close out previous timestep
                if cur_time is not None:
                    times.append(cur_time)
                    rows.append(cur_row)
                try:
                    cur_time = datetime.strptime(line[5:].strip(),
                                                 '%d.%m.%Y %H:%M:%S')
                except ValueError:
                    cur_time = None
                cur_row = np.full(max_layers, np.nan, dtype=np.float32)
            elif line.startswith(prefix_var) and cur_row is not None:
                parts = line.split(',')
                try:
                    n = int(parts[1])
                    if n <= 0:
                        continue
                    k = min(n, max_layers)
                    cur_row[:k] = np.fromiter(
                        (float(x) for x in parts[2:2 + k]),
                        dtype=np.float32, count=k,
                    )
                except (ValueError, IndexError):
                    pass

    # Flush the final timestep
    if cur_time is not None:
        times.append(cur_time)
        rows.append(cur_row)

    if not rows:
        return station_name, [], np.empty((0, max_layers), dtype=np.float32)
    return station_name, times, np.array(rows, dtype=np.float32)


def loc_name_from_path(pro_path):
    """Fallback only — extracts cluster_NNNN from the start of the filename."""
    return pro_path.stem.split('_cluster_')[0]


# Module-level for multiprocessing pickling
_VAR_CODE = '0606'


def _worker(pro_path):
    station, times, arr = parse_var(pro_path, var_code=_VAR_CODE)
    if station is None:
        station = loc_name_from_path(pro_path)
    return station, times, arr


def _init_worker(var_code):
    global _VAR_CODE
    _VAR_CODE = var_code


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pro-dir',
                   default='/data/snowpack/little_prof/output')
    p.add_argument('--zarr',
                   default='/data/snowpack/little_prof/output/'
                           'slope_snowpack.zarr')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--var-code', default='0606',
                   help='SNOWPACK variable code to extract')
    p.add_argument('--var-name', default='critical_cut_length',
                   help='Name to write into Zarr')
    p.add_argument('--units', default='m')
    p.add_argument('--long-name', default='critical cut length (Gaume/Reuter)')
    p.add_argument('--dry-run', action='store_true',
                   help='Parse and report, but do not write Zarr')
    p.add_argument('--sample-only', type=int, default=0,
                   help='If >0, only process this many .pro files (for testing)')
    args = p.parse_args()

    pro_dir = Path(args.pro_dir)
    zarr_path = Path(args.zarr)

    # Inspect existing Zarr
    ds = xr.open_zarr(str(zarr_path))
    if 'location' not in ds.coords:
        raise RuntimeError("Existing Zarr has no 'location' coordinate.")
    locations = list(ds.coords['location'].values.astype(str))
    times = ds.coords['time'].values
    n_loc, n_time = len(locations), len(times)
    n_layer = ds.sizes['layer']
    print(f"Existing Zarr: {n_loc} locations  {n_time} times  {n_layer} layers")
    print(f"  sample vars: {sorted(list(ds.data_vars))[:5]}")

    if args.var_name in ds.data_vars:
        print(f"ABORT: '{args.var_name}' already exists in Zarr. "
              f"Use a different --var-name or delete it first with:")
        print(f"  rm -r {zarr_path}/{args.var_name}")
        return

    # Map .pro files to Zarr locations using StationName from each .pro header
    # (read inside the worker). The filename pattern cluster_X_cluster_Y is
    # not the same as the Zarr's location coord, so we can't pre-filter here.
    pro_files = sorted(pro_dir.glob('cluster_*.pro'))
    print(f"Found {len(pro_files)} .pro files")

    ordered = pro_files
    if args.sample_only:
        ordered = ordered[:args.sample_only]
        print(f"--sample-only: processing first {len(ordered)} files")

    print(f"Parsing {len(ordered)} .pro files for var {args.var_code} "
          f"with {args.workers} workers...")

    # Time -> index lookup, normalized to ns
    time_to_idx = {np.datetime64(t, 'ns'): i for i, t in enumerate(times)}

    arr = np.full((n_loc, n_time, MAX_LAYERS), np.nan, dtype=np.float32)
    matched_t = 0
    unmatched_t = 0
    matched_locs = 0
    unmatched_stations = []

    t0 = time.time()
    done = 0
    loc_index = {loc: i for i, loc in enumerate(locations)}

    with Pool(args.workers,
              initializer=_init_worker, initargs=(args.var_code,)) as pool:
        for loc, file_times, file_arr in pool.imap_unordered(_worker, ordered):
            li = loc_index.get(loc)
            if li is None:
                unmatched_stations.append(loc)
                done += 1
                continue
            matched_locs += 1
            for j, t in enumerate(file_times):
                ti = time_to_idx.get(np.datetime64(t, 'ns'))
                if ti is not None:
                    arr[li, ti, :] = file_arr[j]
                    matched_t += 1
                else:
                    unmatched_t += 1
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(ordered)}  "
                      f"({time.time()-t0:.0f}s elapsed)")

    print(f"Parsed in {time.time()-t0:.0f}s. "
          f"Matched {matched_locs}/{len(ordered)} stations to Zarr locations.")
    if unmatched_stations:
        print(f"  Unmatched station names (first 5): {unmatched_stations[:5]}")
    print(f"Matched {matched_t} timesteps, unmatched {unmatched_t}.")
    finite_lt = np.isfinite(arr).any(axis=-1)  # (loc, time) any finite layer
    print(f"Coverage: {finite_lt.mean()*100:.1f}% of (location, time) cells "
          f"have at least one finite {args.var_name} value.")
    finite_vals = arr[np.isfinite(arr)]
    if finite_vals.size:
        print(f"Value range: min={finite_vals.min():.3g}  "
              f"median={np.median(finite_vals):.3g}  "
              f"max={finite_vals.max():.3g}")

    if args.dry_run:
        print("Dry run — not writing Zarr.")
        ds.close()
        return

    # Match dim order to reference variable for clean append
    ref_var = next(iter(ds.data_vars))
    ref_dims = ds[ref_var].dims
    print(f"Reference variable '{ref_var}' has dims {ref_dims}, "
          f"chunks {ds[ref_var].chunks}")

    coords = {'location': locations, 'time': times}
    if 'layer' in ds.coords:
        coords['layer'] = ds.coords['layer']

    da = xr.DataArray(
        arr,
        dims=('location', 'time', 'layer'),
        coords=coords,
        name=args.var_name,
        attrs={
            'long_name': args.long_name,
            'units': args.units,
            'snowpack_var': args.var_code,
        },
    ).transpose(*ref_dims)

    # Match chunk sizes of reference variable
    ref_chunks = {d: c[0] for d, c in zip(ref_dims, ds[ref_var].chunks)}
    da = da.chunk(ref_chunks)
    print(f"Chunking new var as {ref_chunks}")

    print(f"Writing '{args.var_name}' to {zarr_path} ...")
    da.to_dataset().to_zarr(str(zarr_path), mode='a')
    ds.close()
    print("Done.")


if __name__ == '__main__':
    main()
    
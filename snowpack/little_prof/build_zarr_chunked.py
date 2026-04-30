"""
Build Zarr cache from .pro files in batches.
Reads BATCH_SIZE files eagerly (no Dask), appends to Zarr.
Resumable: skips batches already written.

Usage:
  python build_zarr_chunked.py
  python build_zarr_chunked.py --batch-size 200 --workers 16
"""
import argparse
import time
from pathlib import Path

import numpy as np
import xsnow
import xarray as xr


BATCH_SIZE = 100
WORKERS    = 31


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pro-dir',
                        default='/home/ron/snowpack_model_feeder/snowpack/little_prof/output')
    parser.add_argument('--zarr-out',
                        default='/home/ron/snowpack_model_feeder/snowpack/little_prof/output/'
                                'slope_snowpack.zarr')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--workers',    type=int, default=WORKERS)
    args = parser.parse_args()

    pro_dir  = Path(args.pro_dir)
    zarr_out = Path(args.zarr_out)

    pro_files = sorted(pro_dir.glob("cluster_*_cluster_*.pro"))
    n_total   = len(pro_files)
    MAX_LAYERS = 338   # global max across all .pro files
    n_batches = int(np.ceil(n_total / args.batch_size))
    print(f"Files: {n_total}  Batch size: {args.batch_size}  "
          f"Batches: {n_batches}  Workers: {args.workers}")

    # Track which locations are already written
    written_locs = set()
    if zarr_out.exists():
        try:
            existing = xr.open_zarr(str(zarr_out))
            written_locs = set(existing.coords['location'].values)
            print(f"Resuming — {len(written_locs)} locations already written")
            existing.close()
        except Exception as e:
            print(f"Could not read existing Zarr ({e}), starting fresh")

    t_total = time.time()

    for batch_idx in range(n_batches):
        batch_files = pro_files[batch_idx * args.batch_size:
                                (batch_idx + 1) * args.batch_size]

        # Skip if all locations in this batch are already written
        batch_names = [f.stem.replace('_cluster_', '_') for f in batch_files]
        if written_locs and all(n in written_locs for n in batch_names):
            print(f"  Batch {batch_idx+1}/{n_batches}: SKIP (already written)")
            continue

        print(f"  Batch {batch_idx+1}/{n_batches}: "
              f"{batch_files[0].name} ... {batch_files[-1].name}")

        # Make a temp dir with symlinks for this batch
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            for f in batch_files:
                os.symlink(str(f), os.path.join(tmp, f.name))

            t0 = time.time()
            try:
                ds = xsnow.read(tmp, lazy=False,
                                n_cpus_use=args.workers)
            except Exception as e:
                print(f"    ERROR reading batch: {e}")
                continue
            t_read = time.time() - t0

        # Squeeze singleton dims and pad layers to global max
        ds = ds.squeeze(['slope', 'realization'], drop=True)
        current_layers = ds.dims['layer']
        if current_layers < MAX_LAYERS:
            ds = ds.pad(layer=(0, MAX_LAYERS - current_layers),
                        mode='constant', constant_values=np.nan)
        elif current_layers > MAX_LAYERS:
            ds = ds.isel(layer=slice(0, MAX_LAYERS))

        # Append to Zarr
        t1 = time.time()
        mode = 'w' if not zarr_out.exists() else 'a'
        try:
            if mode == 'a':
                ds.to_zarr(str(zarr_out), mode='a', append_dim='location')
            else:
                ds.to_zarr(str(zarr_out), mode='w')
        except Exception as e:
            print(f"    ERROR writing zarr: {e}")
            continue
        finally:
            del ds  # free memory before next batch

        t_write = time.time() - t1
        elapsed = time.time() - t_total
        done    = (batch_idx + 1) / n_batches
        eta     = elapsed / done * (1 - done) / 60

        print(f"    read={t_read:.1f}s  write={t_write:.1f}s  "
              f"ETA={eta:.0f}min")

    print(f"\nDone in {(time.time()-t_total)/60:.1f} min -> {zarr_out}")


if __name__ == '__main__':
    main()
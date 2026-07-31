"""
build_zarr_chunked.py — CLI wrapper around snowpack_io.build_zarr_cache.

All logic lives in snowpack_io. This file is a thin entry point so the
existing nohup command still works unchanged.

Usage:
    python build_zarr_chunked.py
    python build_zarr_chunked.py --batch-size 200 --workers 16
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from snowpack_io import build_zarr_cache


def main():
    parser = argparse.ArgumentParser(
        description="Build Zarr cache from .pro files")
    parser.add_argument('--pro-dir',
                        default='/home/ron/snowpack/little_prof/output')
    parser.add_argument('--zarr-out',
                        default='/home/ron/snowpack/little_prof/output/'
                                'slope_snowpack.zarr')
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--workers',    type=int, default=31)
    parser.add_argument('--max-layers', type=int, default=338)
    args = parser.parse_args()

    build_zarr_cache(
        pro_dir    = Path(args.pro_dir),
        zarr_out   = Path(args.zarr_out),
        batch_size = args.batch_size,
        n_cpus     = args.workers,
        max_layers = args.max_layers,
    )


if __name__ == '__main__':
    main()


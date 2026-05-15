"""
validate_punstable_spatial.py

Spatial validation of Mayer P_unstable on the Jan 18 2026 D2 event.

Proper counterfactual: release-area clusters vs adjacent skied-but-not-
triggered clusters. Replaces the earlier in/out-of-release comparison,
which diluted the "out" set with irrelevant terrain.

Outputs:
  1. Distribution comparison (printed): release vs skied-non-release.
  2. Hit-rate / lift metrics at Mayer thresholds.
  3. PNG map with both polygons overlaid on P_unstable_max raster.
  4. Optional time series at representative clusters (--time-series).

Usage:
  python validate_punstable_spatial.py \\
      --cluster-raster outputs/analysis/cluster_map.tif \\
      --release-area data/boundaries/avalanche_release_area.geojson \\
      --skied-non-release data/boundaries/skies-non-release.geojson \\
      --time 2026-01-18T18:00:00 \\
      --out outputs/plots/fig_punstable_jan18.png \\
      --time-series
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.features
import xarray as xr
from shapely.geometry import mapping

P_POOR = 0.77
P_VERY_POOR = 0.92


def load_polygon(path):
    """Return unioned shapely geometry + CRS for KML/GeoJSON/shapefile."""
    import geopandas as gpd
    path = Path(path)
    if path.suffix.lower() == '.kml':
        try:
            gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
        except Exception:
            pass
        gdf = gpd.read_file(path, driver='KML')
    else:
        gdf = gpd.read_file(path)
    return gdf.unary_union, gdf.crs


def reproject_if_needed(geom, src_crs, dst_crs):
    if src_crs == dst_crs:
        return geom
    from pyproj import Transformer
    from shapely.ops import transform as shp_transform
    tf = Transformer.from_crs(src_crs, dst_crs, always_xy=True).transform
    return shp_transform(tf, geom)


def cluster_membership(cluster_raster, polygon_mask, unique_ids):
    """Return set of cluster IDs where >=50% of pixels are inside polygon."""
    keep = set()
    for cid in unique_ids:
        if cid == 0:
            continue
        cmask = cluster_raster == cid
        ntotal = cmask.sum()
        if ntotal == 0:
            continue
        if (cmask & polygon_mask).sum() / ntotal >= 0.5:
            keep.add(int(cid))
    return keep


def report_distribution(label, vals):
    print(f"\n{label}  (n={len(vals)})")
    if len(vals) == 0:
        print("  empty")
        return
    for q_label, q in [('min', 0), ('p10', 10), ('p25', 25),
                       ('median', 50), ('p75', 75), ('p90', 90),
                       ('p95', 95), ('max', 100)]:
        print(f"  {q_label:>7s}  {np.percentile(vals, q):.4f}")


def draw_polygon(ax, geom, color, label):
    if geom is None:
        return
    geoms = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
    for i, poly in enumerate(geoms):
        x, y = poly.exterior.xy
        ax.plot(x, y, color=color, linewidth=2,
                label=label if i == 0 else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--zarr',
        default='/home/ron/snowpack_model_feeder/snowpack/little_prof/output/'
                'slope_snowpack.zarr')
    p.add_argument('--cluster-raster', required=True)
    p.add_argument('--release-area', required=True,
                   help='Polygon of the avalanche release area')
    p.add_argument('--skied-non-release', default=None,
                   help='Polygon of adjacent skied-but-not-triggered terrain')
    p.add_argument('--time', default='2026-01-18T18:00:00')
    p.add_argument('--out', default='fig_punstable_release_vs_skied.png')
    p.add_argument('--time-series', action='store_true',
                   help='Also emit a P_unstable_max time-series figure')
    p.add_argument('--ts-out', default=None,
                   help='Path for time series PNG (defaults next to --out)')
    p.add_argument('--cluster-id-from', default='cluster_')
    args = p.parse_args()

    # ----- Zarr snapshot -----
    print(f"Loading Zarr snapshot at {args.time}")
    ds = xr.open_zarr(args.zarr)
    snap = ds[['p_unstable_max', 'p_unstable_max_depth_cm',
               'p_unstable_max_grain_type']].sel(
        time=args.time, method='nearest').load()
    print(f"  selected {snap.time.values}")
    loc_strs = ds.coords['location'].values.astype(str)
    cluster_ids = np.array([int(s.replace(args.cluster_id_from, ''))
                            for s in loc_strs])
    p_max = snap.p_unstable_max.values
    cid_to_pmax = dict(zip(cluster_ids, p_max))

    # ----- Cluster raster -----
    print(f"Loading cluster raster: {args.cluster_raster}")
    with rasterio.open(args.cluster_raster) as src:
        cluster_raster = src.read(1)
        raster_crs = src.crs
        raster_transform = src.transform
        raster_shape = src.shape
        raster_bounds = src.bounds
    unique_ids = np.unique(cluster_raster)
    print(f"  shape={raster_shape}  crs={raster_crs}  "
          f"{len(unique_ids)} unique IDs")

    # Build P_unstable raster for the map
    max_id = int(cluster_raster.max())
    lookup = np.full(max_id + 2, np.nan, dtype=np.float32)
    for cid, val in cid_to_pmax.items():
        if 0 <= cid <= max_id:
            lookup[cid] = val
    pmax_raster = lookup[np.clip(cluster_raster, 0, max_id + 1)]
    pmax_raster = np.where(cluster_raster > 0, pmax_raster, np.nan)

    # ----- Polygons -----
    print(f"\nLoading release area: {args.release_area}")
    release_geom, release_crs = load_polygon(args.release_area)
    release_geom = reproject_if_needed(release_geom, release_crs, raster_crs)
    release_mask = rasterio.features.rasterize(
        [(mapping(release_geom), 1)], out_shape=raster_shape,
        transform=raster_transform, fill=0, dtype=np.uint8).astype(bool)
    release_clusters = cluster_membership(cluster_raster, release_mask,
                                          unique_ids)
    print(f"  area={release_geom.area:.0f} m²  "
          f"{len(release_clusters)} clusters")

    skied_geom = None
    skied_clusters = set()
    if args.skied_non_release:
        print(f"Loading skied-non-release: {args.skied_non_release}")
        skied_geom, skied_crs = load_polygon(args.skied_non_release)
        skied_geom = reproject_if_needed(skied_geom, skied_crs, raster_crs)
        skied_mask = rasterio.features.rasterize(
            [(mapping(skied_geom), 1)], out_shape=raster_shape,
            transform=raster_transform, fill=0, dtype=np.uint8).astype(bool)
        skied_clusters = cluster_membership(cluster_raster, skied_mask,
                                            unique_ids)
        print(f"  area={skied_geom.area:.0f} m²  "
              f"{len(skied_clusters)} clusters")
        overlap = release_clusters & skied_clusters
        if overlap:
            print(f"  WARN: {len(overlap)} clusters in BOTH polygons; "
                  f"removing from skied set")
            skied_clusters = skied_clusters - overlap

    # ----- Distributions -----
    rel_vals = np.array([cid_to_pmax.get(c, np.nan)
                         for c in release_clusters])
    rel_vals = rel_vals[np.isfinite(rel_vals)]
    report_distribution("release-area P_unstable_max", rel_vals)

    if skied_clusters:
        skied_vals = np.array([cid_to_pmax.get(c, np.nan)
                               for c in skied_clusters])
        skied_vals = skied_vals[np.isfinite(skied_vals)]
        report_distribution("skied-non-release P_unstable_max", skied_vals)

        try:
            from scipy.stats import mannwhitneyu
            u, pval = mannwhitneyu(rel_vals, skied_vals,
                                   alternative='greater')
            d_med = np.median(rel_vals) - np.median(skied_vals)
            print(f"\nMann-Whitney U (release > skied): "
                  f"U={u:.0f}  p={pval:.3g}")
            print(f"  median difference: {d_med:+.4f}")
            print("  Note: significance depends on sample size; the median "
                  "difference is the operationally meaningful number.")
        except ImportError:
            print("\n(scipy unavailable; skipping Mann-Whitney)")

        print("\nHit rates at Mayer thresholds:")
        for thr in [0.5, P_POOR, P_VERY_POOR]:
            rel_hit = (rel_vals >= thr).sum()
            skied_hit = (skied_vals >= thr).sum()
            rel_rate = rel_hit / len(rel_vals) * 100 if len(rel_vals) else 0
            skied_rate = skied_hit / len(skied_vals) * 100 if len(skied_vals) else 0
            print(f"  P>={thr:.2f}: release {rel_hit}/{len(rel_vals)} "
                  f"({rel_rate:.1f}%)  vs  skied {skied_hit}/{len(skied_vals)} "
                  f"({skied_rate:.1f}%)")

    # ----- Map -----
    print(f"\nRendering map to {args.out}")
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    extent = (raster_bounds.left, raster_bounds.right,
              raster_bounds.bottom, raster_bounds.top)

    ax = axes[0]
    im = ax.imshow(pmax_raster, extent=extent, origin='upper',
                   cmap='RdYlGn_r', vmin=0, vmax=1)
    draw_polygon(ax, release_geom, 'black', 'Release area')
    draw_polygon(ax, skied_geom, 'blue', 'Skied, no release')
    ax.set_title(f'P_unstable_max  @  {snap.time.values}')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='P_unstable')
    ax.legend(loc='best')

    depth_vals = snap.p_unstable_max_depth_cm.values
    depth_lookup = np.full(max_id + 2, np.nan, dtype=np.float32)
    for cid, val in zip(cluster_ids, depth_vals):
        if 0 <= cid <= max_id:
            depth_lookup[cid] = val
    depth_raster = depth_lookup[np.clip(cluster_raster, 0, max_id + 1)]
    depth_raster = np.where(cluster_raster > 0, depth_raster, np.nan)

    ax = axes[1]
    im = ax.imshow(depth_raster, extent=extent, origin='upper',
                   cmap='viridis_r')
    draw_polygon(ax, release_geom, 'red', 'Release area')
    draw_polygon(ax, skied_geom, 'cyan', 'Skied, no release')
    ax.set_title('Depth below surface of max-P_unstable layer (cm)')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='depth (cm)')
    ax.legend(loc='best')

    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  saved {args.out}")

    # ----- Optional time series -----
    if args.time_series:
        ts_out = args.ts_out or str(Path(args.out).with_name(
            Path(args.out).stem + '_timeseries.png'))
        print(f"\nRendering time series to {ts_out}")
        if not skied_clusters:
            print("  skipping (need --skied-non-release)")
        else:
            rel_pairs = sorted(
                [(c, cid_to_pmax[c]) for c in release_clusters
                 if c in cid_to_pmax and np.isfinite(cid_to_pmax[c])],
                key=lambda x: -x[1])
            skied_pairs = sorted(
                [(c, cid_to_pmax[c]) for c in skied_clusters
                 if c in cid_to_pmax and np.isfinite(cid_to_pmax[c])],
                key=lambda x: -x[1])
            picked = [
                (rel_pairs[0][0], 'release-area (max)', 'C3'),
                (skied_pairs[0][0], 'skied-non-release (max)', 'C0'),
                (skied_pairs[len(skied_pairs)//2][0],
                 'skied-non-release (median)', 'C2'),
            ]
            print(f"  picked cluster IDs: {[c for c, _, _ in picked]}")

            fig, ax = plt.subplots(figsize=(12, 5))
            for cid, label, color in picked:
                loc_name = f"{args.cluster_id_from}{cid:04d}"
                if loc_name not in loc_strs:
                    print(f"  WARN: {loc_name} not in Zarr, skipping")
                    continue
                series = ds.p_unstable_max.sel(location=loc_name).values
                times = ds.time.values
                ax.plot(times, series, color=color, label=label, linewidth=1.5)

            event_time = np.datetime64(args.time)
            ax.axvline(event_time, color='black', linestyle='--',
                       linewidth=1, label='Jan 18 trigger')
            ax.axhline(P_POOR, color='gray', linestyle=':',
                       linewidth=1, label=f'Mayer poor threshold ({P_POOR})')
            ax.set_xlabel('Date')
            ax.set_ylabel('P_unstable_max')
            ax.set_title('P_unstable_max evolution: release vs skied terrain')
            ax.legend(loc='best')
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(ts_out, dpi=120, bbox_inches='tight')
            plt.close()
            print(f"  saved {ts_out}")

    print("\nDone.")


if __name__ == '__main__':
    main()
    
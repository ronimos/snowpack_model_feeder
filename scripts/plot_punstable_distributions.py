"""
plot_punstable_distributions.py

Focused distribution comparison: release-area vs skied-non-release clusters.
Standalone — reuses the polygon loading logic from validate_punstable_spatial
but produces only the diagnostic plot.

Two panels:
  Left:  violin + jittered strip showing the full distribution shape.
  Right: empirical CDF (zoomed to upper region) showing where the curves
         actually diverge.

Usage:
  python plot_punstable_distributions.py \\
      --cluster-raster outputs/analysis/cluster_map.tif \\
      --release-area data/boundaries/avalanche_release_area.geojson \\
      --skied-non-release data/boundaries/skies-non-release.geojson \\
      --time 2026-01-18T18:00:00 \\
      --out outputs/plots/fig_punstable_distributions.png
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


def load_polygon(path):
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


def values_for_polygon(cluster_raster, geom, raster_shape, raster_transform,
                       unique_ids, cid_to_val):
    mask = rasterio.features.rasterize(
        [(mapping(geom), 1)], out_shape=raster_shape,
        transform=raster_transform, fill=0, dtype=np.uint8).astype(bool)
    clusters = cluster_membership(cluster_raster, mask, unique_ids)
    vals = np.array([cid_to_val.get(c, np.nan) for c in clusters])
    return clusters, vals[np.isfinite(vals)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--zarr',
        default='/home/ron/snowpack_model_feeder/snowpack/little_prof/output/'
                'slope_snowpack.zarr')
    p.add_argument('--cluster-raster', required=True)
    p.add_argument('--release-area', required=True)
    p.add_argument('--skied-non-release', required=True)
    p.add_argument('--time', default='2026-01-18T18:00:00')
    p.add_argument('--out', default='fig_punstable_distributions.png')
    p.add_argument('--cluster-id-from', default='cluster_')
    args = p.parse_args()

    ds = xr.open_zarr(args.zarr)
    snap = ds[['p_unstable_max']].sel(time=args.time, method='nearest').load()
    loc_strs = ds.coords['location'].values.astype(str)
    cluster_ids = np.array([int(s.replace(args.cluster_id_from, ''))
                            for s in loc_strs])
    p_max = snap.p_unstable_max.values
    cid_to_pmax = dict(zip(cluster_ids, p_max))

    with rasterio.open(args.cluster_raster) as src:
        cluster_raster = src.read(1)
        raster_crs = src.crs
        raster_transform = src.transform
        raster_shape = src.shape
    unique_ids = np.unique(cluster_raster)

    rel_geom, rel_crs = load_polygon(args.release_area)
    rel_geom = reproject_if_needed(rel_geom, rel_crs, raster_crs)
    rel_clusters, rel_vals = values_for_polygon(
        cluster_raster, rel_geom, raster_shape, raster_transform,
        unique_ids, cid_to_pmax)

    skied_geom, skied_crs = load_polygon(args.skied_non_release)
    skied_geom = reproject_if_needed(skied_geom, skied_crs, raster_crs)
    skied_clusters, skied_vals = values_for_polygon(
        cluster_raster, skied_geom, raster_shape, raster_transform,
        unique_ids, cid_to_pmax)

    # Remove any cluster appearing in both polygons from the skied set
    overlap = rel_clusters & skied_clusters
    if overlap:
        keep = np.array([c not in overlap for c in skied_clusters])
        skied_vals_filtered = []
        for c in skied_clusters:
            if c not in overlap:
                v = cid_to_pmax.get(c, np.nan)
                if np.isfinite(v):
                    skied_vals_filtered.append(v)
        skied_vals = np.array(skied_vals_filtered)

    print(f"Release: n={len(rel_vals)}  Skied: n={len(skied_vals)}")
    print(f"Median: release={np.median(rel_vals):.4f}  "
          f"skied={np.median(skied_vals):.4f}  "
          f"Δ={np.median(rel_vals)-np.median(skied_vals):+.4f}")
    print(f"p95:    release={np.percentile(rel_vals,95):.4f}  "
          f"skied={np.percentile(skied_vals,95):.4f}  "
          f"Δ={np.percentile(rel_vals,95)-np.percentile(skied_vals,95):+.4f}")

    # ----- Figure -----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    rel_color = '#c1272d'
    skied_color = '#0072b2'

    # --- Left: violin with strip overlay ---
    ax = axes[0]
    parts = ax.violinplot([rel_vals, skied_vals], positions=[0, 1],
                          widths=0.7, showmeans=False, showmedians=False,
                          showextrema=False)
    for i, body in enumerate(parts['bodies']):
        body.set_facecolor(rel_color if i == 0 else skied_color)
        body.set_alpha(0.35)
        body.set_edgecolor('black')

    # Strip overlay (jittered scatter)
    rng = np.random.default_rng(42)
    for i, (vals, color) in enumerate([(rel_vals, rel_color),
                                       (skied_vals, skied_color)]):
        x = i + rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(x, vals, s=10, color=color, alpha=0.4,
                   edgecolor='none', zorder=2)

    # Median + IQR
    for i, vals in enumerate([rel_vals, skied_vals]):
        q25, q50, q75 = np.percentile(vals, [25, 50, 75])
        ax.plot([i - 0.25, i + 0.25], [q50, q50], color='black', linewidth=2.5,
                zorder=3)
        ax.plot([i, i], [q25, q75], color='black', linewidth=1.5, zorder=3)

    ax.axhline(P_POOR, color='gray', linestyle=':', linewidth=1,
               label=f'Mayer poor threshold ({P_POOR})')
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'Release area\n(n={len(rel_vals)})',
                        f'Skied, no release\n(n={len(skied_vals)})'])
    ax.set_ylabel('P_unstable_max')
    ax.set_title('Distribution by terrain class')
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3, axis='y')
    ax.legend(loc='upper right')

    # --- Right: ECDF (1 - CDF, log-y for tail focus) ---
    ax = axes[1]
    rel_sorted = np.sort(rel_vals)
    skied_sorted = np.sort(skied_vals)
    rel_ccdf = 1.0 - np.arange(1, len(rel_sorted) + 1) / len(rel_sorted)
    skied_ccdf = 1.0 - np.arange(1, len(skied_sorted) + 1) / len(skied_sorted)

    ax.step(rel_sorted, rel_ccdf, where='post', color=rel_color,
            linewidth=2, label=f'Release area  (n={len(rel_vals)})')
    ax.step(skied_sorted, skied_ccdf, where='post', color=skied_color,
            linewidth=2, label=f'Skied, no release  (n={len(skied_vals)})')

    ax.axvline(P_POOR, color='gray', linestyle=':', linewidth=1,
               label=f'Mayer poor threshold ({P_POOR})')

    ax.set_yscale('log')
    ax.set_xlim(0.2, 1.0)
    ax.set_ylim(1.0 / max(len(rel_vals), len(skied_vals)) / 2, 1.0)
    ax.set_xlabel('P_unstable_max')
    ax.set_ylabel('Fraction of clusters above threshold (log scale)')
    ax.set_title('Survival curve — where the tails diverge')
    ax.grid(alpha=0.3, which='both')
    ax.legend(loc='upper right')

    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=120, bbox_inches='tight')
    print(f"Saved {args.out}")


if __name__ == '__main__':
    main()


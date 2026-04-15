"""
Compare SNOWPACK indices between:
  1. Release zone  — clusters inside the observed Jan 18 release area
  2. Adjacent slope — clusters inside start zone KML but outside release zone
  3. Reference      — clusters inside survey domain but outside start zone

Produces time series plots of Sk38, SSI, Sn38, temperature gradient,
and HS for each group, with the Jan 18 event date marked.

Usage:
  python analyze_release_zone.py
  python analyze_release_zone.py --min-depth 20 --end-date 2026-02-01
"""

import argparse
import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rasterio
import rasterio.features
import rasterio.transform as rt
import xarray as xr
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import xsnow
from config import ProjectConfig

EVENT_DATE  = pd.Timestamp("2026-01-18")
MIN_DEPTH   = 30.0   # cm — bury depth for stability indices

GROUP_COLORS = {
    'release':  '#d32f2f',
    'adjacent': '#1976d2',
    'reference': '#555555',
}
GROUP_LABELS = {
    'release':  'Release zone',
    'adjacent': 'Adjacent slope (start zone)',
    'reference': 'Reference (outside start zone)',
}


# -----------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------

def geojson_to_mask(geojson_path, dem_shape, transform) -> np.ndarray:
    """Rasterize a GeoJSON polygon to a boolean mask (True = inside)."""
    import json
    from shapely.geometry import shape
    from shapely.ops import unary_union

    with open(str(geojson_path)) as f:
        gj = json.load(f)

    polys = [shape(feat['geometry']) for feat in gj['features']]
    merged = unary_union(polys)

    # GeoJSON is WGS84 lon/lat — reproject to DEM CRS (UTM zone 13N)
    transformer = Transformer.from_crs('EPSG:4326', 'EPSG:32613', always_xy=True)

    def reproject_poly(poly):
        from shapely.geometry import Polygon, MultiPolygon
        def reproject_ring(ring):
            return [transformer.transform(x, y) for x, y in ring.coords]
        if poly.geom_type == 'Polygon':
            exterior = reproject_ring(poly.exterior)
            interiors = [reproject_ring(r) for r in poly.interiors]
            return Polygon(exterior, interiors)
        elif poly.geom_type == 'MultiPolygon':
            return MultiPolygon([reproject_poly(p) for p in poly.geoms])
        return poly

    merged_utm = reproject_poly(merged)

    aff = transform if hasattr(transform, 'c') else \
          rt.Affine(*[transform[i] for i in range(6)])

    mask = rasterio.features.geometry_mask(
        [merged_utm.__geo_interface__],
        out_shape=dem_shape,
        transform=aff,
        invert=True)
    print(f"  Release zone mask: {mask.sum()} cells")
    return mask


def kml_to_mask(kml_path, dem_shape, transform) -> np.ndarray:
    """Reproject KML polygon to DEM CRS and rasterize."""
    from xml.etree import ElementTree as ET
    from shapely.geometry import Polygon

    tree = ET.parse(str(kml_path))
    coords_text = None
    for tag in ['.//coordinates',
                './/{http://www.opengis.net/kml/2.2}coordinates']:
        coords_text = tree.find(tag)
        if coords_text is not None:
            break
    if coords_text is None:
        return np.ones(dem_shape, dtype=bool)

    pts_lonlat = []
    for part in coords_text.text.strip().split():
        xyz = part.split(',')
        if len(xyz) >= 2:
            pts_lonlat.append((float(xyz[0]), float(xyz[1])))

    transformer = Transformer.from_crs('EPSG:4326', 'EPSG:32613', always_xy=True)
    pts_utm = [transformer.transform(lon, lat) for lon, lat in pts_lonlat]
    poly = Polygon(pts_utm)

    aff = transform if hasattr(transform, 'c') else \
          rt.Affine(*[transform[i] for i in range(6)])

    mask = rasterio.features.geometry_mask(
        [poly.__geo_interface__],
        out_shape=dem_shape,
        transform=aff,
        invert=True)
    print(f"  Start zone mask:  {mask.sum()} cells")
    return mask


# -----------------------------------------------------------------------
# Cluster group assignment
# -----------------------------------------------------------------------

def assign_cluster_groups(cluster_map, release_mask, start_zone_mask,
                           domain_mask) -> dict:
    """
    Returns dict mapping group name -> set of cluster IDs.
    """
    groups = {'release': set(), 'adjacent': set(), 'reference': set()}

    cluster_ids = np.unique(cluster_map[domain_mask])
    cluster_ids = cluster_ids[cluster_ids > 0]

    for cid in cluster_ids:
        cells = cluster_map == cid
        n = cells.sum()
        if n == 0:
            continue
        n_release   = (cells & release_mask).sum()
        n_start     = (cells & start_zone_mask).sum()

        # Assign to group with majority overlap
        if n_release / n >= 0.3:
            groups['release'].add(cid)
        elif n_start / n >= 0.3:
            groups['adjacent'].add(cid)
        else:
            groups['reference'].add(cid)

    for g, ids in groups.items():
        print(f"  {GROUP_LABELS[g]}: {len(ids)} clusters")
    return groups


# -----------------------------------------------------------------------
# Per-group time series extraction
# -----------------------------------------------------------------------

def extract_group_timeseries(ds, group_ids: set,
                              location_names,
                              min_depth_cm: float) -> pd.DataFrame:
    """
    Compute daily median of key indices for a group of cluster IDs.
    Returns DataFrame indexed by time.
    """
    if not group_ids:
        return pd.DataFrame()

    # Filter locations to this group
    loc_mask = np.array([
        int(str(loc).split('_')[-1]) in group_ids
        for loc in location_names
    ])
    if not loc_mask.any():
        return pd.DataFrame()

    ds_group = ds.isel(location=loc_mask)

    rows = []
    times = pd.DatetimeIndex(ds_group.coords['time'].values)
    # Sample every 6h (SNOWPACK outputs every 6h) then resample to daily
    sample_times = times[times.hour == 12]

    for ts in sample_times:
        try:
            ds_t = ds_group.sel(time=ts, method='nearest')
            z = ds_t['z'].values  # (location, layer)

            in_depth = (z <= -min_depth_cm) & (~np.isnan(z))

            def grp_min(var):
                v = ds_t[var].values
                v_masked = np.where(in_depth, v, np.nan)
                return np.nanmedian(np.nanmin(v_masked, axis=-1))

            def grp_mean(var):
                v = ds_t[var].values
                v_masked = np.where(in_depth, v, np.nan)
                return np.nanmedian(np.nanmean(v_masked, axis=-1))

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                rows.append({
                    'time':   ts,
                    'sk38':   grp_min('sk38'),
                    'ssi':    grp_min('ssi'),
                    'sn38':   grp_min('sn38'),
                    'tg':     float(np.nanmedian(
                                  np.abs(ds_t['temperature_gradient'].values))),
                    'atg':    grp_mean('accumulated_temperature_gradient'),
                    'hs':     float(np.nanmedian(ds_t['HS'].values)),
                    'sdr':    grp_min('stab_deformation_rate'),
                })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index('time').sort_index()


# -----------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------

def plot_comparison(group_series: dict, out_path: Path, min_depth: float):
    """Six-panel time series comparison plot."""
    variables = [
        ('sk38',  'Min Sk38 (buried >{:.0f}cm)'.format(min_depth),
         1.0,  True),
        ('ssi',   'Min SSI',              1.5,  True),
        ('sn38',  'Min Sn38',             1.0,  True),
        ('tg',    'Median |TG| (°C/m)',   10.0, False),
        ('hs',    'Median HS (m)',         None, False),
        ('atg',   'Median Accum TG',       None, False),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    axes = axes.ravel()

    for ax, (var, label, threshold, invert) in zip(axes, variables):
        for grp, df in group_series.items():
            if df.empty or var not in df.columns:
                continue
            ax.plot(df.index, df[var],
                    color=GROUP_COLORS[grp],
                    label=GROUP_LABELS[grp],
                    linewidth=1.4, alpha=0.9)

        if threshold is not None:
            ax.axhline(threshold, color='black', linewidth=0.8,
                       linestyle='--', alpha=0.5,
                       label=f'threshold={threshold}')

        ax.axvline(EVENT_DATE, color='darkred', linewidth=1.5,
                   linestyle='-', alpha=0.8)
        ax.text(EVENT_DATE, ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 1,
                ' Jan 18', color='darkred', fontsize=8, va='top')

        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.3)
        if invert:
            ax.invert_yaxis()

    axes[0].legend(fontsize=8, loc='upper left')
    fig.suptitle(
        f'SNOWPACK indices: release zone vs adjacent vs reference\n'
        f'Little Professor | Jan 18 2026 D2 event | '
        f'buried >{min_depth:.0f}cm',
        fontsize=11)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved: {out_path}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-dir', default='/home/ron/snowpack_model_feeder')
    parser.add_argument('--pro-dir',
                        default='/home/ron/snowpack/little_prof/output')
    parser.add_argument('--zarr-path',
                        default='/home/ron/snowpack/little_prof/output/'
                                'slope_snowpack.zarr')
    parser.add_argument('--release-geojson',
                        default=None,
                        help='Path to release area GeoJSON '
                             '(default: data/boundaries/avalanche_release_area.geojson)')
    parser.add_argument('--min-depth', type=float, default=MIN_DEPTH)
    parser.add_argument('--end-date',  default='2026-02-01')
    args = parser.parse_args()

    cfg = ProjectConfig(project_dir=Path(args.project_dir))

    geojson_path = Path(args.release_geojson) if args.release_geojson else \
                   cfg.project_dir / 'data' / 'boundaries' / \
                   'avalanche_release_area.geojson'
    if not geojson_path.exists():
        print(f"ERROR: release GeoJSON not found: {geojson_path}")
        sys.exit(1)

    # Load DEM
    with rasterio.open(str(cfg.resampled_dir / "dem_1m.tif")) as src:
        dem       = src.read(1).astype(np.float32)
        dem[dem == src.nodata] = np.nan
        transform = src.transform
        dem_shape = dem.shape

    domain_mask = ~np.isnan(dem)

    print("Building spatial masks...")
    release_mask    = geojson_to_mask(geojson_path, dem_shape, transform)
    start_zone_mask = kml_to_mask(cfg.start_zone_kml, dem_shape, transform)

    cluster_map = np.load(str(cfg.analysis_dir / "cluster_map.npy"))

    print("Assigning clusters to groups...")
    groups = assign_cluster_groups(
        cluster_map, release_mask, start_zone_mask, domain_mask)

    # Save group cluster IDs for later use
    import json
    group_ids_path = cfg.analysis_dir / "release_zone_groups.json"
    group_ids_path.write_text(json.dumps(
        {k: sorted(v) for k, v in groups.items()}, indent=2))
    print(f"Cluster group IDs saved: {group_ids_path}")

    # Load SNOWPACK dataset
    zarr_path = Path(args.zarr_path)
    print("Loading SNOWPACK dataset...")
    if zarr_path.exists():
        print(f"  From Zarr: {zarr_path}")
        ds = xsnow.xsnowDataset(xr.open_zarr(str(zarr_path)))
    else:
        print(f"  From .pro files: {args.pro_dir}")
        ds = xsnow.read(args.pro_dir)

    location_names = ds.coords['location'].values

    # Limit time range
    end_ts = pd.Timestamp(args.end_date)

    print("Extracting time series per group...")
    group_series = {}
    for grp, ids in groups.items():
        print(f"  {GROUP_LABELS[grp]}...")
        df = extract_group_timeseries(
            ds, ids, location_names, args.min_depth)
        if not df.empty:
            group_series[grp] = df.loc[:end_ts]

    # Plot
    out_path = cfg.plots_dir / "release_zone_snowpack_comparison.png"
    cfg.plots_dir.mkdir(parents=True, exist_ok=True)
    plot_comparison(group_series, out_path, args.min_depth)


if __name__ == '__main__':
    main()

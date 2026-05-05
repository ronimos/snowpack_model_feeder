"""
reinitialize_snowpack.py — Post-avalanche SNOWPACK reinitialization.

Detects the release area from UAS dHS (min-kernel method), identifies
which clusters are inside the release zone, and modifies their .sno
restart files by removing layers from the top down to the slab/WL
interface depth.

Two-pass SNOWPACK workflow:
  Pass 1:  Run SNOWPACK from season start to event date (already done)
  This:    Scour release cluster .sno files at event timestamp
  Pass 2:  Rerun SNOWPACK from event date to end of season

Called from forcing_pipeline.py as step_reinit(), or standalone:

    python src/snowpack-model-feeder/reinitialize_snowpack.py \
        --date-before 2026-01-14 --date-after 2026-01-20 \
        --event-date 2026-01-18 \
        --snapshot-date 2026-01-17

    # Dry run
    python src/snowpack-model-feeder/reinitialize_snowpack.py \
        --date-before 2026-01-14 --date-after 2026-01-20 \
        --event-date 2026-01-18 --dry-run

    # Use pre-drawn release GeoJSON instead of auto-detection
    python src/snowpack-model-feeder/reinitialize_snowpack.py \
        --release-geojson data/boundaries/avalanche_release_area.geojson \
        --event-date 2026-01-18
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                       "src" / "snowpack-model-feeder"))


# -----------------------------------------------------------------------
# .sno file reader/writer
# -----------------------------------------------------------------------

def read_sno(path: str) -> dict:
    """
    Parse a SNOWPACK .sno restart file.

    Returns dict with:
        header_lines : list of raw header lines (for passthrough)
        header       : dict of parsed header values
        fields       : list of column names
        layers       : list of dicts, one per layer (bottom to top)
    """
    header_lines = []
    header = {}
    fields = []
    layers = []
    in_header = True
    in_data = False

    with open(path) as f:
        for line in f:
            line_stripped = line.rstrip('\n')

            if in_header:
                header_lines.append(line_stripped)
                if line_stripped.strip() == '[DATA]':
                    in_header = False
                    in_data = True
                    continue
                if '=' in line_stripped and not line_stripped.startswith('#'):
                    key, val = line_stripped.split('=', 1)
                    key = key.strip()
                    val = val.strip()
                    header[key] = val
                    if key == 'fields':
                        fields = val.split()
                continue

            if in_data:
                parts = line_stripped.split()
                if len(parts) >= len(fields):
                    layer = {}
                    for i, fname in enumerate(fields):
                        layer[fname] = parts[i]
                    layers.append(layer)

    return {
        'header_lines': header_lines,
        'header': header,
        'fields': fields,
        'layers': layers,
        'path': path,
    }


def write_sno(sno_data: dict, path: str) -> None:
    """Write a modified .sno file."""
    with open(path, 'w') as f:
        for line in sno_data['header_lines']:
            f.write(line + '\n')
        for layer in sno_data['layers']:
            vals = [layer[fname] for fname in sno_data['fields']]
            f.write('     '.join(vals) + '\n')


def scour_sno(sno_data: dict,
              scour_depth_m: float,
              event_timestamp: str) -> dict:
    """
    Remove layers from the top of a .sno file to simulate avalanche scour.

    Parameters
    ----------
    sno_data       : parsed .sno from read_sno()
    scour_depth_m  : depth to remove from surface (m)
    event_timestamp: ISO timestamp for the new ProfileDate

    Returns modified sno_data (new copy).
    """
    import copy
    result = copy.deepcopy(sno_data)
    layers = result['layers']

    if not layers:
        return result

    # Layers are bottom to top — scour from the end
    removed_depth = 0.0
    n_original = len(layers)

    while layers and removed_depth < scour_depth_m:
        top_layer = layers[-1]
        layer_thick = float(top_layer['Layer_Thick'])

        remaining_to_scour = scour_depth_m - removed_depth

        if layer_thick <= remaining_to_scour:
            # Remove entire layer
            layers.pop()
            removed_depth += layer_thick
        else:
            # Partial removal — trim the top layer
            new_thick = layer_thick - remaining_to_scour
            top_layer['Layer_Thick'] = f"{new_thick:.6f}"
            removed_depth += remaining_to_scour

    # Compute new HS
    new_hs = sum(float(l['Layer_Thick']) for l in layers)
    n_removed = n_original - len(layers)

    # Update header
    _update_header_line(result, 'nSnowLayerData', str(len(layers)))
    _update_header_line(result, 'HS_Last', f"{new_hs:.6f}")
    _update_header_line(result, 'ProfileDate', event_timestamp)
    _update_header_line(result, 'ErosionLevel', str(max(0, len(layers) - 1)))

    result['scour_stats'] = {
        'n_layers_original': n_original,
        'n_layers_remaining': len(layers),
        'n_layers_removed': n_removed,
        'scour_depth_requested_m': scour_depth_m,
        'scour_depth_actual_m': removed_depth,
        'hs_new_m': new_hs,
    }

    return result


def _update_header_line(sno_data, key, new_value):
    """Update a header value in both the parsed dict and raw lines."""
    sno_data['header'][key] = new_value
    for i, line in enumerate(sno_data['header_lines']):
        if line.strip().startswith(key) and '=' in line:
            # Preserve original formatting/spacing
            parts = line.split('=', 1)
            # Right-pad key to original width
            sno_data['header_lines'][i] = f"{parts[0]}= {new_value}"
            return


# -----------------------------------------------------------------------
# Core workflow (callable from forcing_pipeline.py)
# -----------------------------------------------------------------------

def run_reinit(cfg,
               date_before: str = '2026-01-14',
               date_after: str = '2026-01-20',
               event_date: str = '2026-01-18',
               event_time: str = '12:00:00',
               snapshot_date: str = '2026-01-17',
               release_geojson: str = None,
               kernel_size: int = 7,
               threshold_sigma: float = 1.2,
               sno_dir: Path = None,
               dry_run: bool = False,
               no_backup: bool = False) -> dict:
    """
    Post-avalanche SNOWPACK reinitialization.

    Auto-detects (or loads) the release area, identifies release clusters,
    and scours their .sno restart files.

    Parameters
    ----------
    cfg : ProjectConfig
    date_before/after : survey dates bracketing the event
    event_date/time : avalanche event timestamp
    snapshot_date : SNOWPACK snapshot for slab thickness lookup
    release_geojson : path to pre-drawn release GeoJSON (None = auto-detect)
    kernel_size : min-kernel filter size
    threshold_sigma : min-kernel threshold in std devs
    sno_dir : directory with _res.sno files (default: cfg.pro_dir)
    dry_run : if True, report without modifying files
    no_backup : if True, skip .sno.bak creation

    Returns
    -------
    dict with keys: n_scoured, n_missing, n_skipped, stats (list of dicts)
    """
    import rasterio
    from avalanche import detect_minkernel, load_slope_mask, \
        build_persistent_noise_mask

    sno_dir = sno_dir or cfg.pro_dir
    event_ts = f"{event_date}T{event_time}"

    # --- Load DEM + cluster map ---
    with rasterio.open(str(cfg.dem_1m_path)) as src:
        dem = src.read(1).astype(np.float32)
        if src.nodata is not None:
            dem[dem == src.nodata] = np.nan
        transform = src.transform

    cluster_map = np.load(str(cfg.cluster_map_path))

    # --- Determine release mask ---
    if release_geojson:
        from snowpack_analysis import geojson_to_mask
        release_mask = geojson_to_mask(release_geojson, dem.shape, transform)
        print(f"Release mask from GeoJSON: {release_mask.sum()} cells")
    else:
        hs_before = np.load(str(cfg.resampled_dir /
                                f"hs_{date_before}.npy"))
        hs_after = np.load(str(cfg.resampled_dir /
                               f"hs_{date_after}.npy"))

        # Station dHS
        try:
            wx = pd.read_csv(str(cfg.weather_csv),
                             parse_dates=[0], index_col=0)
            hs_col = wx.iloc[:, 11]
            period = hs_col.loc[date_before:date_after]
            stn_dhs = float((period.iloc[-1] - period.iloc[0])
                            * 0.1 * 2.54)
        except Exception:
            stn_dhs = 0.0

        start_zone_mask = None
        if cfg.start_zone_kml.exists():
            start_zone_mask = load_slope_mask(
                cfg.start_zone_kml, dem.shape, transform)

        persistent_noise_mask = None
        all_surveys = sorted(cfg.resampled_dir.glob("hs_*.npy"))
        survey_pairs = []
        for i in range(len(all_surveys) - 1):
            d_b = all_surveys[i].name.replace("hs_", "").replace(".npy", "")
            d_a = all_surveys[i+1].name.replace("hs_", "").replace(".npy", "")
            if d_b == date_before and d_a == date_after:
                continue
            survey_pairs.append((all_surveys[i], all_surveys[i+1], 0.0))
        if survey_pairs:
            persistent_noise_mask = build_persistent_noise_mask(
                survey_pairs, dem, min_negative_fraction=0.6)

        print(f"Auto-detecting release area: {date_before} -> {date_after}")
        mk_result = detect_minkernel(
            hs_before, hs_after, dem, stn_dhs,
            kernel_size=kernel_size,
            threshold_sigma=threshold_sigma,
            start_zone_mask=start_zone_mask,
            persistent_noise_mask=persistent_noise_mask,
            transform=transform,
        )
        release_mask = mk_result['release_mask']
        print(f"  Release detected: {mk_result['release_area_m2']} m², "
              f"{mk_result['release_volume_m3']:.0f} m³")

    # --- Find release clusters ---
    release_cids = set(int(c) for c in np.unique(cluster_map[release_mask])
                       if c > 0)
    print(f"\nRelease clusters: {len(release_cids)}")

    # --- Get slab thickness per cluster ---
    feat_csv = cfg.analysis_dir / \
        f"all_start_zone_features_{snapshot_date}.csv"
    if not feat_csv.exists():
        feat_csv = cfg.analysis_dir / \
            f"release_zone_features_{snapshot_date}.csv"

    slab_thickness = {}
    if feat_csv.exists():
        snap_features = pd.read_csv(str(feat_csv), index_col=0)
        snap_features = snap_features[
            ~snap_features.index.duplicated(keep='first')]
        for cid in release_cids:
            if cid in snap_features.index and \
               'slab_thickness' in snap_features.columns:
                st = snap_features.loc[cid, 'slab_thickness']
                if isinstance(st, pd.Series):
                    st = st.iloc[0]
                try:
                    st = float(st)
                    if not np.isnan(st) and st > 0:
                        slab_thickness[cid] = st
                except (ValueError, TypeError):
                    pass
        print(f"  Slab thickness from features: "
              f"{len(slab_thickness)}/{len(release_cids)} clusters")
    else:
        print(f"  WARNING: {feat_csv.name} not found, "
              f"using 80% of HS as scour depth")

    # --- Scour .sno files ---
    print(f"\nEvent timestamp: {event_ts}")
    print(f"SNO directory: {sno_dir}")
    if dry_run:
        print("DRY RUN — no files will be modified\n")

    n_scoured = 0
    n_missing = 0
    n_skipped = 0
    stats = []

    for cid in sorted(release_cids):
        sno_name = f"cluster_{cid:04d}_res.sno"
        sno_path = sno_dir / sno_name

        if not sno_path.exists():
            sno_name = f"cluster_{cid:04d}_cluster_{cid:04d}_res.sno"
            sno_path = sno_dir / sno_name

        if not sno_path.exists():
            n_missing += 1
            continue

        sno_data = read_sno(str(sno_path))
        hs_before_scour = sum(float(l['Layer_Thick'])
                              for l in sno_data['layers'])

        if cid in slab_thickness:
            scour_m = slab_thickness[cid]
            source = 'features'
        else:
            scour_m = hs_before_scour * 0.80
            source = 'HS×0.80'

        # Cap scour at current HS — can't remove more snow than exists.
        # This handles timing mismatches where slab_thickness is from
        # the snapshot date but the .sno is from a different time
        # (e.g., end of season with melt/settlement).
        if scour_m > hs_before_scour:
            print(f"  {cid:04d}: scour {scour_m:.2f}m > HS {hs_before_scour:.2f}m "
                  f"— capping at HS")
            scour_m = hs_before_scour

        if scour_m <= 0.01:
            n_skipped += 1
            continue

        scoured = scour_sno(sno_data, scour_m, event_ts)
        ss = scoured['scour_stats']

        stats.append({
            'cid': cid,
            'hs_before_m': hs_before_scour,
            'scour_depth_m': scour_m,
            'scour_source': source,
            'n_layers_removed': ss['n_layers_removed'],
            'hs_after_m': ss['hs_new_m'],
        })

        if not dry_run:
            if not no_backup:
                shutil.copy2(str(sno_path), str(sno_path) + '.bak')
            write_sno(scoured, str(sno_path))

        n_scoured += 1
        print(f"  {sno_name}: HS {hs_before_scour:.2f}m → "
              f"{ss['hs_new_m']:.2f}m  "
              f"(scour {scour_m:.2f}m, {ss['n_layers_removed']} layers, "
              f"{source})")

    # --- Summary ---
    print(f"\n{'DRY RUN ' if dry_run else ''}Summary:")
    print(f"  Scoured:  {n_scoured}")
    print(f"  Missing:  {n_missing} (.sno file not found)")
    print(f"  Skipped:  {n_skipped} (scour depth ≤ 1cm)")

    if stats:
        scour_depths = [s['scour_depth_m'] for s in stats]
        hs_afters = [s['hs_after_m'] for s in stats]
        print(f"  Scour depth: median={np.median(scour_depths):.2f}m  "
              f"range=[{min(scour_depths):.2f}, {max(scour_depths):.2f}]m")
        print(f"  HS after:    median={np.median(hs_afters):.2f}m  "
              f"range=[{min(hs_afters):.2f}, {max(hs_afters):.2f}]m")

    if stats and not dry_run:
        stats_path = cfg.analysis_dir / f"reinit_stats_{event_date}.json"
        with open(str(stats_path), 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  Stats saved: {stats_path}")

    if not dry_run:
        print(f"\nNext step: rerun SNOWPACK from {event_date} "
              f"for the scoured clusters:")
        print(f"  bash snowpack/little_prof/run_snowpack.sh "
              f"{event_date}T00:00")

    return {
        'n_scoured': n_scoured,
        'n_missing': n_missing,
        'n_skipped': n_skipped,
        'stats': stats,
    }


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Post-avalanche SNOWPACK reinitialization")
    ap.add_argument('--project-dir', type=Path, default=Path('.'))
    ap.add_argument('--date-before', default='2026-01-14')
    ap.add_argument('--date-after', default='2026-01-20')
    ap.add_argument('--event-date', default='2026-01-18')
    ap.add_argument('--event-time', default='12:00:00')
    ap.add_argument('--snapshot-date', default='2026-01-17')
    ap.add_argument('--release-geojson', default=None)
    ap.add_argument('--kernel-size', type=int, default=7)
    ap.add_argument('--threshold-sigma', type=float, default=1.2)
    ap.add_argument('--sno-dir', default=None)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--no-backup', action='store_true')
    args = ap.parse_args()

    from config import ProjectConfig
    cfg = ProjectConfig(project_dir=args.project_dir)

    run_reinit(
        cfg=cfg,
        date_before=args.date_before,
        date_after=args.date_after,
        event_date=args.event_date,
        event_time=args.event_time,
        snapshot_date=args.snapshot_date,
        release_geojson=args.release_geojson,
        kernel_size=args.kernel_size,
        threshold_sigma=args.threshold_sigma,
        sno_dir=Path(args.sno_dir) if args.sno_dir else None,
        dry_run=args.dry_run,
        no_backup=args.no_backup,
    )


if __name__ == '__main__':
    main()
    
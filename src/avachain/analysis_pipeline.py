"""
analysis_pipeline.py — Post-SNOWPACK analysis pipeline (Pipeline B).

Steps (run after SNOWPACK finishes and Zarr cache is available):
    cluster_update  Mid-season cluster splitting (run after new survey lands)
    zarr_build      Build / resume Zarr cache from .pro files
    analyze         Snapshot comparison, Meloche features, deviation plots
    scenarios       Generate AvaFrame com1DFA release scenario ensemble

Usage:
    python analysis_pipeline.py cluster_update
    python analysis_pipeline.py zarr_build
    python analysis_pipeline.py analyze --snapshot-date 2026-01-18
    python analysis_pipeline.py scenarios --snapshot-date 2026-01-18
    python analysis_pipeline.py all --snapshot-date 2026-01-18

Requires Pipeline A (pipeline.py) to have been run first to produce:
    outputs/analysis/cluster_map.npy
    outputs/analysis/release_zone_groups.json  (from step_avalanche)
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

# ---------------------------------------------------------------------------
# Module imports — all logic lives in the modules below
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ProjectConfig
from snowpack_io import load_dataset, load_boundaries, kml_to_mask
from snowpack_analysis import (
    assign_cluster_groups,
    profile_features,
    extract_group_timeseries,
    compute_meloche_features,
    GROUP_COLORS, GROUP_LABELS,
)
from release_geometry import (
    compute_slope_aspect,
    make_release_polygon_2d,
    propagate_release,
    depth_from_snowpack,
    rasterize_release_polygon,
    plot_release_comparison,
)
from scenario_writer import (
    write_trigger_locations,
    write_scenario,
    write_scenario_weights,
    write_summary_csv,
    write_metadata,
)

# ---------------------------------------------------------------------------
# Defaults — path and parameter defaults are in config.py.
# Only the snapshot date stays here (event-specific CLI default).
# ---------------------------------------------------------------------------
DEFAULT_SNAPSHOT = "2026-01-18"  # YYYY-MM-DD


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_dem(cfg):
    dem_tif = cfg.resampled_dir / "dem_1m.tif"
    with rasterio.open(str(dem_tif)) as src:
        dem       = src.read(1).astype(np.float32)
        dem[dem == src.nodata] = np.nan
        transform = src.transform
        crs_wkt   = src.crs.to_wkt()
        profile   = src.profile.copy()
    return dem, transform, crs_wkt, profile


def _load_groups(cfg, dem, transform, snapshot_date=None):
    """Load or build release/adjacent/reference cluster group assignments.

    Groups are date-specific: all observed release GeoJSONs up to
    snapshot_date are unioned to form the release mask.  When no events
    exist for the date (operational no-avalanche day), the release group
    is empty and reference terrain-matching is skipped.
    """
    cache_name = (f"release_zone_groups_{snapshot_date}.json"
                  if snapshot_date else "release_zone_groups.json")
    groups_path = cfg.analysis_dir / cache_name
    if groups_path.exists():
        with open(str(groups_path)) as f:
            raw = json.load(f)
        return {k: set(v) for k, v in raw.items()}

    print("Building cluster group assignments from boundaries...")
    from snowpack_analysis import geojson_to_mask
    cluster_map = np.load(str(cfg.analysis_dir / "cluster_map.npy"))

    release_mask = np.zeros(dem.shape, dtype=bool)
    if snapshot_date:
        gj_paths = cfg.release_geojsons_for_date(snapshot_date)
        for gj_path in gj_paths:
            release_mask |= geojson_to_mask(gj_path, dem.shape, transform)
        if gj_paths:
            print(f"  Release mask from {len(gj_paths)} event GeoJSON(s), "
                  f"{release_mask.sum()} cells")
        else:
            print("  No release area GeoJSONs for this date — empty release group")
    else:
        if cfg.release_geojson.exists():
            release_mask = geojson_to_mask(cfg.release_geojson, dem.shape, transform)

    start_zone_mask = kml_to_mask(cfg.start_zone_kml, dem.shape, transform)
    domain_mask     = ~np.isnan(dem)

    groups = assign_cluster_groups(
        cluster_map, release_mask, start_zone_mask, dem, domain_mask)

    groups_path.write_text(
        json.dumps({k: list(v) for k, v in groups.items()}, indent=2))
    print(f"  Groups saved: { {k: len(v) for k, v in groups.items()} }")
    return groups


def _trigger_weights(n: int, sk38_values: list) -> list:
    """
    Inverse-Sk38 weights for trigger locations.
    Lower Sk38 = more unstable = higher probability of being the trigger.
    """
    inv = [1.0 / max(s, 0.01) for s in sk38_values]
    total = sum(inv)
    return [v / total for v in inv]


def _size_weights(size_factors: list) -> list:
    """
    Log-normal weights centred on the median size factor.
    Tails (P10/P90) get less weight than median.
    """
    from scipy.stats import norm
    n = len(size_factors)
    # Map factors to quantiles assuming log-normal
    log_f = [np.log(f) for f in size_factors]
    mu    = np.mean(log_f)
    sigma = max(np.std(log_f), 0.01)
    probs = [norm.pdf(lf, mu, sigma) for lf in log_f]
    total = sum(probs)
    return [p / total for p in probs]


def _depth_weights(depth_pcts: list) -> list:
    """Simple weights: P50 most likely, P10/P90 less so."""
    weights = {10: 0.20, 25: 0.15, 50: 0.60, 75: 0.15, 90: 0.20}
    raw = [weights.get(p, 1.0 / len(depth_pcts)) for p in depth_pcts]
    total = sum(raw)
    return [r / total for r in raw]


def _load_observed_release_polygon(cfg, snapshot_date=None):
    """
    Load the most recent observed release area GeoJSON up to snapshot_date
    and reproject to UTM.  Returns a single shapely Polygon.
    """
    import json
    from shapely.geometry import shape
    from shapely.ops import unary_union
    from pyproj import Transformer

    if snapshot_date:
        gj_paths = cfg.release_geojsons_for_date(snapshot_date)
        gj_path = gj_paths[-1] if gj_paths else None
    else:
        gj_path = cfg.release_geojson if cfg.release_geojson.exists() else None

    if gj_path is None or not gj_path.exists():
        raise FileNotFoundError(
            f"No observed release GeoJSON found"
            + (f" for date ≤ {snapshot_date}" if snapshot_date else "")
        )

    with open(str(gj_path)) as f:
        gj = json.load(f)

    polys  = [shape(feat['geometry']) for feat in gj['features']]
    merged = unary_union(polys)

    t = Transformer.from_crs('EPSG:4326', 'EPSG:32613', always_xy=True)

    def _reproj(poly):
        from shapely.geometry import Polygon, MultiPolygon
        def ring(r): return [t.transform(x, y) for x, y in r.coords]
        if poly.geom_type == 'Polygon':
            return Polygon(ring(poly.exterior),
                           [ring(i) for i in poly.interiors])
        return MultiPolygon([_reproj(p) for p in poly.geoms])

    utm_poly = _reproj(merged)
    if utm_poly.geom_type == 'MultiPolygon':
        utm_poly = max(utm_poly.geoms, key=lambda p: p.area)

    print(f"Observed release polygon: {utm_poly.area:.0f} m²  "
          f"({gj_path.name})")
    return utm_poly


# ---------------------------------------------------------------------------
# Step: zarr_build
# ---------------------------------------------------------------------------

def step_zarr_build(cfg, args):
    """Build or resume Zarr cache from .pro files."""
    from snowpack_io import build_zarr_cache
    build_zarr_cache(
        pro_dir    = args.pro_dir,
        zarr_out   = args.zarr_path,
        batch_size = args.batch_size,
        n_cpus     = args.workers,
        max_layers = args.max_layers,
    )


# ---------------------------------------------------------------------------
# Step: analyze
# ---------------------------------------------------------------------------

def step_analyze(cfg, args):
    """
    Snapshot feature extraction.

    Always produces:
        outputs/analysis/all_start_zone_features_YYYY-MM-DD.csv
        outputs/analysis/meloche_features_all_YYYY-MM-DD.csv

    With --plots also produces:
        outputs/plots/release_zone_comparison_YYYY-MM-DD.png
        outputs/plots/release_zone_deviations_YYYY-MM-DD.png
        outputs/plots/meloche_comparison_YYYY-MM-DD.png
    """
    import subprocess
    script = Path(__file__).resolve().parent / "analyze_release_zone.py"
    cmd    = [sys.executable, str(script),
              '--project-dir', str(cfg.project_dir),
              '--pro-dir',     str(args.pro_dir),
              '--zarr-path',   str(args.zarr_path),
              '--snapshot-date', args.snapshot_date]
    if args.classifier:
        cmd.append('--classifier')
    if args.plots:
        cmd.append('--plots')
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Step: scenarios
# ---------------------------------------------------------------------------

def step_scenarios(cfg, args):
    """
    Generate AvaFrame com1DFA release scenario ensemble.

    Axes:
      - Trigger location:   top-N weakest Sk38 clusters in start zone
      - Release size:       A_ca × size_factor (Meloche brittle scaling)
      - Slab depth:         SNOWPACK HS × depth_scale at given percentile
    """
    print(f"\n=== step_scenarios  snapshot={args.snapshot_date} ===")

    # --- Load inputs ---
    dem, transform, crs_wkt, raster_profile = _load_dem(cfg)
    cluster_map = np.load(str(cfg.analysis_dir / "cluster_map.npy"))
    ds          = load_dataset(args.pro_dir, zarr_path=args.zarr_path)
    groups      = _load_groups(cfg, dem, transform, args.snapshot_date)

    start_zone_mask = kml_to_mask(cfg.start_zone_kml, dem.shape, transform)

    # --- Load pre-computed analysis outputs ---
    # The analyze step now produces full start zone features by default.
    # Fall back to group-level CSVs for backward compatibility with
    # older analysis outputs.
    feat_csv_all = cfg.analysis_dir / f"all_start_zone_features_{args.snapshot_date}.csv"
    feat_csv_grp = cfg.analysis_dir / f"release_zone_features_{args.snapshot_date}.csv"
    mel_csv_all  = cfg.analysis_dir / f"meloche_features_all_{args.snapshot_date}.csv"
    mel_csv_grp  = cfg.analysis_dir / f"meloche_features_{args.snapshot_date}.csv"

    if feat_csv_all.exists():
        feat_csv = feat_csv_all
    elif feat_csv_grp.exists():
        feat_csv = feat_csv_grp
        print(f"  WARNING: using group-level features ({feat_csv_grp.name}). "
              f"Rerun 'analyze' step to generate full start zone coverage.")
    else:
        raise FileNotFoundError(
            f"Run 'analyze' step first — missing features CSV")

    mel_csv = mel_csv_all if mel_csv_all.exists() else mel_csv_grp

    snap_features = pd.read_csv(str(feat_csv), index_col=0)
    snap_features = snap_features[~snap_features.index.duplicated(keep='first')]
    meloche_df    = pd.read_csv(str(mel_csv),  index_col=0) \
                   if mel_csv.exists() else pd.DataFrame()
    if not meloche_df.empty:
        meloche_df = meloche_df[~meloche_df.index.duplicated(keep='first')]
    print(f"  Features: {len(snap_features)} clusters from {feat_csv.name}")
    print(f"  Meloche:  {len(meloche_df)} clusters from {mel_csv.name}")

    # --- Find trigger locations ---
    # Candidate pool: ALL clusters within the start zone that have features.
    # Group assignment (release/adjacent/reference) is for validation
    # comparison, NOT for trigger selection — operationally we don't know
    # which clusters will be in the release area.
    MIN_TAU_G   = 40.0   # Pa — below this the scaling law is not applicable
    MIN_SLOPE   = 30.0   # degrees — well above the 27° friction angle
    MAX_SK38    = 1.0    # Sk38 >= 1.0 = stable for skier triggering (Schweizer & Jamieson 2007)
    MIN_SLAB_THICKNESS = 0.5  # metres — too thin = D1 at most
    MAX_SLAB_THICKNESS = args.max_slab_thickness  # metres — too thick = skier can't trigger

    # All clusters that overlap the start zone mask
    sz_cids = set(int(c) for c in np.unique(cluster_map[start_zone_mask])
                  if c > 0)
    candidate_ids = [cid for cid in sz_cids
                     if cid in snap_features.index]
    print(f"  Start zone clusters with features: {len(candidate_ids)}")

    def _scalar_val(df, cid, col):
        """Safely get a scalar from a possibly multi-row index."""
        if cid not in df.index or col not in df.columns:
            return np.nan
        v = df.loc[cid, col]
        if isinstance(v, pd.Series): v = v.iloc[0]
        return float(v)

    MIN_SLOPE_TRIGGER = args.stauchwall_deg + 2.0  # must be above stauchwall

    if not meloche_df.empty and 'tau_g' in meloche_df.columns:
        # Pre-compute per-cluster mean slope from DEM
        slope_grid, _ = compute_slope_aspect(dem)
        def _mean_slope(cid):
            pxs = np.argwhere(cluster_map == cid)
            if len(pxs) == 0: return 0.0
            return float(slope_grid[pxs[:,0], pxs[:,1]].mean())

        # Also filter by Sk38 — Sk38 >= 0.5 is stable terrain
        sk38_col = 'min_sk38'
        candidate_ids = [
            cid for cid in candidate_ids
            if cid in meloche_df.index
            and _scalar_val(meloche_df, cid, 'tau_g') >= MIN_TAU_G
            and not np.isnan(_scalar_val(meloche_df, cid, 'tau_g'))
            and _mean_slope(cid) >= MIN_SLOPE_TRIGGER
            and (cid not in snap_features.index
                 or _scalar_val(snap_features, cid, sk38_col) < MAX_SK38)
        ]
        print(f"  Candidates after tau_g >= {MIN_TAU_G} Pa, "
              f"slope >= {MIN_SLOPE_TRIGGER:.1f}°, "
              f"Sk38 < {MAX_SK38} filter: "
              f"{len(candidate_ids)}")
        candidate_ids = [
            cid for cid in candidate_ids
            if ((_scalar_val(snap_features, cid, 'slab_thickness') >= MIN_SLAB_THICKNESS
                 and _scalar_val(snap_features, cid, 'slab_thickness') <= MAX_SLAB_THICKNESS)
                or np.isnan(_scalar_val(snap_features, cid, 'slab_thickness')))
        ]
        print(f"  Candidates after slab_thickness {MIN_SLAB_THICKNESS}-{MAX_SLAB_THICKNESS}m filter: "
            f"{len(candidate_ids)}")
    # Optional: restrict to observed release area (validation mode only)
    if args.restrict_to_release_area:
        from snowpack_analysis import geojson_to_mask
        gj_path = cfg.release_geojson
        if gj_path.exists():
            release_area_mask = geojson_to_mask(gj_path, dem.shape, transform)
            candidate_ids = [
                cid for cid in candidate_ids
                if np.any(release_area_mask[cluster_map == cid])]
            print(f"  Candidates after release area restriction: "
                  f"{len(candidate_ids)}")
        else:
            print(f"  Warning: --restrict-to-release-area set but "
                  f"{gj_path} not found — skipping")

    # Elevation filter: triggers must be in upper portion of candidate zone
    # Use P75 in validation mode (release area = known crown terrain)
    # Use P50 in operational mode (full start zone)
    elev_pct = 75 if args.restrict_to_release_area else 50
    if candidate_ids:
        elevs = []
        for cid in candidate_ids:
            pxs = np.argwhere(cluster_map == cid)
            if len(pxs):
                elevs.append((cid, float(dem[pxs[:,0], pxs[:,1]].mean())))
        if elevs:
            elev_vals = [e for _, e in elevs]
            elev_threshold = np.percentile(elev_vals, elev_pct)
            candidate_ids = [cid for cid, e in elevs
                             if e >= elev_threshold]
            print(f"  Candidates after elevation >= P{elev_pct} "
                  f"({elev_threshold:.0f}m) filter: {len(candidate_ids)}")

    if not candidate_ids:
        raise ValueError("No candidate trigger clusters found in start zone")

    cand_df  = snap_features.loc[candidate_ids].copy()
    sk38_col = 'min_sk38' if 'min_sk38' in cand_df.columns else 'sk38_min'

    # Join Pi1 from meloche features for propagation gate.
    # A trigger that can't propagate (low Pi1) produces a degenerate release —
    # good initiation potential (low Sk38) alone is not sufficient.
    if not meloche_df.empty and 'Pi1_elastic' in meloche_df.columns:
        pi1_vals = {}
        for cid in cand_df.index:
            if cid in meloche_df.index:
                v = meloche_df.loc[cid, 'Pi1_elastic']
                pi1_vals[cid] = float(v.iloc[0]) if isinstance(v, pd.Series) else float(v)
        cand_df['Pi1_elastic'] = pd.Series(pi1_vals)

        pi1_valid = cand_df['Pi1_elastic'].dropna()
        if len(pi1_valid) > 0:
            pi1_median = float(pi1_valid.median())
            pi1_mean   = float(pi1_valid.mean())
            pre_gate   = cand_df.dropna(subset=[sk38_col]).shape[0]
            cand_df    = cand_df[cand_df['Pi1_elastic'] >= pi1_median]
            print(f"  Pi1 propagation gate (>= median {pi1_median:.2f}, "
                  f"mean {pi1_mean:.2f}): "
                  f"{pre_gate} → {cand_df.shape[0]} candidates")

    triggers = (cand_df.dropna(subset=[sk38_col])
                       .nsmallest(args.n_triggers, sk38_col))

    # Validate triggers: drop any that can't produce a release polygon
    # Test with median size_factor — if it fails at median, it fails at all
    if not args.use_observed_release and not args.no_propagation:
        valid_triggers = []
        _mid_sf = args.size_factors[len(args.size_factors) // 2]
        for t_cid_check in triggers.index:
            _a_ca_check = 50.0
            if not meloche_df.empty and t_cid_check in meloche_df.index:
                _v = meloche_df.loc[t_cid_check, 'A_ca_brittle']
                if isinstance(_v, pd.Series): _v = _v.iloc[0]
                if not np.isnan(float(_v)):
                    _a_ca_check = min(float(_v), 300.0)
            _poly_check = make_release_polygon_2d(
                trigger_cluster_id=t_cid_check,
                A_ca=_a_ca_check,
                meloche_df=meloche_df,
                cluster_map=cluster_map,
                dem=dem,
                transform=transform,
                start_zone_mask=start_zone_mask,
                snap_features=snap_features,
                size_factor=_mid_sf,
                stauchwall_deg=args.stauchwall_deg,
                mode3_scale=args.mode3_scale,
                use_propagation=True,
            )
            if _poly_check is not None:
                valid_triggers.append(t_cid_check)
            else:
                print(f"  Dropped trigger cid={t_cid_check} — "
                      f"propagate_release returned None at "
                      f"size_factor={_mid_sf}")
        if valid_triggers:
            triggers = triggers.loc[valid_triggers]
        else:
            print("  WARNING: all triggers failed propagation — "
                  "keeping original set with rectangle fallback")

    print(f"Trigger clusters: {list(triggers.index)} "
          f"Sk38={list(triggers[sk38_col].round(3))}")

    # --- Output directory ---
    base = cfg.output_dir / "scenarios" / args.snapshot_date
    out_dir = base / args.forecast_horizon if args.forecast_horizon else base
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scenarios").mkdir(exist_ok=True)

    # Write trigger locations GeoJSON
    write_trigger_locations(
        triggers, cluster_map, transform,
        out_dir / "trigger_locations.geojson")

    # --- Slab depth grid for snapshot ---
    full_depth = depth_from_snowpack(cluster_map, ds, args.snapshot_date,
                                     snap_features=snap_features)

    # --- Scenario density (release zone median from features) ---
    rel_df       = snap_features[snap_features.index.isin(groups.get('release', set()))]
    density_mean = float(rel_df['slab_density'].median()
                         if 'slab_density' in rel_df.columns else 280.0)
    density_std  = float(rel_df['slab_density'].std()
                         if 'slab_density' in rel_df.columns else 20.0)

    # --- Precompute weights per axis ---
    sk38_vals     = list(triggers[sk38_col])
    t_weights     = _trigger_weights(len(triggers), sk38_vals)
    s_weights     = _size_weights(args.size_factors)
    d_weights     = _depth_weights(args.depth_pcts)

    # --- Load observed release polygon if validation mode ---
    observed_polygon = None
    if args.use_observed_release:
        observed_polygon = _load_observed_release_polygon(cfg, args.snapshot_date)
        print("Mode: OBSERVED RELEASE (validation — flow model only)")
    else:
        print("Mode: MELOCHE-DERIVED RELEASE (full chain)")

    # --- Generate scenarios ---
    rows            = []
    weight_map      = {}
    a_ca_list       = []
    median_polygons = []   # (polygon, size_factor) for comparison plot
    most_likely     = {'weight': -1.0, 'scenario_id': None, 'polygon': None,
                       'size_factor': None, 'trigger': None}  # highest-weight scenario

    for t_rank, (t_cid, _) in enumerate(triggers.iterrows()):
        # A_ca from Meloche features, fallback to 50m
        A_ca    = 50.0   # default fallback (m)
        A_CA_MAX = 300.0 # cap — start zone is ~250m wide; beyond this
                         # the scaling law is unreliable (very low tau_g)
        if not meloche_df.empty and t_cid in meloche_df.index:
            val = meloche_df.loc[t_cid, 'A_ca_brittle']
            if isinstance(val, pd.Series):
                val = val.iloc[0]
            if not np.isnan(float(val)):
                A_ca = min(float(val), A_CA_MAX)
        a_ca_list.append(A_ca)

        # Extract hand hardness at trigger cluster from SNOWPACK profile
        hh_depth_m, hh_depth_mid_m, hh_value = None, None, None
        snap_ts = np.datetime64(f"{args.snapshot_date}T18:00")
        locs = ds.coords['location'].values
        # Match location name format: 'cluster_XXXX_cluster_XXXX' or 'cluster_XXXX'
        t_loc = None
        for loc in locs:
            loc_str = str(loc)
            cid_parsed = int(loc_str.split('_')[-1])
            if cid_parsed == t_cid:
                t_loc = loc_str
                break

        if t_loc is None:
            print(f"    hand_hardness: trigger cid={t_cid} not found in Zarr locations")
        else:
            tidx = int(np.argmin(np.abs(ds.coords['time'].values - snap_ts)))
            ds_t = ds.sel(location=t_loc).isel(time=tidx)
            sq = [d for d in ['slope', 'realization'] if d in ds_t.dims]
            if sq:
                ds_t = ds_t.squeeze(sq)
            if 'hand_hardness' not in ds_t:
                print(f"    scour_depth: hand_hardness not in Zarr dataset")
            else:
                try:
                    hh = ds_t['hand_hardness'].values.ravel()
                    hs_cm = float(np.nanmean(ds_t['HS'].values))
                    n_layers = len(hh)

                    # SNOWPACK hand hardness uses negative values:
                    #   -1=F, -2=4F, -3=1F, -4=P, -5=K, -6=I
                    # More negative = harder. "> 1F" means hh < -3.
                    has_hh = ~np.isnan(hh) & (hh < 0)
                    n_valid = int(np.sum(has_hh))

                    # Find the layer height variable
                    z_var = None
                    for zname in ('height', 'z', 'layer_height'):
                        if zname in ds_t:
                            z_var = zname
                            break

                    if z_var is None:
                        print(f"    scour_depth: no height variable found")
                    else:
                        z = ds_t[z_var].values.ravel()  # cm, per layer
                        # Ensure z and hh have same length
                        n_layers = min(len(hh), len(z))
                        hh = hh[:n_layers]
                        z = z[:n_layers]
                        has_hh = has_hh[:n_layers]

                        # Get slab thickness = depth from surface to failure plane
                        slab_thick_m = None
                        if t_cid in snap_features.index and \
                           'slab_thickness' in snap_features.columns:
                            st = snap_features.loc[t_cid, 'slab_thickness']
                            if isinstance(st, pd.Series):
                                st = st.iloc[0]
                            try:
                                slab_thick_m = float(st)
                            except (ValueError, TypeError):
                                pass

                        if slab_thick_m is None or np.isnan(slab_thick_m):
                            print(f"    scour_depth: no slab_thickness "
                                  f"for trigger cid={t_cid}")
                        else:
                            # Failure plane height from ground (cm)
                            fp_height_cm = hs_cm - slab_thick_m * 100.0
                            fp_mid_cm = hs_cm - slab_thick_m * 50.0  # midpoint

                            # Find the layer index at the failure plane
                            # Layers are bottom-to-top, z is height from ground
                            fp_layer_idx = None
                            for li in range(n_layers - 1, -1, -1):
                                if z[li] <= fp_height_cm:
                                    fp_layer_idx = li
                                    break

                            if fp_layer_idx is None:
                                fp_layer_idx = 0

                            print(f"    scour_depth: {n_layers} layers, "
                                  f"{n_valid} with valid HH, "
                                  f"HS={hs_cm:.1f}cm")
                            print(f"    Failure plane at "
                                  f"{fp_height_cm:.1f}cm from ground "
                                  f"(slab={slab_thick_m:.2f}m, "
                                  f"layer idx={fp_layer_idx})")

                            # Scan DOWNWARD from failure plane to find
                            # first hard layer (HH harder than 1F = hh < -3)
                            # This is the entrainable depth — how deep the
                            # avalanche can scour before hitting resistance.
                            hard_layer_idx = None
                            for li in range(fp_layer_idx, -1, -1):
                                if has_hh[li] and hh[li] < -3:
                                    hard_layer_idx = li
                                    break

                            if hard_layer_idx is not None:
                                hard_z = float(z[hard_layer_idx])
                                # Depth from bottom of failure plane
                                scour_bottom = (fp_height_cm - hard_z) / 100.0
                                # Depth from midpoint of failure plane
                                scour_mid = (fp_mid_cm - hard_z) / 100.0
                                hh_value = float(hh[hard_layer_idx])

                                # Use bottom-of-failure-plane as primary
                                hh_depth_m = max(scour_bottom, 0.0)
                                hh_depth_mid_m = max(scour_mid, 0.0)

                                print(f"    → Hard layer (HH={hh_value:.1f}) "
                                      f"at {hard_z:.1f}cm from ground")
                                print(f"    → Scour from failure plane bottom: "
                                      f"{scour_bottom:.2f}m")
                                print(f"    → Scour from failure plane mid: "
                                      f"{scour_mid:.2f}m")
                            else:
                                # No hard layer below failure plane —
                                # entrainable all the way to ground
                                hh_depth_m = fp_height_cm / 100.0
                                hh_depth_mid_m = fp_mid_cm / 100.0
                                hh_value = None
                                print(f"    → No hard layer below failure "
                                      f"plane — scour to ground: "
                                      f"{hh_depth_m:.2f}m")

                except Exception as e:
                    print(f"    scour_depth extraction failed for "
                          f"cid={t_cid}: {e}")
                    import traceback
                    traceback.print_exc()

        for s_idx, size_f in enumerate(args.size_factors):
            for d_idx, d_pct in enumerate(args.depth_pcts):

                # Build release polygon
                if args.use_observed_release:
                    # Use observed Jan 18 GeoJSON — validation mode
                    # Still scale by size_factor to bracket uncertainty
                    from shapely import affinity
                    centroid = observed_polygon.centroid
                    polygon  = affinity.scale(
                        observed_polygon,
                        xfact=size_f, yfact=size_f,
                        origin=centroid)
                else:
                    polygon = make_release_polygon_2d(
                        trigger_cluster_id = t_cid,
                        A_ca               = A_ca,
                        meloche_df         = meloche_df,
                        cluster_map        = cluster_map,
                        dem                = dem,
                        transform          = transform,
                        start_zone_mask    = start_zone_mask,
                        snap_features      = snap_features,
                        size_factor        = size_f,
                        stauchwall_deg     = args.stauchwall_deg,
                        mode3_scale        = args.mode3_scale,
                        use_propagation    = not args.no_propagation,
                    )

                if polygon is None:
                    print(f"  Skip scenario (degenerate polygon): "
                          f"trigger={t_cid} size_f={size_f}")
                    continue

                # Scale depth by percentile
                depth_scale  = cfg.depth_scales.get(d_pct, 1.0)
                scaled_depth = full_depth * depth_scale
                release_depth = rasterize_release_polygon(
                    polygon, scaled_depth, dem.shape, transform)

                # Scenario weight = trigger × size × depth
                weight = t_weights[t_rank] * s_weights[s_idx] * d_weights[d_idx]

                scenario_id = f"scenario_{len(rows)+1:03d}"
                # Capture reference polygon ONCE per trigger
                # Use middle size_factor in the list (closest to 1.0)
                _mid = args.size_factors[len(args.size_factors) // 2]
                if abs(size_f - _mid) < 1e-9 and d_idx == 0:
                    label = (f"T{t_rank+1}  cid={t_cid}  "
                             f"Sk38={sk38_vals[t_rank]:.2f}  "
                             f"Aca={A_ca:.0f}m")
                    median_polygons.append((polygon, size_f, label))

                row = write_scenario(
                    scenario_dir       = out_dir / "scenarios",
                    scenario_id        = scenario_id,
                    release_polygon    = polygon,
                    depth_raster       = release_depth,
                    dem_shape          = dem.shape,
                    transform          = transform,
                    crs_wkt            = crs_wkt,
                    density_mean       = density_mean,
                    density_std        = density_std,
                    trigger_cluster_id = t_cid,
                    A_ca               = A_ca,
                    size_factor        = size_f,
                    depth_percentile   = d_pct,
                    weight             = weight,
                    mu                 = args.mu,
                    xi                 = args.xi,
                    profile            = raster_profile,
                    scour_depth_m         = hh_depth_m,
                    scour_depth_mid_m     = hh_depth_mid_m,
                    hand_hardness_value   = hh_value,
                )
                # Augment row with trigger rank and sk38
                row['sk38_rank'] = t_rank + 1
                row['min_sk38']  = float(sk38_vals[t_rank])
                rows.append(row)
                weight_map[scenario_id] = weight

                # Track the highest-weight scenario for the comparison plot.
                # (argmax is unaffected by the later normalisation.)
                if weight > most_likely['weight']:
                    most_likely = {'weight': weight, 'scenario_id': scenario_id,
                                   'polygon': polygon, 'size_factor': size_f,
                                   'trigger': t_cid}

    if not rows:
        print("WARNING: No scenarios generated — check trigger clusters and Meloche features")
        return out_dir

    # Normalise weights to sum to 1
    total_w = sum(weight_map.values())
    weight_map = {k: v / total_w for k, v in weight_map.items()}
    for row in rows:
        row['weight'] = weight_map.get(row['scenario_id'], row['weight'])

    # Write ensemble outputs
    write_scenario_weights(weight_map, out_dir / "scenario_weights.json")
    write_summary_csv(rows, out_dir / "summary.csv")
    write_metadata(
        out_path          = out_dir / "metadata.json",
        snapshot_date     = args.snapshot_date,
        n_scenarios       = len(rows),
        n_triggers        = len(triggers),
        size_factors      = args.size_factors,
        depth_percentiles = args.depth_pcts,
        mu                = args.mu,
        xi                = args.xi,
        a_ca_stats        = {
            'mean_m':   round(float(np.mean(a_ca_list)), 1),
            'min_m':    round(float(np.min(a_ca_list)),  1),
            'max_m':    round(float(np.max(a_ca_list)),  1),
        },
    )

    # --- Release comparison plot (research/debug; skip in operational default) ---
    if not args.plots:
        print(f"\n{len(rows)} scenarios written → {out_dir}")
        print(f"  Triggers: {len(triggers)}  "
              f"Size factors: {len(args.size_factors)}  "
              f"Depth pcts: {len(args.depth_pcts)}")
        print(f"  A_ca range: {min(a_ca_list):.1f}–{max(a_ca_list):.1f} m")
        print(f"  Density: {density_mean:.1f} ± {density_std:.1f} kg/m³")
        return out_dir

    import traceback
    print(f"\nGenerating release comparison plot...")
    print(f"  median_polygons collected: {len(median_polygons)}")
    mode_str  = "observed" if args.use_observed_release else "meloche"
    plot_path = out_dir / f"release_comparison_{mode_str}_{args.snapshot_date}_physical_model.png"
    print(f"  Output path: {plot_path}")
    try:
        obs_poly   = _load_observed_release_polygon(cfg, args.snapshot_date)
        poly_pairs = [(p, sf) for p, sf, _ in median_polygons]
        t_labels   = [lbl for _, _, lbl in median_polygons]

        # Compute trigger centroids from cluster_map (stable position)
        t_centroids = []
        for t_cid_plot in triggers.index:
            px = np.argwhere(cluster_map == t_cid_plot)
            if len(px):
                cx = transform.c + px[:, 1].mean() * transform.a
                cy = transform.f + px[:, 0].mean() * transform.e
                t_centroids.append((cx, cy))
            else:
                t_centroids.append(None)

        ml_poly  = most_likely['polygon']
        ml_label = None
        if most_likely['scenario_id'] is not None:
            ml_label = (f"Most likely: {most_likely['scenario_id']} "
                        f"(p={weight_map.get(most_likely['scenario_id'], 0.0):.3f}, "
                        f"sf={most_likely['size_factor']:.2f})")

        plot_release_comparison(
            meloche_polygons = poly_pairs,
            observed_polygon = obs_poly,
            dem              = dem,
            transform        = transform,
            start_zone_mask  = start_zone_mask,
            trigger_labels   = t_labels,
            trigger_centroids = t_centroids,
            most_likely_polygon = ml_poly,
            most_likely_label   = ml_label,
            out_path         = plot_path,
            title            = (f"Release polygon comparison — {args.snapshot_date} | "
                               f"Little Professor | Jan 18 event\n"
                               f"Blue = Meloche-derived  Red = Observed  "
                               f"Green = Start zone  Black dashed = Most likely"),
        )
    except Exception:
        print("WARNING: comparison plot failed — full traceback:")
        traceback.print_exc()

    print(f"\n{len(rows)} scenarios written → {out_dir}")
    print(f"  Triggers: {len(triggers)}  "
          f"Size factors: {len(args.size_factors)}  "
          f"Depth pcts: {len(args.depth_pcts)}")
    print(f"  A_ca range: {min(a_ca_list):.1f}–{max(a_ca_list):.1f} m")
    print(f"  Density: {density_mean:.1f} ± {density_std:.1f} kg/m³")
    return out_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Post-SNOWPACK analysis pipeline (Pipeline B)")
    parser.add_argument('step',
        choices=['cluster_update', 'zarr_build', 'analyze', 'scenarios', 'all'],
        help="Pipeline step to run")

    # Common
    parser.add_argument('--project-dir', type=Path, default=Path('.'))
    parser.add_argument('--pro-dir',     type=Path, default=None)
    parser.add_argument('--zarr-path',   type=Path, default=None)
    parser.add_argument('--snapshot-date', default=DEFAULT_SNAPSHOT,
                        help='SNOWPACK snapshot date YYYY-MM-DD')

    # zarr_build
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--workers',    type=int, default=31)
    parser.add_argument('--max-layers', type=int, default=338)

    # analyze
    parser.add_argument('--classifier', action='store_true')
    parser.add_argument('--plots', action='store_true',
                        help='Generate research/diagnostic plots '
                             '(analyze: comparison, deviations, Meloche; '
                             'scenarios: release comparison). Off by default.')

    # scenarios
    parser.add_argument('--forecast-horizon', default=None,
                        help='Forecast horizon label (e.g. T_1, T_2). '
                             'When set, scenarios are written to '
                             'outputs/scenarios/YYYY-MM-DD/<horizon>/. '
                             'Omit for T+0 (writes directly to YYYY-MM-DD/).')
    parser.add_argument('--n-triggers',    type=int,   default=None)
    parser.add_argument('--size-factors',  type=float, nargs='+',
                        default=None,
                        help='Size factor multipliers for arrest thresholds. '
                             '>1.0 = relaxed = larger release; '
                             '<1.0 = tighter = smaller release; '
                             '1.0 = baseline.')
    parser.add_argument('--depth-pcts',    type=int,   nargs='+',
                        default=None)
    parser.add_argument('--mu',            type=float, default=None)
    parser.add_argument('--xi',            type=float, default=None)
    parser.add_argument('--stauchwall-deg',type=float, default=None)
    parser.add_argument('--max-slab-thickness', type=float, default=1.5,
                        help='Maximum slab thickness for skier trigger (m). '
                             'Use 10.0 for natural trigger scenarios. '
                             'Default 1.5 (skier stress negligible below ~1.5m).')
    parser.add_argument('--mode3-scale',   type=float, default=1.5,
                        help='Lateral arrest multiplier on trigger Pi1 '
                             '(default 1.5, must be > 1.0). Higher = '
                             'tighter cross-slope arrest. Mode III '
                             'propagates near shear wave speed and arrests '
                             'more easily than downslope mode II. '
                             'Calibrate against observed flank location.')
    parser.add_argument('--no-propagation', action='store_true',
                        help='Skip BFS propagation, use rectangle fallback')
    parser.add_argument('--restrict-to-release-area', action='store_true',
                        help='Restrict trigger candidates to clusters inside '
                             'the observed release area GeoJSON. Use for '
                             'validation only — in operational mode this '
                             'file does not exist.')
    parser.add_argument('--use-observed-release', action='store_true',
                        help='Use observed Jan 18 release GeoJSON as polygon '
                             'instead of Meloche-derived geometry. '
                             'Validates flow model independently of '
                             'release area estimation.')

    args = parser.parse_args()
    cfg  = ProjectConfig(project_dir=args.project_dir)
    cfg.ensure_dirs()

    # Fill None args from config defaults
    if args.pro_dir is None:
        args.pro_dir = cfg.pro_dir
    if args.zarr_path is None:
        args.zarr_path = cfg.zarr_path
    if args.n_triggers is None:
        args.n_triggers = cfg.n_triggers
    if args.size_factors is None:
        args.size_factors = cfg.size_factors
    if args.depth_pcts is None:
        args.depth_pcts = cfg.depth_percentiles
    if args.mu is None:
        args.mu = cfg.mu
    if args.xi is None:
        args.xi = cfg.xi
    if args.stauchwall_deg is None:
        args.stauchwall_deg = cfg.stauchwall_deg

    if args.step == 'cluster_update':
        from cluster_update import step_cluster_update
        step_cluster_update(cfg, args)
        return

    if args.step in ('zarr_build', 'all'):
        step_zarr_build(cfg, args)

    if args.step in ('analyze', 'all'):
        step_analyze(cfg, args)

    if args.step in ('scenarios', 'all'):
        step_scenarios(cfg, args)


if __name__ == '__main__':
    main()
    
"""
analysis_pipeline.py — Post-SNOWPACK analysis pipeline (Pipeline B).

Steps (run after SNOWPACK finishes and Zarr cache is available):
    zarr_build  Build / resume Zarr cache from .pro files
    analyze     Snapshot comparison, Meloche features, deviation plots
    scenarios   Generate AvaFrame com1DFA release scenario ensemble

Usage:
    python analysis_pipeline.py zarr_build
    python analysis_pipeline.py analyze --snapshot-date 2026-01-17
    python analysis_pipeline.py scenarios --snapshot-date 2026-01-17
    python analysis_pipeline.py all --snapshot-date 2026-01-17

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
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_PRO_DIR   = Path("/home/ron/snowpack/little_prof/output")
DEFAULT_ZARR_PATH = DEFAULT_PRO_DIR / "slope_snowpack.zarr"
DEFAULT_SNAPSHOT  = "2026-01-17"

# Scenario ensemble axes
DEFAULT_N_TRIGGERS     = 5
DEFAULT_SIZE_FACTORS   = [0.70, 0.85, 1.00, 1.15, 1.30]  # P10–P90
DEFAULT_DEPTH_PCTS     = [10, 50, 90]
DEFAULT_DEPTH_SCALES   = {10: 0.75, 50: 1.00, 90: 1.30}   # HS multiplier
DEFAULT_MU             = 0.155    # Voellmy dry friction
DEFAULT_XI             = 1500.0   # Voellmy turbulent friction (m/s²)
DEFAULT_STAUCHWALL_DEG = 28.0


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


def _load_groups(cfg, dem, transform):
    """Load or build release/adjacent/reference cluster group assignments."""
    groups_path = cfg.analysis_dir / "release_zone_groups.json"
    if groups_path.exists():
        with open(str(groups_path)) as f:
            raw = json.load(f)
        return {k: set(v) for k, v in raw.items()}

    # Build from boundary files
    print("Building cluster group assignments from boundaries...")
    from snowpack_analysis import geojson_to_mask
    cluster_map = np.load(str(cfg.analysis_dir / "cluster_map.npy"))

    gj_path = cfg.project_dir / "data/boundaries/avalanche_release_area.geojson"
    release_mask    = geojson_to_mask(gj_path, dem.shape, transform)
    start_zone_mask = kml_to_mask(cfg.start_zone_kml, dem.shape, transform)
    domain_mask     = ~np.isnan(dem)

    groups = assign_cluster_groups(
        cluster_map, release_mask, start_zone_mask, dem, domain_mask)

    # Persist for next run
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


def _load_observed_release_polygon(cfg):
    """
    Load the observed Jan 18 release area GeoJSON and reproject to UTM.
    Returns a single shapely Polygon for use as the release geometry.
    """
    import json
    from shapely.geometry import shape
    from shapely.ops import unary_union
    from pyproj import Transformer

    gj_path = cfg.project_dir / "data/boundaries/avalanche_release_area.geojson"
    if not gj_path.exists():
        raise FileNotFoundError(f"Observed release GeoJSON not found: {gj_path}")

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
    Snapshot comparison and Meloche feature extraction.

    Produces:
        outputs/analysis/release_zone_features_YYYY-MM-DD.csv
        outputs/analysis/meloche_features_YYYY-MM-DD.csv
        outputs/plots/release_zone_comparison_YYYY-MM-DD.png
        outputs/plots/release_zone_deviations_YYYY-MM-DD.png
        outputs/plots/meloche_comparison_YYYY-MM-DD.png
    """
    # Delegate to analyze_release_zone diagnostic script for now.
    # When plot functions are moved into analysis modules, call them directly.
    import subprocess
    script = Path(__file__).resolve().parent / "analyze_release_zone.py"
    cmd    = [sys.executable, str(script),
              '--project-dir', str(cfg.project_dir),
              '--pro-dir',     str(args.pro_dir),
              '--zarr-path',   str(args.zarr_path),
              '--snapshot-date', args.snapshot_date]
    if args.classifier:
        cmd.append('--classifier')
    if args.all_clusters:
        cmd.append('--all-clusters')
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
    groups      = _load_groups(cfg, dem, transform)

    start_zone_mask = kml_to_mask(cfg.start_zone_kml, dem.shape, transform)

    # --- Load pre-computed analysis outputs ---
    feat_csv = cfg.analysis_dir / f"release_zone_features_{args.snapshot_date}.csv"
    mel_csv  = cfg.analysis_dir / f"meloche_features_{args.snapshot_date}.csv"

    if not feat_csv.exists():
        raise FileNotFoundError(
            f"Run 'analyze' step first — missing {feat_csv}")

    snap_features = pd.read_csv(str(feat_csv), index_col=0)
    meloche_df    = pd.read_csv(str(mel_csv),  index_col=0) \
                   if mel_csv.exists() else pd.DataFrame()

    # --- Find trigger locations ---
    # Candidate: clusters in start zone (release + adjacent) sorted by Sk38
    # Restrict trigger candidates to clusters with meaningful driving stress
    # τ_g → 0 near stauchwall (slope ≈ ϕ=27°) blows up the brittle scaling
    MIN_TAU_G   = 50.0   # Pa — below this the scaling law is not applicable
    MIN_SLOPE   = 30.0   # degrees — well above the 27° friction angle
    MAX_SK38    = 0.5    # Sk38 >= 0.5 means terrain is stable — not a valid trigger

    candidate_ids = (groups.get('release', set()) |
                     groups.get('adjacent', set()))
    candidate_ids = [cid for cid in candidate_ids
                     if cid in snap_features.index]

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

    # Optional: restrict to observed release area (validation mode only)
    if args.restrict_to_release_area:
        from snowpack_analysis import geojson_to_mask
        gj_path = cfg.project_dir / 'data/boundaries/avalanche_release_area.geojson'
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
    triggers = (cand_df.dropna(subset=[sk38_col])
                       .nsmallest(args.n_triggers, sk38_col))
    print(f"Trigger clusters: {list(triggers.index)} "
          f"Sk38={list(triggers[sk38_col].round(3))}")

    # --- Output directory ---
    out_dir = cfg.output_dir / "scenarios" / args.snapshot_date
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scenarios").mkdir(exist_ok=True)

    # Write trigger locations GeoJSON
    write_trigger_locations(
        triggers, cluster_map, transform,
        out_dir / "trigger_locations.geojson")

    # --- Slab depth grid for snapshot ---
    full_depth = depth_from_snowpack(cluster_map, ds, args.snapshot_date)

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
        observed_polygon = _load_observed_release_polygon(cfg)
        print("Mode: OBSERVED RELEASE (validation — flow model only)")
    else:
        print("Mode: MELOCHE-DERIVED RELEASE (full chain)")

    # --- Generate scenarios ---
    rows            = []
    weight_map      = {}
    a_ca_list       = []
    median_polygons = []   # (polygon, size_factor) for comparison plot

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
                depth_scale  = DEFAULT_DEPTH_SCALES.get(d_pct, 1.0)
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
                )
                # Augment row with trigger rank and sk38
                row['sk38_rank'] = t_rank + 1
                row['min_sk38']  = float(sk38_vals[t_rank])
                rows.append(row)
                weight_map[scenario_id] = weight

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

    # --- Release comparison plot ---
    import traceback
    print(f"\nGenerating release comparison plot...")
    print(f"  median_polygons collected: {len(median_polygons)}")
    mode_str  = "observed" if args.use_observed_release else "meloche"
    plot_path = out_dir / f"release_comparison_{mode_str}_{args.snapshot_date}_physical_model.png"
    print(f"  Output path: {plot_path}")
    try:
        obs_poly   = _load_observed_release_polygon(cfg)
        poly_pairs = [(p, sf) for p, sf, _ in median_polygons]
        t_labels   = [lbl for _, _, lbl in median_polygons]
        plot_release_comparison(
            meloche_polygons = poly_pairs,
            observed_polygon = obs_poly,
            dem              = dem,
            transform        = transform,
            start_zone_mask  = start_zone_mask,
            trigger_labels   = t_labels,
            out_path         = plot_path,
            title            = (f"Release polygon comparison — {args.snapshot_date} | "
                               f"Little Professor | Jan 18 event\n"
                               f"Blue = Meloche-derived  Red = Observed  "
                               f"Green = Start zone"),
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
        choices=['zarr_build', 'analyze', 'scenarios', 'all'],
        help="Pipeline step to run")

    # Common
    parser.add_argument('--project-dir', type=Path, default=Path('.'))
    parser.add_argument('--pro-dir',     type=Path, default=DEFAULT_PRO_DIR)
    parser.add_argument('--zarr-path',   type=Path, default=DEFAULT_ZARR_PATH)
    parser.add_argument('--snapshot-date', default=DEFAULT_SNAPSHOT,
                        help='SNOWPACK snapshot date YYYY-MM-DD')

    # zarr_build
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--workers',    type=int, default=31)
    parser.add_argument('--max-layers', type=int, default=338)

    # analyze
    parser.add_argument('--classifier', action='store_true')
    parser.add_argument('--all-clusters', action='store_true',
                        help='Extract features for ALL start zone clusters. '
                             'Produces all_start_zone_features and '
                             'meloche_features_all CSVs for the '
                             'probabilistic boundary model.')

    # scenarios
    parser.add_argument('--n-triggers',    type=int,   default=DEFAULT_N_TRIGGERS)
    parser.add_argument('--size-factors',  type=float, nargs='+',
                        default=DEFAULT_SIZE_FACTORS,
                        help='Pi1 threshold multipliers. '
                             '>1.0 = stricter = smaller release; '
                             '<1.0 = looser = larger release; '
                             '1.0 = baseline (all terrain as '
                             'unstable as trigger cluster).')
    parser.add_argument('--depth-pcts',    type=int,   nargs='+',
                        default=DEFAULT_DEPTH_PCTS)
    parser.add_argument('--mu',            type=float, default=DEFAULT_MU)
    parser.add_argument('--xi',            type=float, default=DEFAULT_XI)
    parser.add_argument('--stauchwall-deg',type=float, default=DEFAULT_STAUCHWALL_DEG)
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

    if args.step in ('zarr_build', 'all'):
        step_zarr_build(cfg, args)

    if args.step in ('analyze', 'all'):
        step_analyze(cfg, args)

    if args.step in ('scenarios', 'all'):
        step_scenarios(cfg, args)


if __name__ == '__main__':
    main()
    
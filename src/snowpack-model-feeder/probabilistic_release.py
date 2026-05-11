"""
probabilistic_release.py — Probabilistic release boundary + comparison plot

Uses the logistic boundary model from fit_boundary_model.py to compute
P(arrest) at each cluster-pair boundary, then grows a probabilistic release
polygon by flood-filling where P(arrest) < propagate_threshold.

Produces a release comparison plot (_probability_model) alongside the
physics model plot (_physical_model) from analysis_pipeline.py.

Usage:
  python src/snowpack-model-feeder/probabilistic_release.py \
      --snapshot-date 2026-01-18 \
      --trigger-cid 3178 \
      --model outputs/models/boundary_model_2026-01-18.pkl \
      --propagate-threshold 0.4
"""

import argparse, json, pickle, sys
from collections import deque
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))


# -----------------------------------------------------------------------
# Probabilistic flood-fill
# -----------------------------------------------------------------------

def probabilistic_release_zone(
        trigger_cid: int,
        pipeline,
        feature_cols: list,
        meloche_df: pd.DataFrame,
        snap_features: pd.DataFrame,
        cluster_map: np.ndarray,
        slope_grid: np.ndarray,
        transform,
        start_zone_mask=None,
        propagate_threshold: float = 0.4,
        k_neighbours: int = 8,
        max_clusters: int = 500) -> tuple:
    """
    BFS flood-fill using P(arrest) from logistic model.
    Propagates where P(arrest) < propagate_threshold.

    Returns: polygon, failed_set, {cid: p_arrest}
    """
    from sklearn.neighbors import NearestNeighbors
    from shapely.ops import unary_union
    import rasterio.features

    cids = np.array([c for c in np.unique(cluster_map) if c > 0])
    pxs  = np.array([np.argwhere(cluster_map == c).mean(axis=0)
                     for c in cids])
    k_actual = min(k_neighbours + 1, len(cids))
    nbrs     = NearestNeighbors(n_neighbors=k_actual).fit(pxs)
    _, idxs  = nbrs.kneighbors(pxs)
    neighbours = {int(cids[i]): [int(cids[j]) for j in idxs[i][1:]]
                  for i in range(len(cids))}

    def _get(df, cid, col):
        if df is None or df.empty or cid not in df.index: return np.nan
        if col not in df.columns: return np.nan
        v = df.loc[cid, col]
        if isinstance(v, pd.DataFrame): v = v.iloc[0][col]
        elif isinstance(v, pd.Series):  v = v.iloc[0]
        try:    return float(v)
        except: return np.nan

    def _slope(cid):
        px = np.argwhere(cluster_map == cid)
        if len(px) == 0: return np.nan
        return float(slope_grid[px[:, 0], px[:, 1]].mean())

    def _rel_change(va, vb):
        if np.isnan(va) or np.isnan(vb) or va == 0: return np.nan
        return abs(vb - va) / abs(va)

    def _pair_features(cid_a, cid_b):
        lam_a  = _get(meloche_df,    cid_a, 'Lambda')
        lam_b  = _get(meloche_df,    cid_b, 'Lambda')
        h_a    = _get(snap_features, cid_a, 'slab_thickness')
        h_b    = _get(snap_features, cid_b, 'slab_thickness')
        tp_a   = _get(snap_features, cid_a, 'wl_shear_strength')
        tp_b   = _get(snap_features, cid_b, 'wl_shear_strength')
        tg_a   = _get(meloche_df,    cid_a, 'tau_g')
        tg_b   = _get(meloche_df,    cid_b, 'tau_g')
        th_a   = _get(meloche_df,    cid_a, 'theta')
        th_b   = _get(meloche_df,    cid_b, 'theta')
        pi1_a  = _get(meloche_df,    cid_a, 'Pi1_elastic')
        pi1_b  = _get(meloche_df,    cid_b, 'Pi1_elastic')
        sl_a   = _slope(cid_a)
        sl_b   = _slope(cid_b)
        def _signed(va, vb):
            if np.isnan(va) or np.isnan(vb) or va == 0: return np.nan
            return (vb - va) / abs(va)
        return {
            # Robust unsigned gradients — primary operational features
            'delta_lambda_rel':  _rel_change(lam_a, lam_b),
            'delta_h_rel':       _rel_change(h_a,   h_b),
            'delta_tau_p_rel':   _rel_change(tp_a,  tp_b),
            'slope_mean':        np.nanmean([sl_a, sl_b]),
            'tau_g_min':         (min(tg_a, tg_b)
                                  if not (np.isnan(tg_a) or np.isnan(tg_b))
                                  else np.nan),
            # Signed features — interpretability, may be NaN if neighbor
            # not in meloche_df / snap_features
            'dlambda_signed':    _signed(lam_a, lam_b),
            'dh_signed':         _signed(h_a,   h_b),
            'dtau_p_signed':     _signed(tp_a,  tp_b),
            'tau_g_nbr':         tg_b,
            'lambda_nbr':        lam_b,
        }

    failed       = {trigger_cid}
    frontier     = deque([trigger_cid])
    visited      = {trigger_cid}
    arrest_probs = {}

    while frontier and len(failed) < max_clusters:
        current = frontier.popleft()
        for nbr in neighbours.get(current, []):
            if nbr in visited: continue
            visited.add(nbr)

            if start_zone_mask is not None:
                px = np.argwhere(cluster_map == nbr)
                if len(px) == 0: continue
                r, c = int(px.mean(axis=0)[0]), int(px.mean(axis=0)[1])
                if not start_zone_mask[r, c]: continue

            feats = _pair_features(current, nbr)
            feat_vals = [feats.get(f, np.nan) for f in feature_cols]

            nan_feats = [f for f, v in zip(feature_cols, feat_vals)
                         if np.isnan(v)]
            if nan_feats:
                # Log first few NaN cases for diagnosis
                if len(failed) <= 3:
                    print(f"    NaN features for ({current}→{nbr}): "
                          f"{nan_feats}")
                # Partial prediction: use available features
                avail_idx  = [i for i, v in enumerate(feat_vals)
                              if not np.isnan(v)]
                if len(avail_idx) >= 2:
                    X_partial = np.array(
                        [feat_vals[i] for i in avail_idx]).reshape(1, -1)
                    # Use only available features with a fresh LR on subset
                    # Fallback: mean-impute missing features
                    feat_filled = feat_vals.copy()
                    scaler = pipeline.named_steps['scaler']
                    for i, v in enumerate(feat_filled):
                        if np.isnan(v):
                            feat_filled[i] = float(scaler.mean_[i])
                    X        = np.array(feat_filled).reshape(1, -1)
                    p_arrest = float(pipeline.predict_proba(X)[0, 1])
                else:
                    p_arrest = 0.8  # too few features — conservative arrest
            else:
                X        = np.array(feat_vals).reshape(1, -1)
                p_arrest = float(pipeline.predict_proba(X)[0, 1])

            arrest_probs[nbr] = p_arrest
            if p_arrest < propagate_threshold:
                failed.add(nbr)
                frontier.append(nbr)

    # Diagnostic: show P(arrest) distribution
    if arrest_probs:
        vals = sorted(arrest_probs.values())
        print(f"  P(arrest) distribution across {len(vals)} tested clusters:")
        for pct, label in [(10,'P10'), (25,'P25'), (50,'P50'),
                           (75,'P75'), (90,'P90')]:
            idx = int(pct / 100 * len(vals))
            print(f"    {label}: {vals[min(idx, len(vals)-1)]:.3f}")
        print(f"    Propagated ({len(failed)-1} clusters): "
              f"P(arrest) < {propagate_threshold}")
        print(f"    Arrested   ({len(vals)-(len(failed)-1)} clusters): "
              f"P(arrest) >= {propagate_threshold}")
    if arrest_probs:
        probs = list(arrest_probs.values())
        print(f"  P(arrest) distribution across {len(probs)} tested clusters:")
        print(f"    min={min(probs):.3f}  p10={np.percentile(probs,10):.3f}  "
              f"p25={np.percentile(probs,25):.3f}  median={np.median(probs):.3f}  "
              f"p75={np.percentile(probs,75):.3f}  max={max(probs):.3f}")
        n_below = sum(1 for p in probs if p < propagate_threshold)
        print(f"    Clusters with P(arrest) < {propagate_threshold}: {n_below}/{len(probs)}")
    print(f"  Probabilistic release: {len(failed)} clusters  "
          f"(P(arrest) < {propagate_threshold})")

    if len(failed) < 2:
        return None, failed, arrest_probs

    mask   = np.isin(cluster_map, list(failed)).astype(np.uint8)
    shapes = list(rasterio.features.shapes(
        mask, mask=mask.astype(bool), transform=transform))
    if not shapes:
        return None, failed, arrest_probs

    polys   = [__import__('shapely.geometry', fromlist=['shape']).shape(s)
               for s, v in shapes if v == 1]
    polygon = __import__('shapely.ops', fromlist=['unary_union']).unary_union(polys)
    if polygon.geom_type == 'MultiPolygon':
        polygon = max(polygon.geoms, key=lambda p: p.area)

    return polygon, failed, arrest_probs


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshot-date',       required=True)
    ap.add_argument('--trigger-cid',         type=int, required=True)
    ap.add_argument('--model',               required=True)
    ap.add_argument('--propagate-threshold', type=float, default=0.6,
                    help='P(arrest) threshold. With class_weight=balanced '
                         'the model outputs higher probabilities — typical '
                         'useful range is 0.5-0.75. Run with --propagate-threshold 0.99 '
                         'first to see the full P(arrest) distribution.')
    ap.add_argument('--dem',         default='outputs/resampled_1m/dem_1m.tif')
    ap.add_argument('--cluster-map', default='outputs/analysis/cluster_map.npy')
    ap.add_argument('--features-dir',default='outputs/analysis')
    ap.add_argument('--start-zone-kml',
                    default='data/boundaries/Litte_prof_start_zone.kml')
    ap.add_argument('--release-geojson',
                    default='data/boundaries/avalanche_release_area.geojson')
    ap.add_argument('--out-dir', default='outputs/scenarios/probabilistic')
    args = ap.parse_args()

    import rasterio
    from snowpack_analysis import geojson_to_mask
    from release_geometry import (compute_slope_aspect, plot_release_comparison)
    from snowpack_io import load_boundaries
    from snowpack_analysis import geojson_to_mask as _geojson_to_mask

    date    = args.snapshot_date
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_dir = Path(args.features_dir)

    print(f"Probabilistic release — {date}  cid={args.trigger_cid}")

    with rasterio.open(args.dem) as src:
        dem       = src.read(1).astype(float)
        dem[dem == src.nodata] = np.nan
        transform = src.transform

    cluster_map     = np.load(args.cluster_map)
    slope_grid, _   = compute_slope_aspect(dem)
    # Use full start zone coverage if available (--all-clusters was run)
    # Falls back to release+adjacent groups if not
    _snap_all   = feat_dir / f'all_start_zone_features_{date}.csv'
    _snap_grp   = feat_dir / f'release_zone_features_{date}.csv'
    snap_features = pd.read_csv(
        _snap_all if _snap_all.exists() else _snap_grp, index_col=0)
    print(f"  snap_features: {len(snap_features)} clusters "
          f"({'full SZ' if _snap_all.exists() else 'groups only'})")

    _mel_all    = feat_dir / f'meloche_features_all_{date}.csv'
    _mel_grp    = feat_dir / f'meloche_features_{date}.csv'
    meloche_df  = pd.read_csv(
        _mel_all if _mel_all.exists() else _mel_grp, index_col=0)
    print(f"  meloche_df:    {len(meloche_df)} clusters "
          f"({'full SZ' if _mel_all.exists() else 'groups only'})")
    # load_boundaries takes project_dir and returns a dict of masks
    boundaries      = load_boundaries(Path('.'))
    start_zone_mask = boundaries.get('start_zone_mask')

    with open(args.model, 'rb') as f:
        pipeline = pickle.load(f)

    lr = pipeline.named_steps['lr']
    feature_cols = (list(lr.feature_names_in_)
                    if hasattr(lr, 'feature_names_in_')
                    else ['delta_lambda_rel', 'delta_h_rel',
                          'delta_tau_p_rel',  'delta_pi1_rel',
                          'tau_g_min',        'theta_mean', 'slope_mean'])

    polygon, failed, arrest_probs = probabilistic_release_zone(
        trigger_cid         = args.trigger_cid,
        pipeline            = pipeline,
        feature_cols        = feature_cols,
        meloche_df          = meloche_df,
        snap_features       = snap_features,
        cluster_map         = cluster_map,
        slope_grid          = slope_grid,
        transform           = transform,
        start_zone_mask     = start_zone_mask,
        propagate_threshold = args.propagate_threshold)

    if polygon is None:
        print("No polygon generated.")
        return

    print(f"  Release area: {polygon.area:.0f} m²")

    # --- Save GeoJSON ---
    gj_path = out_dir / f'release_probabilistic_{date}_cid{args.trigger_cid}.geojson'
    gj = {"type": "FeatureCollection", "features": [{
        "type": "Feature",
        "properties": {
            "trigger_cid":          args.trigger_cid,
            "snapshot_date":        date,
            "method":               "probabilistic_logistic",
            "propagate_threshold":  args.propagate_threshold,
            "area_m2":              polygon.area,
            "n_clusters":           len(failed),
        },
        "geometry": polygon.__geo_interface__
    }]}
    with open(gj_path, 'w') as f:
        json.dump(gj, f, indent=2)
    print(f"  GeoJSON: {gj_path}")

    # --- Save arrest probability map ---
    prob_path = out_dir / f'arrest_probs_{date}_cid{args.trigger_cid}.csv'
    pd.Series(arrest_probs, name='p_arrest').to_csv(prob_path)
    print(f"  Arrest probs: {prob_path}")

    # --- Comparison plot (same as physical model) ---
    obs_polygon = None
    gj_obs = Path(args.release_geojson)
    if gj_obs.exists():
        obs_polygon = geojson_to_mask(str(gj_obs), dem.shape, transform)
        # Load as polygon rather than mask
        import json as _json
        from shapely.geometry import shape
        from shapely.ops import unary_union
        from pyproj import Transformer
        with open(str(gj_obs)) as f:
            gj_data = _json.load(f)
        t = Transformer.from_crs('EPSG:4326', 'EPSG:32613', always_xy=True)
        def _reproj(poly):
            from shapely.geometry import Polygon, MultiPolygon
            def ring(r): return [t.transform(x,y) for x,y in r.coords]
            if poly.geom_type == 'Polygon':
                return Polygon(ring(poly.exterior),
                               [ring(i) for i in poly.interiors])
            return MultiPolygon([_reproj(p) for p in poly.geoms])
        obs_polys   = [_reproj(shape(f['geometry'])) for f in gj_data['features']]
        obs_polygon = unary_union(obs_polys)

    label = (f"T1 cid={args.trigger_cid}  "
             f"P_thresh={args.propagate_threshold}  "
             f"({polygon.area:.0f} m²)")

    plot_path = out_dir / f"release_comparison_{date}_probability_model.png"
    plot_release_comparison(
        meloche_polygons = [(polygon, 1.0)],
        observed_polygon = obs_polygon,
        dem              = dem,
        transform        = transform,
        start_zone_mask  = start_zone_mask,
        trigger_labels   = [label],
        out_path         = plot_path,
        title            = (f"Release polygon comparison — {date} | "
                            f"Little Professor | Jan 18 event\n"
                            f"Probabilistic model  |  "
                            f"P(arrest) threshold = {args.propagate_threshold}"))
    print(f"  Plot: {plot_path}")


if __name__ == '__main__':
    main()
    
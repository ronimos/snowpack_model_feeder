"""
fit_boundary_model.py — Probabilistic release boundary model

Extracts labeled cluster-pair transitions from a known event and fits
a logistic regression model:

  P(arrest | features) = logistic(Xβ)

Boundary transitions (arrest=1): pairs where one cluster is inside the
observed release area and its neighbour is outside.
Interior transitions (arrest=0): pairs where both clusters are inside.

Feature set (per cluster pair):
  Core snowpack gradients:
    delta_lambda_rel   |ΔΛ/Λ_current|           — slab elastic length change
    delta_h_rel        |Δh/h_current|            — slab burial depth change
    delta_tau_p_rel    |Δτ_p/τ_p_current|       — WL shear strength change
    delta_pi1_rel      |ΔΠ₁/Π₁_current|        — composite stability change
  Absolute values at transition:
    tau_g_min          min(τ_g_a, τ_g_b)        — weakest driving stress
    theta_mean         mean(θ_a, θ_b)            — WL heterogeneity
    slope_mean         mean slope of pair        — terrain
    lambda_mean        mean Λ of pair            — slab stiffness level

As events accumulate, rerun with --append-pairs to grow training dataset.

Usage:
  python src/snowpack-model-feeder/fit_boundary_model.py \
      --snapshot-date 2026-01-17 \
      --event-label "jan18_little_prof"
"""

import argparse, json, pickle, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))


# -----------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------

def extract_pair_features(
        release_cids: set,
        meloche_df: pd.DataFrame,
        snap_features: pd.DataFrame,
        cluster_map: np.ndarray,
        slope_grid: np.ndarray,
        k_neighbours: int = 8) -> pd.DataFrame:
    """
    For every cluster pair (current, neighbour) where at least one cluster
    is inside the observed release area, extract features and label.

    arrest=1 — boundary (one inside, one outside)
    arrest=0 — interior (both inside)
    """
    from sklearn.neighbors import NearestNeighbors

    cids = np.array([c for c in np.unique(cluster_map) if c > 0])
    pxs  = np.array([np.argwhere(cluster_map == c).mean(axis=0)
                     for c in cids])
    k_actual = min(k_neighbours + 1, len(cids))
    nbrs     = NearestNeighbors(n_neighbors=k_actual).fit(pxs)
    _, idxs  = nbrs.kneighbors(pxs)
    neighbours = {int(cids[i]): [int(cids[j]) for j in idxs[i][1:]]
                  for i in range(len(cids))}

    def _get(df, cid, col, scale=1.0):
        if df is None or df.empty or cid not in df.index: return np.nan
        if col not in df.columns: return np.nan
        v = df.loc[cid, col]
        if isinstance(v, pd.DataFrame): v = v.iloc[0][col]
        elif isinstance(v, pd.Series):  v = v.iloc[0]
        try:    return float(v) * scale
        except: return np.nan

    def _slope(cid):
        px = np.argwhere(cluster_map == cid)
        if len(px) == 0: return np.nan
        return float(slope_grid[px[:, 0], px[:, 1]].mean())

    def _rel_change(va, vb):
        """Unsigned relative change."""
        if np.isnan(va) or np.isnan(vb) or va == 0: return np.nan
        return abs(vb - va) / abs(va)

    def _signed_rel(va, vb):
        """Signed relative change: positive = neighbor is larger."""
        if np.isnan(va) or np.isnan(vb) or va == 0: return np.nan
        return (vb - va) / abs(va)

    rows, seen = [], set()

    for cid in cids:
        if int(cid) not in release_cids:
            continue
        for nbr in neighbours.get(int(cid), []):
            pair_key = tuple(sorted([int(cid), int(nbr)]))
            if pair_key in seen: continue
            seen.add(pair_key)

            arrest = 0 if int(nbr) in release_cids else 1

            # Snowpack properties at each cluster
            lam_a  = _get(meloche_df,   cid, 'Lambda')
            lam_b  = _get(meloche_df,   nbr, 'Lambda')
            h_a    = _get(snap_features, cid, 'slab_thickness')
            h_b    = _get(snap_features, nbr, 'slab_thickness')
            tp_a   = _get(snap_features, cid, 'wl_shear_strength')  # kPa
            tp_b   = _get(snap_features, nbr, 'wl_shear_strength')
            tg_a   = _get(meloche_df,   cid, 'tau_g')               # Pa
            tg_b   = _get(meloche_df,   nbr, 'tau_g')
            th_a   = _get(meloche_df,   cid, 'theta')               # Pa/m
            th_b   = _get(meloche_df,   nbr, 'theta')
            pi1_a  = _get(meloche_df,   cid, 'Pi1_elastic')
            pi1_b  = _get(meloche_df,   nbr, 'Pi1_elastic')
            sl_a   = _slope(cid)
            sl_b   = _slope(nbr)

            rows.append({
                'cid_a': int(cid), 'cid_b': int(nbr),
                'arrest': arrest,
                # --- Unsigned gradients (|Δx/x|) ---
                # Primary arrest signal: any sharp discontinuity
                'delta_lambda_rel': _rel_change(lam_a, lam_b),
                'delta_h_rel':      _rel_change(h_a,   h_b),
                'delta_tau_p_rel':  _rel_change(tp_a,  tp_b),
                # --- Signed gradients (Δx/x, positive = neighbor larger) ---
                # Arrest direction matters: moving into stiffer/thicker
                # slab vs softer/thinner have different physics
                'dlambda_signed':   _signed_rel(lam_a, lam_b),
                'dh_signed':        _signed_rel(h_a,   h_b),
                'dtau_p_signed':    _signed_rel(tp_a,  tp_b),
                # --- Absolute values at transition ---
                'tau_g_min':    (min(tg_a, tg_b)
                                 if not (np.isnan(tg_a) or np.isnan(tg_b))
                                 else np.nan),
                'tau_g_nbr':    tg_b,   # driving stress at candidate
                'lambda_nbr':   lam_b,  # slab stiffness at candidate
                'h_nbr':        h_b,    # burial depth at candidate
                'slope_mean':   np.nanmean([sl_a, sl_b]),
                'theta_mean':   np.nanmean([th_a, th_b]),
            })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------
# Model fitting
# -----------------------------------------------------------------------

def fit_model(pairs_df: pd.DataFrame,
              feature_cols: list,
              event_label: str = '') -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, classification_report
    from sklearn.pipeline import Pipeline

    df = pairs_df[feature_cols + ['arrest']].dropna()
    print(f"  Training pairs: {len(df)}  "
          f"(boundary={df.arrest.sum()}  "
          f"interior={len(df)-df.arrest.sum()})")

    X = df[feature_cols].values
    y = df['arrest'].values

    from sklearn.calibration import CalibratedClassifierCV
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lr',     CalibratedClassifierCV(
                       LogisticRegression(class_weight='balanced',
                                          max_iter=1000, random_state=42),
                       method='isotonic', cv=5))
    ])
    # CalibratedClassifierCV wraps the LR so P(arrest) is properly
    # calibrated: P=0.5 means 50% of pairs with those features are
    # actually boundaries. This makes the threshold directly interpretable.
    pipe.fit(X, y)
    # Store feature names for later retrieval
    pipe.named_steps['lr'].feature_names_in_ = np.array(feature_cols)

    y_proba = pipe.predict_proba(X)[:, 1]
    y_pred  = pipe.predict(X)
    auc     = roc_auc_score(y, y_proba)

    print(f"  ROC-AUC: {auc:.3f}")
    print(classification_report(y, y_pred,
                                 target_names=['interior', 'boundary']))

    # Extract base LR from calibrated wrapper for coefficient inspection
    calibrated = pipe.named_steps['lr']
    scaler     = pipe.named_steps['scaler']
    # Average coefficients across CV folds
    base_lrs  = [est.estimator for est in calibrated.calibrated_classifiers_]
    mean_coef = np.mean([lr.coef_[0] for lr in base_lrs], axis=0)
    raw_coefs = {f: float(mean_coef[i] / scaler.scale_[i])
                 for i, f in enumerate(feature_cols)}

    print("  Unscaled coefficients (mean across CV folds):")
    for f, c in sorted(raw_coefs.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"    {f:25s}: {c:+.4f}")

    # Report calibrated P(arrest) distribution on training data
    print("\n  Calibrated P(arrest) percentiles (training data):")
    for pct in [10, 25, 50, 75, 90]:
        idx = int(pct / 100 * len(y_proba))
        print(f"    P{pct:2d}: {sorted(y_proba)[idx]:.3f}")

    return {
        'pipeline':       pipe,
        'feature_cols':   feature_cols,
        'auc':            float(auc),
        'coefs_unscaled': raw_coefs,
        'n_pairs':        len(df),
        'n_boundary':     int(df.arrest.sum()),
        'n_interior':     int(len(df) - df.arrest.sum()),
        'event_label':    event_label,
    }


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshot-date', required=True)
    ap.add_argument('--release-geojson',
                    default='data/boundaries/avalanche_release_area.geojson')
    ap.add_argument('--cluster-map',
                    default='outputs/analysis/cluster_map.npy')
    ap.add_argument('--dem',  default='outputs/resampled_1m/dem_1m.tif')
    ap.add_argument('--features-dir', default='outputs/analysis')
    ap.add_argument('--out-dir',      default='outputs/models')
    ap.add_argument('--event-label',  default='')
    args = ap.parse_args()

    import rasterio
    from snowpack_analysis import geojson_to_mask
    from release_geometry import compute_slope_aspect

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_dir = Path(args.features_dir)
    date = args.snapshot_date

    with rasterio.open(args.dem) as src:
        dem       = src.read(1).astype(float)
        dem[dem == src.nodata] = np.nan
        transform = src.transform

    cluster_map   = np.load(args.cluster_map)
    slope_grid, _ = compute_slope_aspect(dem)
    _snap_all   = feat_dir / f'all_start_zone_features_{date}.csv'
    snap_features = pd.read_csv(
        _snap_all if _snap_all.exists()
        else feat_dir / f'release_zone_features_{date}.csv', index_col=0)
    _mel_all    = feat_dir / f'meloche_features_all_{date}.csv'
    meloche_df  = pd.read_csv(
        _mel_all if _mel_all.exists()
        else feat_dir / f'meloche_features_{date}.csv', index_col=0)
    print(f"  snap_features: {len(snap_features)} clusters")
    print(f"  meloche_df:    {len(meloche_df)} clusters")

    release_mask = geojson_to_mask(
        args.release_geojson, dem.shape, transform)
    release_cids = set(int(c) for c in np.unique(cluster_map[release_mask])
                       if c > 0)
    print(f"Release area clusters: {len(release_cids)}")

    print("Extracting pair features...")
    pairs_df = extract_pair_features(
        release_cids, meloche_df, snap_features, cluster_map, slope_grid)
    print(f"  Total pairs: {len(pairs_df)}  "
          f"(boundary={pairs_df.arrest.sum()}  "
          f"interior={(pairs_df.arrest==0).sum()})")

    pairs_df.to_csv(out_dir / f'boundary_pairs_{date}.csv', index=False)

    # Feature sets
    # NOTE: signed/absolute features require BOTH clusters to be in their
    # respective dataframes. During inference, neighbor clusters are often
    # not in meloche_df or snap_features (only release+adjacent groups are
    # stored). Use only robust unsigned gradients for the operational model.
    # Signed features are informative for interpretation but not reliable
    # for inference until all start-zone clusters have feature coverage.
    feature_cols_robust = [
        'delta_lambda_rel',   # |ΔΛ/Λ| — both from meloche_df
        'delta_h_rel',        # |Δh/h|  — both from snap_features
        'delta_tau_p_rel',    # |Δτ_p/τ_p| — both from snap_features
        'slope_mean',         # from DEM — always available
        'tau_g_min',          # from meloche_df — NaN if neighbor missing
    ]
    # Extended: signed features for interpretability (training only)
    feature_cols_extended = feature_cols_robust + [
        'dlambda_signed', 'dh_signed', 'dtau_p_signed',
        'tau_g_nbr', 'lambda_nbr',
    ]
    feature_cols = feature_cols_robust  # robust set for operational use

    print("\nFitting logistic boundary model (full feature set)...")
    label  = args.event_label or f'jan18_{date}'
    result = fit_model(pairs_df, feature_cols, event_label=label)

    # Physical check
    print("\nPhysical calibration check:")
    print("  LAMBDA_JUMP_FACTOR=0.5     → 'delta_lambda_rel' P=0.5 threshold")
    print("  THICKNESS_JUMP_FACTOR=0.25 → 'delta_h_rel' P=0.5 threshold")
    print("\nNote: negative signed coefficients physically expected:")
    print("  dlambda_signed < 0 → softer slab ahead arrests crack")
    print("  dtau_p_signed  > 0 → stronger WL ahead arrests crack")

    stem = out_dir / f'boundary_model_{date}'
    with open(f'{stem}.pkl', 'wb') as f:
        pickle.dump(result['pipeline'], f)
    meta = {k: v for k, v in result.items() if k != 'pipeline'}
    meta['snapshot_date'] = date
    with open(f'{stem}.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nModel saved: {stem}.pkl / .json")

if __name__ == '__main__':
    main()
    
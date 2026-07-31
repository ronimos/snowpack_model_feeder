"""
Compare SNOWPACK indices between:
  1. Release zone  — clusters inside the observed Jan 18 release area GeoJSON
  2. Adjacent slope — clusters inside start zone KML but outside release zone
  3. Reference      — clusters inside survey domain but outside start zone,
                      matched by elevation/aspect to the release zone

For each group, extracts:
  - Slab properties:  mean density, hardness, grain size above weak layer
  - Weak layer props: shear strength, grain size, burial depth, thickness
  - Stability:        min Sk38, SSI, Sn38 (buried >min_depth cm)
  - HS

Also computes weak layer strength gradient (theta) between neighboring
clusters for spatial heterogeneity analysis.

Optionally trains a Random Forest classifier (release vs adjacent) at each
timestep and plots feature importance evolution.

Usage:
  python analyze_release_zone.py
  python analyze_release_zone.py --min-depth 20 --end-date 2026-02-01 --classifier
"""

import argparse
import json
import sys
import warnings
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
from snowpack_io import load_dataset, kml_to_mask
from snowpack_analysis import (
    geojson_to_mask, assign_cluster_groups,
    split_wl_slab, profile_features,
    extract_group_timeseries, compute_meloche_features,
    EVENT_DATE, WL_TYPES, SLAB_TYPES,
    GROUP_COLORS, GROUP_LABELS,
)




# -----------------------------------------------------------------------
# Classifier
# -----------------------------------------------------------------------

def run_classifier(ds, groups: dict, location_names,
                    min_depth_cm: float, end_ts: pd.Timestamp) -> pd.DataFrame:
    """
    Train RF classifier (release=1 vs adjacent=0) at each daily timestep
    using one sample per cluster — not group medians.

    Returns DataFrame of feature importances over time, indexed by date.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    print("  Extracting per-cluster features for classifier...")
    rel_pc = extract_group_timeseries(
        ds, groups['release'], location_names,
        min_depth_cm, end_ts, per_cluster=True)
    adj_pc = extract_group_timeseries(
        ds, groups['adjacent'], location_names,
        min_depth_cm, end_ts, per_cluster=True)

    if rel_pc.empty or adj_pc.empty:
        print("  Not enough per-cluster data for classifier")
        return pd.DataFrame()

    feature_cols = [c for c in rel_pc.columns
                    if c in adj_pc.columns]

    # Get common timestamps
    rel_times = rel_pc.index.get_level_values('time').unique()
    adj_times = adj_pc.index.get_level_values('time').unique()
    common_times = sorted(set(rel_times) & set(adj_times))

    importance_rows = []
    for ts in common_times:
        try:
            r_rows = rel_pc.loc[ts][feature_cols].dropna()
            a_rows = adj_pc.loc[ts][feature_cols].dropna()
        except KeyError:
            continue

        if len(r_rows) < 2 or len(a_rows) < 2:
            continue

        X = pd.concat([r_rows, a_rows]).values
        y = np.array([1] * len(r_rows) + [0] * len(a_rows))

        if np.any(np.isnan(X)):
            # Fill NaN with column median
            col_meds = np.nanmedian(X, axis=0)
            for j in range(X.shape[1]):
                X[np.isnan(X[:, j]), j] = col_meds[j]

        sc   = StandardScaler()
        X_sc = sc.fit_transform(X)
        rf   = RandomForestClassifier(
            n_estimators=200, class_weight='balanced',
            max_features='sqrt', random_state=42)
        rf.fit(X_sc, y)

        row = {'time': ts,
               'n_release': len(r_rows),
               'n_adjacent': len(a_rows)}
        row.update(dict(zip(feature_cols, rf.feature_importances_)))
        importance_rows.append(row)
        print(f"    {ts.date()}: {len(r_rows)} release, "
              f"{len(a_rows)} adjacent, "
              f"top feat: {feature_cols[np.argmax(rf.feature_importances_)]}")

    if not importance_rows:
        return pd.DataFrame()
    return pd.DataFrame(importance_rows).set_index('time')


# -----------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------

def plot_snapshot_comparison(snap_data: dict, out_path: Path,
                              min_depth: float, snap_ts: pd.Timestamp):
    """
    Box/violin plots comparing release zone vs adjacent vs reference
    at a single timestamp. One box per group per variable.
    """
    features = [
        ('min_sk38',           'Min Sk38',               1.0,  True),
        ('min_ssi',            'Min SSI',                 1.5,  True),
        ('min_sn38',           'Min Sn38',                1.0,  True),
        ('slab_thickness',     'Slab thickness (m)',      None, False),
        ('slab_density',       'Slab density (kg/m³)',    None, False),
        ('slab_hand_hardness', 'Slab hardness',           None, False),
        ('wl_shear_strength',  'WL shear strength (Pa)',  None, False),
        ('wl_burial_depth',    'WL burial depth (m)',     None, False),
        ('wl_grain_size',      'WL grain size (mm)',      None, False),
        ('n_crust_layers',     'N crust layers',          None, False),
        ('crust_thickness',    'Crust thickness (m)',     None, False),
        ('crust_top_depth',    'Crust top depth (m)',     None, False),
    ]

    ncols = 3
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.5*nrows))
    axes = axes.ravel()

    groups_order = ['release', 'adjacent', 'reference']
    positions    = [1, 2, 3]
    colors       = [GROUP_COLORS[g] for g in groups_order]

    for ax, (col, label, threshold, invert) in zip(axes, features):
        data_by_group = []
        labels        = []
        for grp in groups_order:
            df = snap_data.get(grp, pd.DataFrame())
            if df.empty or col not in df.columns:
                data_by_group.append([])
            else:
                vals = df[col].dropna().values
                data_by_group.append(vals)
            labels.append(GROUP_LABELS[grp].split(' ')[0])  # short label

        # Clip outliers at 1.5*IQR before plotting for cleaner display
        def clip_outliers(vals):
            if len(vals) < 4:
                return vals
            q1, q3 = np.percentile(vals, [25, 75])
            iqr = q3 - q1
            return vals[(vals >= q1 - 1.5*iqr) & (vals <= q3 + 1.5*iqr)]

        clipped = [clip_outliers(np.array(d)) if len(d) > 3 else d
                   for d in data_by_group]

        # notch=True shows 95% CI around median
        # bootstrap=1000 computes CI by resampling
        try:
            bp = ax.boxplot(clipped, positions=positions,
                            patch_artist=True, widths=0.6,
                            notch=True, bootstrap=1000,
                            showfliers=False,
                            medianprops=dict(color='black', linewidth=2))
        except Exception:
            # notch can fail with small n — fall back
            bp = ax.boxplot(clipped, positions=positions,
                            patch_artist=True, widths=0.6,
                            notch=False, showfliers=False,
                            medianprops=dict(color='black', linewidth=2))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        if threshold is not None:
            ax.axhline(threshold, color='black', linewidth=0.8,
                       linestyle='--', alpha=0.5)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(True, axis='y', alpha=0.25)
        if invert:
            ax.invert_yaxis()

    for ax in axes[len(features):]:
        ax.set_visible(False)

    fig.suptitle(
        'Release zone vs adjacent vs reference - ' + str(snap_ts.date()) + ' | '
        'Little Professor | Jan 18 | buried >{:.0f}cm'.format(min_depth),
        fontsize=11)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"Comparison plot saved: {out_path}")


def plot_snapshot_deviations(snap_data: dict, out_path: Path,
                              snap_ts, min_depth: float):
    """
    Median-centered deviation plots alongside absolute medians.

    For each variable, each group is centered on its own median so that
    within-group spread and between-group shape are visible even when
    absolute values differ greatly between groups. The absolute median
    for each group is shown as a text annotation.

    This is the complement to plot_snapshot_comparison — use both together.
    """
    features = [
        'min_sk38', 'min_ssi', 'min_sn38',
        'slab_thickness', 'slab_density', 'wl_shear_strength',
        'wl_burial_depth', 'wl_grain_size',
        'n_crust_layers', 'crust_thickness',
        'E_slab', 'sigma_t', 'Lambda',
    ]

    ncols = 3
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.5*nrows))
    axes = axes.ravel()

    groups_order = ['release', 'adjacent', 'reference']
    positions    = [1, 2, 3]
    colors       = [GROUP_COLORS[g] for g in groups_order]

    for ax, col in zip(axes, features):
        group_medians = {}
        data_centered = []

        for grp in groups_order:
            df = snap_data.get(grp, pd.DataFrame())
            if df.empty or col not in df.columns:
                data_centered.append(np.array([]))
                group_medians[grp] = np.nan
                continue
            vals = df[col].dropna().values
            if len(vals) < 4:
                data_centered.append(vals)
                group_medians[grp] = float(np.median(vals)) if len(vals) else np.nan
                continue
            # Clip outliers at 1.5 IQR
            q1, q3 = np.percentile(vals, [25, 75])
            iqr = q3 - q1
            vals = vals[(vals >= q1 - 1.5*iqr) & (vals <= q3 + 1.5*iqr)]
            med = float(np.median(vals))
            group_medians[grp] = med
            data_centered.append(vals - med)   # center on median

        try:
            bp = ax.boxplot(data_centered, positions=positions,
                            patch_artist=True, widths=0.6,
                            notch=True, bootstrap=1000, showfliers=False,
                            medianprops=dict(color='black', linewidth=2))
        except Exception:
            bp = ax.boxplot(data_centered, positions=positions,
                            patch_artist=True, widths=0.6,
                            notch=False, showfliers=False,
                            medianprops=dict(color='black', linewidth=2))

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Annotate each box with its absolute median
        y_top = ax.get_ylim()[1]
        for pos, grp in zip(positions, groups_order):
            med = group_medians.get(grp, np.nan)
            if not np.isnan(med):
                ax.text(pos, y_top, f'{med:.2g}',
                        ha='center', va='bottom', fontsize=7,
                        color=GROUP_COLORS[grp])

        ax.axhline(0, color='black', linewidth=0.7, linestyle='--', alpha=0.5)
        ax.set_xticks(positions)
        ax.set_xticklabels(['Rel', 'Adj', 'Ref'], fontsize=8)
        ax.set_ylabel(f'Δ {col} (centred on median)', fontsize=7)
        ax.grid(True, axis='y', alpha=0.25)

    for ax in axes[len(features):]:
        ax.set_visible(False)

    fig.suptitle(
        f'Deviation from group median - {snap_ts.date()} | '
        f'Little Professor | Jan 18 | buried >{min_depth:.0f}cm\n'
        f'Boxes show within-group spread; numbers show absolute median',
        fontsize=10)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"Deviation plot saved: {out_path}")


def run_snapshot_classifier(snap_data: dict, plots_dir: Path,
                              snap_ts: pd.Timestamp):
    """
    RF classifier: release (1) vs adjacent (0) at snapshot.
    Each cluster = one sample. Plots feature importance + SHAP if available.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report

    rel_df = snap_data.get('release', pd.DataFrame())
    adj_df = snap_data.get('adjacent', pd.DataFrame())
    if rel_df.empty or adj_df.empty:
        print("  Not enough data for classifier")
        return

    feature_cols = [c for c in rel_df.columns if c in adj_df.columns]
    rel_X = rel_df[feature_cols].copy()
    adj_X = adj_df[feature_cols].copy()

    X = pd.concat([rel_X, adj_X])
    y = np.array([1]*len(rel_X) + [0]*len(adj_X))

    # Fill NaN with median
    for c in feature_cols:
        med = X[c].median()
        X[c] = X[c].fillna(med)

    sc   = StandardScaler()
    X_sc = sc.fit_transform(X.values)

    rf = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                 max_features='sqrt', random_state=42)
    rf.fit(X_sc, y)
    y_pred = rf.predict(X_sc)

    print(f"  n_release={len(rel_X)}, n_adjacent={len(adj_X)}")
    print(classification_report(y, y_pred,
                                 target_names=['adjacent', 'release']))

    # Feature importance plot
    importances = pd.Series(rf.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ['#d32f2f' if imp >= importances.quantile(0.75) else '#888888'
              for imp in importances]
    importances.plot.barh(ax=ax, color=colors, edgecolor='none')
    ax.set_xlabel('Feature importance (RF impurity)', fontsize=9)
    ax.set_title(
        f'Feature importance: release vs adjacent - {snap_ts.date()} | '
        f'n_release={len(rel_X)}, n_adjacent={len(adj_X)}',
        fontsize=10)
    plt.tight_layout()
    out = plots_dir / f"feature_importance_{snap_ts.date()}.png"
    fig.savefig(str(out), dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"Feature importance plot: {out}")

    # Save trained model for future use
    import pickle
    model_path = plots_dir / f"rf_model_{snap_ts.date()}.pkl"
    with open(str(model_path), 'wb') as f_pkl:
        pickle.dump({'model': rf, 'scaler': sc,
                     'feature_cols': feature_cols,
                     'train_date': str(snap_ts.date()),
                     'n_release': len(rel_X),
                     'n_adjacent': len(adj_X)}, f_pkl)
    print(f"RF model saved: {model_path}")

    # SHAP if available
    try:
        import shap
        explainer   = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_sc)
        shap_rel    = shap_values[1] if isinstance(shap_values, list) else shap_values
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        shap.summary_plot(shap_rel, X_sc, feature_names=feature_cols,
                          show=False, plot_type='bar')
        out2 = plots_dir / f"shap_importance_{snap_ts.date()}.png"
        plt.savefig(str(out2), dpi=130, bbox_inches='tight')
        plt.close()
        print(f"SHAP plot: {out2}")
    except ImportError:
        print("  SHAP not installed — skipping (pip install shap)")




def cross_date_test(ds, groups: dict, location_names,
                    min_depth_cm: float, train_ts: pd.Timestamp,
                    plots_dir: Path):
    """
    Train RF on train_ts snapshot, score clusters at all available noon
    timestamps. Plots mean predicted P(release) per group over time.
    Answers: does the model trained on Jan 17 see the release zone
    as high-probability before the event, and low-probability after?
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    # Build training data at train_ts
    all_locs = {}
    for grp in ('release', 'adjacent'):
        ids = groups[grp]
        loc_mask = np.array([
            int(str(loc).split('_')[-1]) in ids
            for loc in location_names])
        if not loc_mask.any():
            continue
        ds_g = ds.isel(location=loc_mask)
        loc_ids = [int(str(loc).split('_')[-1])
                   for loc in location_names[loc_mask]]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ds_t = ds_g.sel(time=train_ts, method='nearest').compute()
        rows = []
        n_locs = ds_t.sizes.get('location', 1)
        for i in range(n_locs):
            try:
                ds_loc = ds_t.isel(location=i) if 'location' in ds_t.dims else ds_t
                feat = profile_features(ds_loc, min_depth_cm)
                if feat:
                    feat['cluster_id'] = loc_ids[i]
                    feat['group'] = grp
                    rows.append(feat)
            except Exception:
                continue
        all_locs[grp] = rows

    if not all_locs.get('release') or not all_locs.get('adjacent'):
        print("  Not enough training data")
        return

    rel_rows = all_locs['release']
    adj_rows = all_locs['adjacent']
    feature_cols = [k for k in rel_rows[0] if k not in ('cluster_id', 'group')]
    feature_cols = [k for k in feature_cols if k in
                    {k2 for k2 in adj_rows[0] if k2 not in ('cluster_id','group')}]

    X_train_rel = [[r.get(k, np.nan) for k in feature_cols] for r in rel_rows]
    X_train_adj = [[r.get(k, np.nan) for k in feature_cols] for r in adj_rows]
    X_train = np.array(X_train_rel + X_train_adj, dtype=float)
    y_train = np.array([1]*len(rel_rows) + [0]*len(adj_rows))

    col_meds = np.nanmedian(X_train, axis=0)
    for j in range(X_train.shape[1]):
        X_train[np.isnan(X_train[:, j]), j] = col_meds[j]

    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_train)
    rf = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                 max_features='sqrt', random_state=42)
    rf.fit(X_tr_sc, y_train)
    print(f"  Trained on {len(rel_rows)} release + {len(adj_rows)} adjacent clusters")

    # Score at each noon timestamp for each group
    times = pd.DatetimeIndex(ds.coords['time'].values)
    noon_ts = sorted({t for t in times if t.hour == 12})

    results = {grp: [] for grp in ('release', 'adjacent', 'reference')}
    timestamps = []

    for ts in noon_ts:
        ts_results = {}
        for grp, ids in groups.items():
            loc_mask = np.array([
                int(str(loc).split('_')[-1]) in ids
                for loc in location_names])
            if not loc_mask.any():
                ts_results[grp] = np.nan
                continue
            ds_g = ds.isel(location=loc_mask)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                ds_t = ds_g.sel(time=ts, method='nearest').compute()
            n_locs = ds_t.sizes.get('location', 1)
            probs = []
            for i in range(n_locs):
                try:
                    ds_loc = ds_t.isel(location=i) if 'location' in ds_t.dims else ds_t
                    feat = profile_features(ds_loc, min_depth_cm)
                    if not feat:
                        continue
                    x = np.array([[feat.get(k, np.nan) for k in feature_cols]])
                    for j in range(x.shape[1]):
                        if np.isnan(x[0, j]):
                            x[0, j] = col_meds[j]
                    x_sc = sc.transform(x)
                    probs.append(rf.predict_proba(x_sc)[0][1])
                except Exception:
                    continue
            ts_results[grp] = float(np.mean(probs)) if probs else np.nan

        timestamps.append(ts)
        for grp in results:
            results[grp].append(ts_results.get(grp, np.nan))

    # Plot
    fig, ax = plt.subplots(figsize=(12, 4))
    for grp, vals in results.items():
        ax.plot(timestamps, vals, color=GROUP_COLORS[grp],
                label=GROUP_LABELS[grp], linewidth=1.5, alpha=0.9)
    ax.axvline(pd.Timestamp('2026-01-18'), color='darkred',
               linewidth=2, linestyle='-', label='Jan 18 event')
    ax.axvline(train_ts, color='gray', linewidth=1.5, linestyle='--',
               label=f'Training snapshot ({train_ts.date()})')
    ax.set_ylabel('Mean P(release) per group', fontsize=10)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.set_title(
        f'Cross-date test: P(release) trained on {train_ts.date()} - '
        f'scored across season',
        fontsize=10)
    plt.tight_layout()
    out = plots_dir / f'cross_date_test_{train_ts.date()}.png'
    fig.savefig(str(out), dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"Cross-date test plot: {out}")


def plot_meloche_comparison(meloche_df: pd.DataFrame, plots_dir: Path,
                             snap_ts):
    """Box plots of Meloche et al. (2025) parameters by group."""
    panels = [
        ('theta',         'θ — WL strength gradient (Pa/m)', None, False),
        ('tau_g',         'τ_g — driving shear stress (Pa)',  None, False),
        ('Lambda',        'Λ — elastic length (m)',           None, False),
        ('Pi1_elastic',   'Π₁ elastic (lower = longer arrest)', None, True),
        ('Pi2_brittle',   'Π₂ brittle (lower = longer arrest)', None, True),
        ('A_ca_brittle',  'A_ca brittle estimate (m)',        None, False),
        ('L_t',           'L_t quasi-static tensile length (m)', None, False),
        ('slope_angle',   'Slope angle (°)',                  None, False),
    ]

    ncols = 3
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.5*nrows))
    axes = axes.ravel()

    groups_order = ['release', 'adjacent', 'reference']
    positions    = [1, 2, 3]
    colors       = [GROUP_COLORS[g] for g in groups_order]

    for ax, (col, label, threshold, invert) in zip(axes, panels):
        data_by_group = []
        for grp in groups_order:
            sub = meloche_df[meloche_df['group'] == grp]
            vals = sub[col].dropna().values if col in sub.columns else np.array([])
            if len(vals) > 3:
                q1, q3 = np.percentile(vals, [25, 75])
                iqr = q3 - q1
                vals = vals[(vals >= q1 - 1.5*iqr) & (vals <= q3 + 1.5*iqr)]
            data_by_group.append(vals)

        try:
            bp = ax.boxplot(data_by_group, positions=positions,
                            patch_artist=True, widths=0.6,
                            notch=True, bootstrap=1000, showfliers=False,
                            medianprops=dict(color='black', linewidth=2))
        except Exception:
            bp = ax.boxplot(data_by_group, positions=positions,
                            patch_artist=True, widths=0.6,
                            notch=False, showfliers=False,
                            medianprops=dict(color='black', linewidth=2))

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticks(positions)
        ax.set_xticklabels(['Release', 'Adjacent', 'Ref'], fontsize=8)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(True, axis='y', alpha=0.25)
        if invert:
            ax.invert_yaxis()

    for ax in axes[len(panels):]:
        ax.set_visible(False)

    fig.suptitle(
        'Meloche et al. (2025) crack arrest parameters - ' + str(snap_ts.date()) +
        ' | Little Professor',
        fontsize=11)
    plt.tight_layout()
    out = plots_dir / f"meloche_comparison_{snap_ts.date()}.png"
    fig.savefig(str(out), dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"Meloche comparison plot: {out}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-dir',
                        default='/home/ron/snowpack_model_feeder')
    parser.add_argument('--pro-dir',
                        default='/home/ron/snowpack/little_prof/output')
    parser.add_argument('--zarr-path',
                        default='/home/ron/snowpack/little_prof/output/'
                                'slope_snowpack.zarr')
    parser.add_argument('--release-geojson', default=None)
    parser.add_argument('--min-depth',  type=float, default=30.0)
    parser.add_argument('--snapshot-date', default='2026-01-18',
                        help='Date for snapshot analysis (default: day before event)')
    parser.add_argument('--classifier', action='store_true',
                        help='Train RF classifier and plot feature importance')
    parser.add_argument('--all-clusters', action='store_true',
                        help='Also extract features for ALL start zone '
                             'clusters (not just release+adjacent groups). '
                             'Saves all_start_zone_features_YYYY-MM-DD.csv '
                             'for use by the probabilistic boundary model.')
    parser.add_argument('--cross-date-test', action='store_true',
                        help='Train on snapshot date, score other survey dates')
    args = parser.parse_args()

    cfg = ProjectConfig(project_dir=Path(args.project_dir))

    geojson_path = Path(args.release_geojson) if args.release_geojson else \
        cfg.project_dir / 'data' / 'boundaries' / 'avalanche_release_area.geojson'

    with rasterio.open(str(cfg.resampled_dir / "dem_1m.tif")) as src:
        dem       = src.read(1).astype(np.float32)
        dem[dem == src.nodata] = np.nan
        transform = src.transform

    domain_mask = ~np.isnan(dem)
    cluster_map = np.load(str(cfg.analysis_dir / "cluster_map.npy"))

    print("Building spatial masks...")
    release_mask    = geojson_to_mask(geojson_path, dem.shape, transform)
    start_zone_mask = kml_to_mask(cfg.start_zone_kml, dem.shape, transform)

    print("Assigning clusters...")
    groups = assign_cluster_groups(
        cluster_map, release_mask, start_zone_mask, dem, domain_mask)

    group_ids_path = cfg.analysis_dir / "release_zone_groups.json"
    serializable_groups = {
        k: [int(x) for x in sorted(v)]
        for k, v in groups.items()
    }
    group_ids_path.write_text(json.dumps(serializable_groups, indent=2))
    print(f"Cluster IDs saved: {group_ids_path}")

    print("Loading SNOWPACK dataset...")
    zarr_path = Path(args.zarr_path)
    if zarr_path.exists():
        _raw = xr.open_zarr(str(zarr_path))
        # Deduplicate time index — can arise from batched append
        if _raw.indexes['time'].duplicated().any():
            print("  Warning: duplicate time entries in Zarr, deduplicating...")
            _, idx = np.unique(_raw.coords['time'].values, return_index=True)
            _raw = _raw.isel(time=idx)
        ds = xsnow.xsnowDataset(_raw)
    else:
        ds = xsnow.read(args.pro_dir)

    location_names = ds.coords['location'].values

    snap_ts = pd.Timestamp(args.snapshot_date)
    print(f"Snapshot: {snap_ts.date()}")

    print("Extracting per-cluster features at snapshot...")
    snap_data = {}
    for grp, ids in groups.items():
        print(f"  {GROUP_LABELS[grp]} ({len(ids)} clusters)...")
        loc_mask = np.array([
            int(str(loc).split('_')[-1]) in ids
            for loc in location_names])
        if not loc_mask.any():
            snap_data[grp] = pd.DataFrame()
            continue

        ds_group = ds.isel(location=loc_mask)
        loc_ids  = [int(str(loc).split('_')[-1])
                    for loc in location_names[loc_mask]]

        # Only look at conditions from Dec 1 onward (prior year)
        season_year = snap_ts.year - 1 if snap_ts.month < 9 else snap_ts.year
        dec1 = pd.Timestamp(f'{season_year}-12-01')
        if snap_ts < dec1:
            print(f"    Snapshot {snap_ts.date()} is before Dec 1 {season_year} — skipping")
            snap_data[grp] = pd.DataFrame()
            continue

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ds_t = ds_group.sel(time=snap_ts, method='nearest').compute()

        rows = []
        n_locs = ds_t.sizes.get('location', 1)
        for i in range(n_locs):
            try:
                ds_loc = ds_t.isel(location=i) \
                    if 'location' in ds_t.dims else ds_t
                feat = profile_features(ds_loc, args.min_depth)
                if feat:
                    feat['cluster_id'] = loc_ids[i]
                    rows.append(feat)
            except Exception:
                continue

        df = pd.DataFrame(rows).set_index('cluster_id') if rows \
             else pd.DataFrame()
        snap_data[grp] = df
        if not df.empty:
            print(f"    {len(df)} clusters, {df.notna().all(axis=1).sum()} complete")

    cfg.plots_dir.mkdir(parents=True, exist_ok=True)

    # --- Statistical comparison plot ---
    out_plot = cfg.plots_dir / f"release_zone_comparison_{snap_ts.date()}.png"
    plot_snapshot_comparison(snap_data, out_plot, args.min_depth, snap_ts)

    out_dev = cfg.plots_dir / f"release_zone_deviations_{snap_ts.date()}.png"
    plot_snapshot_deviations(snap_data, out_dev, snap_ts, args.min_depth)

    # --- Classifier ---
    if args.classifier:
        print("Running RF classifier...")
        run_snapshot_classifier(snap_data, cfg.plots_dir, snap_ts)

        # Cross-date test: train on snapshot, score all survey dates
        if args.cross_date_test:
            print("Cross-date test: training on snapshot, scoring other dates...")
            cross_date_test(ds, groups, location_names, args.min_depth,
                            snap_ts, cfg.plots_dir)

    # --- Meloche et al. (2025) spatial features ---
    print("Computing Meloche et al. (2025) spatial features (θ, Π₁, Π₂, A_ca)...")
    meloche_df = compute_meloche_features(
        snap_data, cluster_map, dem, transform, snap_ts)

    if not meloche_df.empty:
        out_meloche = cfg.analysis_dir / f"meloche_features_{snap_ts.date()}.csv"
        meloche_df.to_csv(str(out_meloche))
        print(f"Meloche features saved: {out_meloche}")

        # Print summary by group
        group_map = {}
        for grp, df in snap_data.items():
            for cid in df.index:
                group_map[cid] = grp
        meloche_df['group'] = meloche_df.index.map(lambda x: group_map.get(x, 'unknown'))
        for grp in ('release', 'adjacent', 'reference'):
            sub = meloche_df[meloche_df['group'] == grp]
            if sub.empty:
                continue
            for col in ('theta', 'Pi1_elastic', 'Pi2_brittle', 'A_ca_brittle'):
                if col in sub.columns:
                    med = sub[col].median()
                    print(f"  {grp:10s} {col:15s}: median={med:.4f}")

        # Plot Meloche comparison
        plot_meloche_comparison(meloche_df, cfg.plots_dir, snap_ts)

    # --- Save feature table (release + adjacent + reference groups) ---
    out_csv = cfg.analysis_dir / f"release_zone_features_{snap_ts.date()}.csv"
    rows = []
    for grp, df in snap_data.items():
        if df.empty:
            continue
        df2 = df.copy()
        df2['group'] = grp
        rows.append(df2)
    if rows:
        pd.concat(rows).to_csv(str(out_csv))
        print(f"Feature table saved: {out_csv}")

    # --- All start zone clusters (for probabilistic boundary model) ---
    if args.all_clusters:
        print("\nExtracting features for ALL start zone clusters...")
        print("(Required for probabilistic boundary model signed features)")

        # All cluster IDs in start zone
        all_sz_ids = set(int(c) for c in np.unique(cluster_map[start_zone_mask])
                         if c > 0)
        # Already-processed IDs
        done_ids   = set()
        for df in snap_data.values():
            if not df.empty:
                done_ids.update(df.index.tolist())
        remaining_ids = all_sz_ids - done_ids
        print(f"  Total start zone clusters: {len(all_sz_ids)}")
        print(f"  Already processed:         {len(done_ids)}")
        print(f"  Remaining to process:      {len(remaining_ids)}")

        loc_mask = np.array([
            int(str(loc).split('_')[-1]) in remaining_ids
            for loc in location_names])

        all_rows = []
        if loc_mask.any():
            ds_all = ds.isel(location=loc_mask)
            loc_ids_all = [int(str(loc).split('_')[-1])
                           for loc in location_names[loc_mask]]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                ds_t_all = ds_all.sel(
                    time=snap_ts, method='nearest').compute()
            n_all = ds_t_all.sizes.get('location', 1)
            for i in range(n_all):
                try:
                    ds_loc = ds_t_all.isel(location=i) \
                        if 'location' in ds_t_all.dims else ds_t_all
                    feat = profile_features(ds_loc, args.min_depth)
                    if feat:
                        feat['cluster_id'] = loc_ids_all[i]
                        feat['group']      = 'start_zone'
                        all_rows.append(feat)
                except Exception:
                    continue
            print(f"  Extracted features for {len(all_rows)} additional clusters")

        # Merge with existing snap_data and save
        existing_rows = []
        for grp, df in snap_data.items():
            if df.empty: continue
            df2 = df.copy(); df2['group'] = grp
            existing_rows.append(df2)
        all_dfs = existing_rows
        if all_rows:
            all_dfs.append(
                pd.DataFrame(all_rows).set_index('cluster_id'))
        if all_dfs:
            all_features_df = pd.concat(all_dfs)
            # Deduplicate — existing group assignments take priority
            all_features_df = all_features_df[
                ~all_features_df.index.duplicated(keep='first')]
            out_all = (cfg.analysis_dir /
                       f"all_start_zone_features_{snap_ts.date()}.csv")
            all_features_df.to_csv(str(out_all))
            print(f"  All start zone features saved: {out_all}")
            print(f"  Total clusters: {len(all_features_df)}")

        # Recompute Meloche features over full start zone
        if all_dfs:
            snap_data_full = {}
            for grp, df in snap_data.items():
                snap_data_full[grp] = df
            if all_rows:
                snap_data_full['start_zone'] = (
                    pd.DataFrame(all_rows).set_index('cluster_id'))
            print("  Recomputing Meloche features over full start zone...")
            meloche_full = compute_meloche_features(
                snap_data_full, cluster_map, dem, transform, snap_ts)
            if not meloche_full.empty:
                out_mel_all = (cfg.analysis_dir /
                               f"meloche_features_all_{snap_ts.date()}.csv")
                meloche_full.to_csv(str(out_mel_all))
                print(f"  Full Meloche features saved: {out_mel_all} "
                      f"({len(meloche_full)} clusters)")


if __name__ == '__main__':
    main()
    
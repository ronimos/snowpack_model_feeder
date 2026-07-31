"""
Diagnostic plots for the distributed SNOWPACK forcing pipeline.

All functions write PNG files to cfg.plots_dir and print the output path.
They are side-effect only — no return values.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import ProjectConfig


def plot_avalanche_results(period_results: dict, corrected_periods: dict,
                           dem: np.ndarray, transform, cfg: ProjectConfig):
    """Summary plot: bar chart of all periods + spatial maps of corrected ones."""
    bounds = [transform[2], transform[2] + dem.shape[1],
              transform[5] + dem.shape[0] * transform[4], transform[5]]

    pair_ids = sorted(period_results.keys())
    n_corrected = len(corrected_periods)
    n_cols = max(2, 1 + n_corrected)

    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    # Panel 1: bar chart — largest erosion volume per period
    ax = axes[0]
    metrics, colors, labels = [], [], []
    for pid in pair_ids:
        pr = period_results[pid]
        vol = pr['regions'][0]['volume_m3'] if pr['regions'] else 0
        metrics.append(vol)
        if pid in corrected_periods:
            colors.append('#d32f2f')
        elif pr['is_known']:
            colors.append('#ff9800')
        else:
            colors.append('#78909c')
        labels.append(pid.split('__')[1])

    ax.bar(range(len(pair_ids)), metrics, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(range(len(pair_ids)))
    ax.set_xticklabels(labels, rotation=60, fontsize=7, ha='right')
    ax.set_ylabel('Largest erosion region (m³)')
    ax.set_title('Avalanche detection by period\nRed=corrected, Orange=known, Gray=none')

    # Remaining panels: spatial maps of corrected periods
    for idx, pid in enumerate(sorted(corrected_periods.keys())):
        if idx + 1 >= len(axes):
            break
        ax = axes[idx + 1]
        avy_path = cfg.analysis_dir / f"avalanche_dhs_{pid}.npy"
        if avy_path.exists():
            avy_dhs = np.load(str(avy_path))
            ax.imshow(dem, cmap='terrain', extent=bounds, alpha=0.3)
            display = np.where(np.abs(avy_dhs) > 0.01, avy_dhs, np.nan)
            im = ax.imshow(np.clip(display, -2, 0.5), cmap='RdBu',
                           vmin=-1.5, vmax=0.5, extent=bounds, alpha=0.8)
            plt.colorbar(im, ax=ax, shrink=0.7, label='Avy ΔHS (m)')

        pr = period_results[pid]
        r0 = pr['regions'][0] if pr['regions'] else None
        info = (f"{pid}\n{r0['n_cells']} cells, {r0['volume_m3']:.0f}m³"
                if r0 else pid)
        ax.set_title(info)

    for ax in axes:
        ax.set_xlabel('Easting')

    plt.tight_layout()
    out_path = cfg.plots_dir / "avalanche_detection.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot saved to {out_path}")


def plot_gap_fill_validation(all_metrics: list, cfg: ProjectConfig):
    """Validation plots for gap-fill results: RMSE/R² bar charts + per-period scatter."""
    valid_metrics = [m for m in all_metrics if not np.isnan(m.get('r', np.nan))]
    n = len(valid_metrics)
    if n == 0:
        return

    n_scatter = min(n, 12)
    n_cols = min(4, n_scatter)
    n_rows = (n_scatter + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows + 1, n_cols, figsize=(5 * n_cols, 4 * (n_rows + 1)))
    if n_rows + 1 == 1:
        axes = axes[np.newaxis, :]

    ax_rmse = fig.add_subplot(n_rows + 1, 2, 1)
    ax_r2 = fig.add_subplot(n_rows + 1, 2, 2)

    for j in range(n_cols):
        axes[0, j].set_visible(False)

    labels = [m['date_b'] for m in valid_metrics]
    rmses = [m['rmse'] * 100 for m in valid_metrics]
    r2s = [m['r'] ** 2 if not np.isnan(m['r']) else 0 for m in valid_metrics]

    colors_rmse = ['#d32f2f' if r > 100 else '#ff9800' if r > 60 else '#4caf50'
                   for r in rmses]
    ax_rmse.bar(range(n), rmses, color=colors_rmse, edgecolor='black', linewidth=0.5)
    ax_rmse.set_xticks(range(n))
    ax_rmse.set_xticklabels(labels, rotation=60, fontsize=7, ha='right')
    ax_rmse.set_ylabel('RMSE (cm)')
    ax_rmse.set_title('Endpoint RMSE by period')
    ax_rmse.axhline(np.median(rmses), color='black', linestyle='--', linewidth=0.8,
                    label=f'median={np.median(rmses):.0f} cm')
    ax_rmse.legend(fontsize=8)

    colors_r2 = ['#d32f2f' if r < 0.3 else '#ff9800' if r < 0.6 else '#4caf50'
                 for r in r2s]
    ax_r2.bar(range(n), r2s, color=colors_r2, edgecolor='black', linewidth=0.5)
    ax_r2.set_xticks(range(n))
    ax_r2.set_xticklabels(labels, rotation=60, fontsize=7, ha='right')
    ax_r2.set_ylabel('R²')
    ax_r2.set_title('Endpoint R² by period')
    ax_r2.axhline(np.median(r2s), color='black', linestyle='--', linewidth=0.8,
                  label=f'median={np.median(r2s):.2f}')
    ax_r2.set_ylim(0, 1.05)
    ax_r2.legend(fontsize=8)

    for idx in range(n_scatter):
        row = 1 + idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        m = valid_metrics[idx]
        mask = m['valid'] & ~np.isnan(m['pred']) & ~np.isnan(m['obs'])
        obs = m['obs'][mask].ravel()
        pred = m['pred'][mask].ravel()

        step = max(1, len(obs) // 5000)
        ax.scatter(obs[::step], pred[::step], s=1, alpha=0.2, c='steelblue')
        lim = max(np.percentile(obs, 99), np.percentile(pred, 99), 0.5)
        ax.plot([0, lim], [0, lim], 'r--', linewidth=0.8)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect('equal')
        ax.set_title(f"{m['date_b']}\nR²={m['r']**2:.2f}, RMSE={m['rmse']*100:.0f}cm",
                     fontsize=8)
        ax.set_xlabel('Observed (m)', fontsize=7)
        ax.set_ylabel('Predicted (m)', fontsize=7)
        ax.tick_params(labelsize=6)

    for idx in range(n_scatter, n_rows * n_cols):
        row = 1 + idx // n_cols
        col = idx % n_cols
        if row < axes.shape[0] and col < axes.shape[1]:
            axes[row, col].set_visible(False)

    plt.tight_layout()
    out_path = cfg.plots_dir / "gap_fill_validation.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nValidation plot saved to {out_path}")

    print(f"\n{'Period':<35s} {'RMSE(cm)':>10s} {'R²':>8s} {'Bias(cm)':>10s} {'Hours':>6s}")
    print("-" * 75)
    for m in valid_metrics:
        r2 = m['r'] ** 2 if not np.isnan(m['r']) else np.nan
        print(f"{m['pair_id']:<35s} {m['rmse']*100:>10.1f} {r2:>8.3f} "
              f"{m['bias']*100:>10.1f} {m['n_hours']:>6d}")
    med_rmse = np.median([m['rmse'] * 100 for m in valid_metrics])
    med_r2 = np.median([m['r'] ** 2 for m in valid_metrics if not np.isnan(m['r'])])
    print("-" * 75)
    print(f"{'Median':<35s} {med_rmse:>10.1f} {med_r2:>8.3f}")


def plot_cluster_map(cluster_map: np.ndarray, dem: np.ndarray, transform,
                     cids: np.ndarray, sizes: list, hs_matrix: np.ndarray,
                     cell_idx: np.ndarray, survey_dates: list,
                     cfg: ProjectConfig):
    """Cluster visualisation: spatial map, HS trajectories, size histogram."""
    from matplotlib.colors import LightSource

    bounds = [transform[2], transform[2] + dem.shape[1],
              transform[5] + dem.shape[0] * transform[4], transform[5]]

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    # Panel 1: cluster map on hillshaded DEM
    ax = axes[0]
    fill = np.where(np.isnan(dem), np.nanmean(dem), dem)
    ls = LightSource(azdeg=315, altdeg=45)
    hillshade = ls.hillshade(fill, dx=1.0, dy=1.0)
    ax.imshow(hillshade, cmap='gray', extent=bounds, alpha=0.6)
    display = np.where(cluster_map > 0, cluster_map, np.nan)
    ax.imshow(display, cmap='nipy_spectral', interpolation='nearest',
              extent=bounds, alpha=0.7)
    ax.set_title(f'{len(cids)} clusters\n({int(np.sum(cluster_map > 0))} cells)')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')

    # Panel 2: sample cluster HS trajectories
    ax = axes[1]
    np.random.seed(42)
    sample_n = min(30, len(cids))
    sample_idx = np.random.choice(len(cids), sample_n, replace=False)
    for idx in sample_idx:
        c = cids[idx]
        mask_c = cluster_map[cell_idx[:, 0], cell_idx[:, 1]] == c
        if mask_c.any():
            mean_hs = np.mean(hs_matrix[mask_c], axis=0)
            ax.plot(range(len(survey_dates)), np.cumsum(mean_hs), alpha=0.5, linewidth=1)

    ax.set_xticks(range(len(survey_dates)))
    short_dates = [d[-5:] if len(d) > 5 else d for d in survey_dates]
    ax.set_xticklabels(short_dates, rotation=60, fontsize=7, ha='right')
    ax.set_ylabel('HS (m)')
    ax.set_title(f'Cluster mean HS trajectories\n({sample_n} of {len(cids)} shown)')
    ax.set_xlim(-0.5, len(survey_dates) - 0.5)

    # Panel 3: cluster size histogram
    ax = axes[2]
    ax.hist(sizes, bins=max(10, len(cids) // 5), edgecolor='black',
            alpha=0.7, color='steelblue')
    ax.set_xlabel('Cells per cluster')
    ax.set_ylabel('Count')
    ax.set_title(f'Cluster sizes\nmedian={int(np.median(sizes))}, '
                 f'range=[{min(sizes)}, {max(sizes)}]')
    ax.axvline(np.median(sizes), color='red', linestyle='--', linewidth=1.5,
               label=f'median={int(np.median(sizes))}')
    ax.legend()

    plt.tight_layout()
    out_path = cfg.plots_dir / f"cluster_map_{len(cids)}.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Cluster plot saved to {out_path}")


def plot_cluster_variability(cluster_map: np.ndarray, dem: np.ndarray,
                              transform, survey_grids: dict,
                              cfg: ProjectConfig, cids: np.ndarray):
    """Intra-cluster HS variability diagnostics: spatial maps, scatter plots, summary table."""
    bounds = [transform[2], transform[2] + dem.shape[1],
              transform[5] + dem.shape[0] * transform[4], transform[5]]

    dates = sorted(survey_grids.keys())
    survey_stack = np.stack([survey_grids[d] for d in dates])

    cids = np.unique(cluster_map[cluster_map > 0])
    n_clusters = len(cids)

    intra_std_map = np.full(dem.shape, np.nan)
    intra_range_map = np.full(dem.shape, np.nan)
    intra_cv_map = np.full(dem.shape, np.nan)

    stats = []
    for cid in cids:
        mask = cluster_map == cid
        rows, cols = np.where(mask)
        n_cells = len(rows)

        hs_vals = survey_stack[:, rows, cols]  # (n_surveys, n_cells)
        cell_std = np.nanstd(hs_vals, axis=1)
        cell_mean = np.nanmean(hs_vals, axis=1)
        cell_range = np.nanmax(hs_vals, axis=1) - np.nanmin(hs_vals, axis=1)

        mean_std = float(np.mean(cell_std))
        mean_cv = float(np.mean(cell_std / np.maximum(cell_mean, 0.01)))
        mean_range = float(np.mean(cell_range))
        cluster_mean_hs = np.nanmean(hs_vals, axis=1, keepdims=True)
        rmse = float(np.sqrt(np.nanmean((hs_vals - cluster_mean_hs) ** 2)))

        intra_std_map[mask] = mean_std
        intra_range_map[mask] = mean_range
        intra_cv_map[mask] = mean_cv

        stats.append({
            'n_cells': n_cells, 'mean_std': mean_std,
            'mean_cv': mean_cv, 'mean_range': mean_range,
            'rmse': rmse, 'mean_hs': float(np.mean(cell_mean)),
        })

    stds = [s['mean_std'] * 100 for s in stats]
    ranges_cm = [s['mean_range'] * 100 for s in stats]
    rmses = [s['rmse'] * 100 for s in stats]
    sizes = [s['n_cells'] for s in stats]

    tight = sum(1 for s in stds if s < 10)
    medium = sum(1 for s in stds if 10 <= s < 30)
    loose = sum(1 for s in stds if s >= 30)

    fig, axes = plt.subplots(2, 3, figsize=(20, 13))

    # Row 1: spatial maps
    ax = axes[0, 0]
    im = ax.imshow(intra_std_map * 100, cmap='YlOrRd', vmin=0, vmax=50, extent=bounds)
    ax.set_title('Intra-cluster HS std (cm)\n(lower = more uniform)')
    plt.colorbar(im, ax=ax, label='Std (cm)', shrink=0.7)

    ax = axes[0, 1]
    im = ax.imshow(intra_range_map * 100, cmap='YlOrRd', vmin=0, vmax=200, extent=bounds)
    ax.set_title('Intra-cluster HS range (cm)\n(max−min within cluster)')
    plt.colorbar(im, ax=ax, label='Range (cm)', shrink=0.7)

    ax = axes[0, 2]
    im = ax.imshow(intra_cv_map, cmap='YlOrRd', vmin=0, vmax=2, extent=bounds)
    ax.set_title('Intra-cluster CV\n(std/mean, normalised variability)')
    plt.colorbar(im, ax=ax, label='CV', shrink=0.7)

    for ax in axes[0]:
        ax.set_xlabel('Easting (m)')
    axes[0, 0].set_ylabel('Northing (m)')

    # Row 2 left: size vs variability scatter
    ax = axes[1, 0]
    ax.scatter(sizes, stds, s=8, alpha=0.4, c='steelblue', edgecolor='none')
    ax.set_xlabel('Cluster size (cells)')
    ax.set_ylabel('Intra-cluster std (cm)')
    ax.set_title('Size vs variability')
    ax.axhline(np.median(stds), color='red', linestyle='--', linewidth=1,
               label=f'median={np.median(stds):.0f}cm')
    ax.legend(fontsize=8)

    # Row 2 centre: std histogram
    ax = axes[1, 1]
    ax.hist(stds, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(10, color='green', linestyle='--', linewidth=1, label='tight (<10cm)')
    ax.axvline(30, color='red', linestyle='--', linewidth=1, label='loose (>30cm)')
    ax.set_xlabel('Intra-cluster std (cm)')
    ax.set_ylabel('Count')
    ax.set_title(f'Intra-cluster std distribution\n{n_clusters} clusters')
    ax.legend(fontsize=8)

    # Row 2 right: summary table
    ax = axes[1, 2]
    ax.axis('off')
    table_data = [
        ['Metric', 'Median', 'P10', 'P90'],
        ['Std (cm)', f'{np.median(stds):.0f}',
         f'{np.percentile(stds, 10):.0f}', f'{np.percentile(stds, 90):.0f}'],
        ['Range (cm)', f'{np.median(ranges_cm):.0f}',
         f'{np.percentile(ranges_cm, 10):.0f}', f'{np.percentile(ranges_cm, 90):.0f}'],
        ['RMSE (cm)', f'{np.median(rmses):.0f}',
         f'{np.percentile(rmses, 10):.0f}', f'{np.percentile(rmses, 90):.0f}'],
    ]
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.5, 2.0)
    for j in range(len(table_data[0])):
        table[0, j].set_facecolor('#cce5ff')
        table[0, j].set_text_props(fontweight='bold')
    for i in range(1, len(table_data)):
        color = '#f0f4f8' if i % 2 == 0 else 'white'
        for j in range(len(table_data[0])):
            table[i, j].set_facecolor(color)

    summary = (f"\nCluster quality ({n_clusters} clusters):\n"
               f"  Tight  (<10 cm std):  {tight:>5d}  ({100*tight/n_clusters:.0f}%)\n"
               f"  Medium (10-30 cm):    {medium:>5d}  ({100*medium/n_clusters:.0f}%)\n"
               f"  Loose  (>30 cm):      {loose:>5d}  ({100*loose/n_clusters:.0f}%)")
    ax.text(0.5, 0.22, summary, transform=ax.transAxes, fontsize=10,
            ha='center', va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f8f8',
                      edgecolor='#cccccc'))

    plt.tight_layout()
    out_path = cfg.plots_dir / f"cluster_variability_{n_clusters}.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Variability plot saved to {out_path}")

    print(f"\n  Cluster variability ({n_clusters} clusters):")
    print(f"    Std:   median={np.median(stds):.0f}cm, "
          f"P10={np.percentile(stds,10):.0f}cm, P90={np.percentile(stds,90):.0f}cm")
    print(f"    Range: median={np.median(ranges_cm):.0f}cm, "
          f"P90={np.percentile(ranges_cm,90):.0f}cm")
    print(f"    Tight (<10cm): {tight}/{n_clusters} ({100*tight/n_clusters:.0f}%)")
    print(f"    Medium (10-30cm): {medium}/{n_clusters} ({100*medium/n_clusters:.0f}%)")
    print(f"    Loose (>30cm): {loose}/{n_clusters} ({100*loose/n_clusters:.0f}%)")
    print(f"    RMSE (cm): Median: {np.median(rmses):.0f},  "
          f"P10={np.percentile(rmses,10):.0f}cm, P90={np.percentile(rmses,90):.0f}cm")


def plot_avalanche_boundaries(hs_before: np.ndarray,
                               hs_after: np.ndarray,
                               dem: np.ndarray,
                               transform,
                               result: dict,
                               period_id: str,
                               out_dir,
                               param_tag: str = '',
                               start_zone_mask: np.ndarray = None) -> str:
    """
    Six-panel plot for avalanche boundary detection results.

    Panels:
      1. HS before survey
      2. HS after survey
      3. dHS anomaly (raw)
      4. Canny edges overlaid on dHS anomaly
      5. Release mask + contours overlaid on dHS anomaly
      6. Deposit mask + contours overlaid on dHS anomaly
    """
    from matplotlib.colors import LightSource
    from matplotlib.patches import Patch
    from pathlib import Path

    out_dir = Path(out_dir)

    bounds = [transform[2],
              transform[2] + dem.shape[1] * transform[0],
              transform[5] + dem.shape[0] * transform[4],
              transform[5]]

    fill_dem = np.where(np.isnan(dem), np.nanmean(dem), dem)
    hillshade = LightSource(azdeg=315, altdeg=45).hillshade(
        fill_dem, dx=1.0, dy=1.0)

    dhs_anomaly = result['dhs_anomaly']
    vlim = np.nanpercentile(np.abs(dhs_anomaly[~np.isnan(dhs_anomaly)]), 95)
    vlim = max(vlim, 0.05)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.ravel()

    def base(ax, title):
        ax.imshow(hillshade, cmap='gray', extent=bounds,
                  aspect='auto', alpha=0.5)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    # Panel 1 — HS before
    base(axes[0], 'HS before (m)')
    im = axes[0].imshow(hs_before, cmap='Blues', vmin=0, vmax=3.5,
                        extent=bounds, alpha=0.8, aspect='auto')
    plt.colorbar(im, ax=axes[0], shrink=0.75)

    # Panel 2 — HS after
    base(axes[1], 'HS after (m)')
    im = axes[1].imshow(hs_after, cmap='Blues', vmin=0, vmax=3.5,
                        extent=bounds, alpha=0.8, aspect='auto')
    plt.colorbar(im, ax=axes[1], shrink=0.75)

    # Panel 3 — dHS anomaly
    base(axes[2], 'dHS anomaly (m)')
    im = axes[2].imshow(dhs_anomaly, cmap='RdBu', vmin=-vlim, vmax=vlim,
                        extent=bounds, alpha=0.85, aspect='auto')
    plt.colorbar(im, ax=axes[2], shrink=0.75)

    # Panel 4 — Canny edges on dHS anomaly
    base(axes[3], 'Canny edges on dHS anomaly')
    axes[3].imshow(dhs_anomaly, cmap='RdBu', vmin=-vlim, vmax=vlim,
                   extent=bounds, alpha=0.7, aspect='auto')
    edge_overlay = np.ma.masked_where(~result['crown_edges'],
                                       result['crown_edges'].astype(float))
    axes[3].imshow(edge_overlay, cmap='autumn', vmin=0, vmax=1,
                   extent=bounds, alpha=1.0, aspect='auto')

    # Panel 5 — Release zone (detected mask only, no start zone outline)
    base(axes[4], f"Release zone  {result['release_area_m2']:.0f} m²"
                  f"  {result['release_volume_m3']:.1f} m³")
    axes[4].imshow(dhs_anomaly, cmap='RdBu', vmin=-vlim, vmax=vlim,
                   extent=bounds, alpha=0.6, aspect='auto')
    if result['release_mask'].any():
        rel_overlay = np.ma.masked_where(
            ~result['release_mask'], np.ones_like(dem))
        axes[4].imshow(rel_overlay, cmap='Reds', vmin=0, vmax=1,
                       extent=bounds, alpha=0.5, aspect='auto')
        # Only draw contours with enough points to be real detections
        for contour in result['release_contours']:
            if len(contour) < 50:
                continue
            xs = [transform[2] + c * transform[0] for r, c in contour]
            ys = [transform[5] + r * transform[4] for r, c in contour]
            axes[4].plot(xs, ys, 'r-', linewidth=2.0)

    # Panel 6 — Deposit zone
    base(axes[5], f"Deposit zone  {result['deposit_area_m2']:.0f} m²"
                  f"  {result['deposit_volume_m3']:.1f} m³")
    axes[5].imshow(dhs_anomaly, cmap='RdBu', vmin=-vlim, vmax=vlim,
                   extent=bounds, alpha=0.6, aspect='auto')
    if result['deposit_mask'].any():
        dep_overlay = np.ma.masked_where(
            ~result['deposit_mask'], np.ones_like(dem))
        axes[5].imshow(dep_overlay, cmap='Blues', vmin=0, vmax=1,
                       extent=bounds, alpha=0.5, aspect='auto')
    for contour in result['deposit_contours']:
        xs = [transform[2] + c * transform[0] for r, c in contour]
        ys = [transform[5] + r * transform[4] for r, c in contour]
        axes[5].plot(xs, ys, 'b-', linewidth=1.5)

    # Parameter summary
    p = result['params']
    pre_str = "  pre-filter=ON" if p.get('pre_event_filter') else ""
    param_str = (f"σ={p['canny_sigma']}  lo={p['canny_low']}  "
                 f"hi={p['canny_high']}  "
                 f"ero_σ={p['erosion_threshold_sigma']}  "
                 f"min_slope={p.get('min_slope_deg',15)}°  "
                 f"roughness={p.get('dem_roughness_threshold',0.5)}m  "
                 f"min={p['min_area_m2']}m²{pre_str}")
    fig.suptitle(
        f"Avalanche boundary detection — {period_id}\n{param_str}",
        fontsize=10, y=1.01)

    plt.tight_layout()
    tag = f"_{param_tag}" if param_tag else ""
    out_path = out_dir / f"avalanche_boundaries_{period_id}{tag}.png"
    fig.savefig(str(out_path), dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  Boundary plot -> {out_path}")
    return str(out_path)
    
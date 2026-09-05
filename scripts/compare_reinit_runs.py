"""
compare_reinit_runs.py — Compare no-reinit vs avalanche-aware SNOWPACK runs.

Reads two Zarr stores (built from separate SNOWPACK runs), spatially interpolates
per-cluster values onto the DEM grid, and generates:
  - Side-by-side + difference maps for each field at each post-event survey date
  - Time series of release-zone mean values (HS, slab thickness, min Sk38)
  - Summary CSV with daily release-zone stats for both runs

Fields compared:
  HS            — total snow depth (m), direct from Zarr
  slab_thickness — depth from surface to WL interface (m), derived per cluster
  min_sk38      — minimum Sk38 near the WL interface, derived per cluster

Usage:
  python compare_reinit_runs.py \
      --zarr-no-reinit   path/no_reinit/slope_snowpack.zarr \
      --zarr-with-reinit path/with_reinit/slope_snowpack.zarr \
      --smet-dir         outputs/smet \
      --dem-tif          outputs/resampled_1m/dem_1m.tif \
      --release-geojson  data/boundaries/avalanche_release_area.geojson \
      --event-date       2026-01-18 \
      --out-dir          outputs/plots/comparison_v2
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import zarr
from pyproj import Transformer
from scipy.interpolate import griddata

matplotlib.use("Agg")

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

FIELDS = {
    "HS": {
        "label": "Snow depth (m)",
        "cmap": "Blues",
        "vmin": 0.0,
        "vmax": 2.5,
        "diff_cmap": "RdBu",
        "diff_vmax": 0.5,
    },
    "slab_thickness": {
        "label": "Slab thickness (m)",
        "cmap": "YlOrBr",
        "vmin": 0.0,
        "vmax": 1.2,
        "diff_cmap": "RdBu",
        "diff_vmax": 0.4,
    },
    "min_sk38": {
        "label": "Min Sk38",
        "cmap": "RdYlGn",
        "vmin": 0.0,
        "vmax": 3.0,
        "diff_cmap": "RdBu",
        "diff_vmax": 0.5,
    },
}

EPOCH_STR = "hours since "


# ──────────────────────────────────────────────────────────────────────────────
# Zarr helpers
# ──────────────────────────────────────────────────────────────────────────────

def open_zarr(path: Path) -> zarr.Group:
    return zarr.open(str(path), mode="r")


def zarr_times_as_datetimes(z: zarr.Group) -> list[datetime]:
    t_arr = z["time"]
    units = t_arr.attrs.get("units", "")
    if units.startswith(EPOCH_STR):
        epoch = datetime.fromisoformat(units[len(EPOCH_STR):])
    else:
        raise ValueError(f"Unrecognised time units: {units}")
    return [epoch + timedelta(hours=int(h)) for h in t_arr[:]]


def find_time_index(times: list[datetime], target: datetime, tol_h: int = 12) -> int | None:
    for i, t in enumerate(times):
        if abs((t - target).total_seconds()) <= tol_h * 3600:
            return i
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Per-cluster feature extraction at a single time index
# ──────────────────────────────────────────────────────────────────────────────

def extract_cluster_features(z: zarr.Group, t_idx: int) -> pd.DataFrame:
    """
    Return a DataFrame indexed by cluster name with columns:
      HS, slab_thickness, min_sk38

    Reads only the time slice at t_idx.  Uses a simplified WL detection:
    the WL interface is placed at the layer with the minimum sk38 value
    (subject to sk38 < 1.5 and height > 0).  Slab thickness = depth from
    surface to that interface.
    """
    locations = z["location"][:]
    n_loc = len(locations)

    # Read the full time-slice for layer variables — shape (n_loc, n_layers)
    hs_arr = z["HS"][:, t_idx]                        # (n_loc,)
    sk38_arr = z["sk38"][:, t_idx, :]                 # (n_loc, n_layers)
    height_arr = z["height"][:, t_idx, :]             # (n_loc, n_layers) — z from surface, m

    rows = []
    for i in range(n_loc):
        hs = float(hs_arr[i])

        sk38 = sk38_arr[i]
        hgt = height_arr[i]     # negative = below surface (SNOWPACK convention: top > 0 , bottom < 0)

        # Only consider layers with valid data
        valid = (np.isfinite(sk38) & np.isfinite(hgt) & (hgt != 0.0))
        if valid.sum() < 2 or not np.isfinite(hs) or hs <= 0:
            rows.append({"HS": np.nan, "slab_thickness": np.nan, "min_sk38": np.nan})
            continue

        sk_v = sk38[valid]
        hgt_v = hgt[valid]

        # WL interface: layer with minimum sk38 (rough proxy for weakest layer)
        # Restrict to layers within the slab (upper 60% of HS) where sk38 < 2
        # to avoid picking up near-surface new-snow layers.
        slab_zone = (hgt_v > 0) & (hgt_v < hs * 0.6) & (sk_v < 2.0)
        if slab_zone.sum() < 1:
            # Fall back to global minimum
            wl_idx = np.argmin(sk_v)
        else:
            slab_sk = np.where(slab_zone, sk_v, np.inf)
            wl_idx = np.argmin(slab_sk)

        interface_depth = float(hgt_v[wl_idx])   # positive = above surface top (cm from bottom)
        # SNOWPACK height coords: positive means above base, so slab thickness
        # = HS - interface_depth when interface is measured from base.
        # The .pro file stores height as cm from bottom (positive up).
        # Zarr stores the same in metres.
        # interface_depth > 0 means the layer top is that far from the base.
        slab_thick = hs - interface_depth if interface_depth > 0 else np.nan

        # min sk38 in the ±3-layer window around WL interface
        lo = max(0, wl_idx - 2)
        hi = min(len(sk_v), wl_idx + 3)
        min_sk38 = float(np.nanmin(sk_v[lo:hi]))

        rows.append({
            "HS": hs,
            "slab_thickness": max(slab_thick, 0.0) if np.isfinite(slab_thick) else np.nan,
            "min_sk38": min_sk38,
        })

    return pd.DataFrame(rows, index=locations)


# ──────────────────────────────────────────────────────────────────────────────
# Spatial interpolation
# ──────────────────────────────────────────────────────────────────────────────

def load_cluster_coords_utm(smet_dir: Path, dem_crs_wkt: str) -> dict[str, tuple[float, float]]:
    """Read lat/lon from each cluster SMET header and project to DEM CRS (UTM)."""
    tr = Transformer.from_crs("EPSG:4326", dem_crs_wkt, always_xy=True)
    coords: dict[str, tuple[float, float]] = {}
    for smet in sorted(smet_dir.glob("cluster_*.smet")):
        cid = smet.stem
        lat = lon = None
        with open(smet) as fh:
            for line in fh:
                if line.startswith("latitude"):
                    lat = float(line.split("=")[1])
                elif line.startswith("longitude"):
                    lon = float(line.split("=")[1])
                elif line.startswith("[DATA]"):
                    break
        if lat is not None and lon is not None:
            x, y = tr.transform(lon, lat)
            coords[cid] = (x, y)
    return coords


def interpolate_to_grid(
    values: np.ndarray,
    xy: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    method: str = "linear",
) -> np.ndarray:
    """Interpolate scattered (n_pts, 2) xy and values onto grid_x/grid_y meshgrid."""
    pts = np.column_stack([xy[:, 0], xy[:, 1]])
    grid = griddata(pts, values, (grid_x, grid_y), method=method)
    return grid.astype(np.float32)


def build_release_mask(geojson_path: Path, dem_transform, dem_shape) -> np.ndarray:
    """Rasterize a GeoJSON polygon onto the DEM grid. Returns bool array."""
    if not geojson_path.exists():
        return np.ones(dem_shape, dtype=bool)
    gdf = gpd.read_file(geojson_path)
    gdf = gdf.to_crs("EPSG:6342")
    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]
    mask = rasterio.features.rasterize(
        shapes, out_shape=dem_shape, transform=dem_transform,
        fill=0, dtype="uint8",
    )
    return mask.astype(bool)


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_comparison_date(
    no_reinit_grids: dict[str, np.ndarray],
    with_reinit_grids: dict[str, np.ndarray],
    hillshade: np.ndarray,
    release_mask: np.ndarray,
    crown_mask: np.ndarray | None,
    date_str: str,
    out_path: Path,
):
    fields = list(FIELDS.keys())
    n_fields = len(fields)
    # Columns: no_reinit | with_reinit | difference
    fig, axes = plt.subplots(n_fields, 3, figsize=(15, 4 * n_fields))
    fig.suptitle(f"No-reinit vs With-reinit — {date_str}", fontsize=14, fontweight="bold")

    col_titles = ["No Reinit", "With Reinit", "Difference (reinit − no_reinit)"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, pad=4)

    def _show(ax, data, cmap, vmin, vmax, label):
        hs_bg = hillshade if hillshade is not None else np.zeros_like(data)
        ax.imshow(hs_bg, cmap="gray", vmin=0, vmax=255, alpha=0.4,
                  interpolation="nearest")
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.85,
                       interpolation="nearest")
        cb = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cb.set_label(label, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    def _add_contours(ax, mask, color, lw=1.5):
        if mask is not None and mask.any():
            ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=lw)

    for row, fname in enumerate(fields):
        cfg = FIELDS[fname]
        nr = no_reinit_grids.get(fname)
        wr = with_reinit_grids.get(fname)

        if nr is None or wr is None:
            for col in range(3):
                axes[row, col].axis("off")
            continue

        diff = wr - nr
        vmax_diff = cfg["diff_vmax"]

        _show(axes[row, 0], nr, cfg["cmap"], cfg["vmin"], cfg["vmax"], cfg["label"])
        _show(axes[row, 1], wr, cfg["cmap"], cfg["vmin"], cfg["vmax"], cfg["label"])
        _show(axes[row, 2], diff, cfg["diff_cmap"], -vmax_diff, vmax_diff,
              f"Δ {cfg['label']}")

        for col in range(3):
            _add_contours(axes[row, col], release_mask, "red")
            if crown_mask is not None:
                _add_contours(axes[row, col], crown_mask, "yellow", lw=1.2)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path.name}")


def plot_time_series(
    stats_no_reinit: pd.DataFrame,
    stats_with_reinit: pd.DataFrame,
    event_date: datetime,
    out_path: Path,
):
    fields = list(FIELDS.keys())
    fig, axes = plt.subplots(len(fields), 1, figsize=(12, 3.5 * len(fields)), sharex=True)
    fig.suptitle("Release-zone mean — no-reinit vs with-reinit", fontsize=13)

    for ax, fname in zip(axes, fields):
        cfg = FIELDS[fname]
        if fname in stats_no_reinit.columns:
            ax.plot(stats_no_reinit.index, stats_no_reinit[fname],
                    color="steelblue", lw=2, label="No reinit")
        if fname in stats_with_reinit.columns:
            ax.plot(stats_with_reinit.index, stats_with_reinit[fname],
                    color="firebrick", lw=2, linestyle="--", label="With reinit")
        ax.axvline(event_date, color="black", linestyle=":", lw=1.5, label="Event")
        ax.set_ylabel(cfg["label"], fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3)

    axes[-1].set_xlabel("Date")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Compare no-reinit vs with-reinit SNOWPACK runs.")
    ap.add_argument("--zarr-no-reinit",   required=True, type=Path)
    ap.add_argument("--zarr-with-reinit", required=True, type=Path)
    ap.add_argument("--smet-dir",         required=True, type=Path)
    ap.add_argument("--dem-tif",          required=True, type=Path)
    ap.add_argument("--release-geojson",  required=True, type=Path)
    ap.add_argument("--crown-geojson",    default=None,  type=Path)
    ap.add_argument("--event-date",       default="2026-01-18")
    ap.add_argument("--out-dir",          required=True, type=Path)
    ap.add_argument("--days-after",       type=int, default=30,
                    help="Number of days after event to plot (default 30)")
    ap.add_argument("--plot-interval-h",  type=int, default=24,
                    help="Time step between spatial plots in hours (default 24)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "spatial").mkdir(exist_ok=True)

    event_dt = datetime.fromisoformat(args.event_date)
    print(f"Event date: {event_dt.date()}")

    # ── Load DEM ──
    with rasterio.open(args.dem_tif) as src:
        dem = src.read(1).astype(np.float32)
        dem_transform = src.transform
        dem_crs_wkt = src.crs.wkt
        dem_shape = src.shape
        dem_bounds = src.bounds

    # Hillshade for background
    from matplotlib.colors import LightSource
    ls = LightSource(azdeg=315, altdeg=45)
    hillshade = ls.hillshade(dem, vert_exag=2.0)
    hillshade = (hillshade * 255).astype(np.uint8)

    # Build pixel-coordinate meshgrid (for griddata output)
    rows_idx = np.arange(dem_shape[0])
    cols_idx = np.arange(dem_shape[1])
    grid_col, grid_row = np.meshgrid(cols_idx, rows_idx)
    # Convert pixel coords to UTM (for griddata target points)
    xs = dem_bounds.left + (grid_col + 0.5) * dem_transform.a
    ys = dem_bounds.top  + (grid_row + 0.5) * dem_transform.e  # e is negative

    # ── Load cluster coordinates ──
    print("Loading cluster coordinates...")
    coords = load_cluster_coords_utm(args.smet_dir, dem_crs_wkt)
    print(f"  {len(coords)} clusters")

    # ── Load Zarr stores ──
    print("Opening Zarr stores...")
    z_nr = open_zarr(args.zarr_no_reinit)
    z_wr = open_zarr(args.zarr_with_reinit)

    times_nr = zarr_times_as_datetimes(z_nr)
    times_wr = zarr_times_as_datetimes(z_wr)
    print(f"  No-reinit:   {times_nr[0]} → {times_nr[-1]}  ({len(times_nr)} steps)")
    print(f"  With-reinit: {times_wr[0]} → {times_wr[-1]}  ({len(times_wr)} steps)")

    locs_nr = list(z_nr["location"][:])
    locs_wr = list(z_wr["location"][:])

    # ── Release and crown masks ──
    release_mask = build_release_mask(args.release_geojson, dem_transform, dem_shape)
    crown_mask = build_release_mask(args.crown_geojson, dem_transform, dem_shape) \
        if args.crown_geojson and args.crown_geojson.exists() else None

    # ── Build list of comparison dates (post-event, every plot_interval_h) ──
    compare_datetimes: list[datetime] = []
    t = event_dt + timedelta(hours=args.plot_interval_h)
    end_t = event_dt + timedelta(days=args.days_after)
    while t <= end_t:
        compare_datetimes.append(t)
        t += timedelta(hours=args.plot_interval_h)

    print(f"\nComputing features and grids for {len(compare_datetimes)} dates...")

    stats_nr_rows: list[dict] = []
    stats_wr_rows: list[dict] = []

    for dt in compare_datetimes:
        ti_nr = find_time_index(times_nr, dt, tol_h=args.plot_interval_h // 2)
        ti_wr = find_time_index(times_wr, dt, tol_h=args.plot_interval_h // 2)

        if ti_nr is None or ti_wr is None:
            continue

        date_str = dt.strftime("%Y-%m-%d %H:%M")
        date_tag  = dt.strftime("%Y%m%d_%H%M")

        feats_nr = extract_cluster_features(z_nr, ti_nr)
        feats_wr = extract_cluster_features(z_wr, ti_wr)

        # Build scatter arrays — only clusters present in both and in coords dict
        common = set(feats_nr.index) & set(feats_wr.index) & set(coords.keys())
        if len(common) < 10:
            print(f"  {date_str}: only {len(common)} common clusters — skipping")
            continue

        common_sorted = sorted(common)
        xy = np.array([coords[c] for c in common_sorted])

        grids_nr: dict[str, np.ndarray] = {}
        grids_wr: dict[str, np.ndarray] = {}

        for fname in FIELDS:
            vals_nr = feats_nr.loc[common_sorted, fname].values.astype(float)
            vals_wr = feats_wr.loc[common_sorted, fname].values.astype(float)

            ok = np.isfinite(vals_nr) & np.isfinite(vals_wr)
            if ok.sum() < 10:
                continue

            g_nr = interpolate_to_grid(vals_nr[ok], xy[ok], xs, ys)
            g_wr = interpolate_to_grid(vals_wr[ok], xy[ok], xs, ys)

            grids_nr[fname] = g_nr
            grids_wr[fname] = g_wr

        # Release-zone mean stats
        row_nr = {"date": dt}
        row_wr = {"date": dt}
        for fname in FIELDS:
            for run_grids, row in [(grids_nr, row_nr), (grids_wr, row_wr)]:
                g = run_grids.get(fname)
                if g is not None:
                    vals_in = g[release_mask & np.isfinite(g)]
                    row[fname] = float(np.nanmean(vals_in)) if len(vals_in) > 0 else np.nan
                else:
                    row[fname] = np.nan
        stats_nr_rows.append(row_nr)
        stats_wr_rows.append(row_wr)

        # Spatial comparison plot
        out_spatial = args.out_dir / "spatial" / f"compare_{date_tag}.png"
        plot_comparison_date(
            grids_nr, grids_wr, hillshade, release_mask, crown_mask,
            date_str, out_spatial,
        )

    # ── Time series ──────────────────────────────────────────────────────────
    if stats_nr_rows:
        stats_nr = pd.DataFrame(stats_nr_rows).set_index("date")
        stats_wr = pd.DataFrame(stats_wr_rows).set_index("date")

        plot_time_series(stats_nr, stats_wr, event_dt,
                         args.out_dir / "release_zone_timeseries.png")

        # Save summary CSV
        merged = stats_nr.add_suffix("_no_reinit").join(stats_wr.add_suffix("_with_reinit"))
        merged.to_csv(args.out_dir / "release_zone_stats.csv")
        print(f"  → release_zone_stats.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()

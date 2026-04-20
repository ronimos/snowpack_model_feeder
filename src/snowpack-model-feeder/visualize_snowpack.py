"""
Visualize distributed SNOWPACK output for Little Professor.

Generates daily PNG frames in three thematic sets:
  - loading:      HS, dHS/dt
  - stability:    min Sk38, min SSI, min Sn38
  - propagation:  min temperature gradient, accumulated TG, stab deformation rate

Each set lives in its own subdirectory. The HTML flipbook has three tabs
that stay in sync on date.

Usage:
  python visualize_snowpack.py
  python visualize_snowpack.py --end-date 2026-04-15 --min-depth 30

Assemble video for one panel set:
  ffmpeg -r 4 -pattern_type glob -i 'plots/daily_frames/stability/*.png' \
         -vcodec libx264 -pix_fmt yuv420p stability.mp4
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

import xarray as xr
import xsnow

from config import ProjectConfig
from snowpack_io import load_dataset, load_boundaries, kml_to_mask
from snowpack_analysis import (
    split_wl_slab, reduce_scalars_at_time,
    EVENT_DATE,
)

# =====================================================================
# Constants
# =====================================================================

DEFAULT_END_DATE  = "2026-02-01"
DEFAULT_MIN_DEPTH = 30.0   # cm — exclude surface wind slab
DEFAULT_MAX_DEPTH = 300.0  # cm

NOON_HOUR = 12  # UTC

# Stability thresholds for contour lines
SK38_CRIT = 1.0
SSI_CRIT  = 1.5
SN38_CRIT = 1.0
TG_CRIT   = 10.0   # °C/m — active faceting threshold


# Panel definitions: (var_name, label, cmap, vmin, vmax, threshold, invert_good)
# invert_good=True means low values are bad (e.g. stability indices)
PANEL_SETS = {
    "loading": [
        ("HS",      "HS (cm)",       "Blues",   0,   350, None,  False),
        ("dhs_dt",  "dHS/dt (cm/d)", "RdBu_r", -20,  20,  0.0,  False),
    ],
    "stability": [
        ("sk38_min", "Min Sk38",  "RdYlGn", 0, 3.0, SK38_CRIT, True),
        ("ssi_min",  "Min SSI",   "RdYlGn", 0, 4.0, SSI_CRIT,  True),
        ("sn38_min", "Min Sn38",  "RdYlGn", 0, 3.0, SN38_CRIT, True),
    ],
    "propagation": [
        ("tg_min",   "Min TG (°C/m)",    "RdYlGn_r", 0,  30.0, TG_CRIT, False),
        ("atg_mean", "Accum TG (°C/m·d)","YlOrRd",   0, 500.0, None,    False),
        ("sdr_min",  "Min Stab Def Rate","RdYlGn",   0,   6.0, 1.0,     True),
    ],
    "structure": [
        ("HS",            "HS (cm)",               "Blues",    0,  350, None, False),
        ("wl_burial",     "WL burial depth (m)",   "YlOrRd",   None, None, None, False),
        ("slab_thick",    "Slab thickness (m)",    "Blues",    None, None, None, False),
        ("wl_strength",   "WL shear strength (kPa)","RdYlGn",  None, None, None, True),
        ("wl_grain",      "WL grain size (mm)",    "YlOrRd",   None, None, None, False),
        ("slab_density",  "Slab density (kg/m³)",  "Blues",    None, None, None, False),
    ],
    "meloche": [
        ("E_slab",   "Slab E (MPa)",           "Blues",    None, None, None, False),
        ("Lambda",   "Λ elastic length (m)",   "YlOrRd",   None, None, None, False),
        ("sigma_t",  "σ_t tensile str (kPa)",  "Blues",    None, None, None, False),
        ("tau_g",    "τ_g driving stress (Pa)","RdYlGn_r", None, None, None, False),
        ("wl_burial","WL burial depth (m)",    "YlOrRd",   None, None, None, False),
        ("slab_density","Slab density (kg/m³)","Blues",    None, None, None, False),
    ],
}


# =====================================================================
# Load all clusters — lazy via Dask
# =====================================================================

# Boundary helpers delegated to snowpack_io
# load_boundaries() and kml_to_mask() imported above


def load_all_clusters(pro_dir: Path,
                      zarr_path: Path | None = None) -> xr.Dataset:
    """Thin wrapper — delegates to snowpack_io.load_dataset."""
    return load_dataset(pro_dir, zarr_path=zarr_path)


# =====================================================================
# Per-frame reduction
# =====================================================================

def reduce_at_time(ds, timestamp, prev_hs, min_depth_cm,
                   max_depth_cm, wl_method='simple'):
    """Delegates to snowpack_analysis.reduce_scalars_at_time."""
    return reduce_scalars_at_time(
        ds, timestamp, prev_hs, min_depth_cm, max_depth_cm,
        wl_method=wl_method)


# =====================================================================
# Cluster scalars -> 2D grid
# =====================================================================

def scalars_to_grid(values, location_names, cluster_map):
    grid = np.full(cluster_map.shape, np.nan, dtype=np.float32)
    for val, loc in zip(values, location_names):
        try:
            cid = int(str(loc).split('_')[-1])
        except ValueError:
            continue
        mask = cluster_map == cid
        if mask.any():
            grid[mask] = val
    return grid


# =====================================================================
# Single panel-set frame
# =====================================================================

def plot_frame(grids, dem, hillshade, bounds, timestamp,
               panel_set_name, panel_defs, output_path, min_depth_cm,
               boundaries=None, start_zone_mask=None):
    n = len(panel_defs)
    if n <= 3:
        nrows, ncols = 1, n
    else:
        ncols = 3
        nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7 * ncols, 5.5 * nrows))
    axes = np.array(axes).ravel()
    # Hide unused axes for non-divisible panel counts
    for ax in axes[n:]:
        ax.set_visible(False)

    is_event = abs((timestamp - EVENT_DATE).days) <= 1
    title_suffix = "  *** JAN 18 EVENT ***" if is_event else ""

    for ax, (var, label, cmap, vmin, vmax, threshold, _) in zip(axes, panel_defs):
        ax.imshow(hillshade, cmap='gray', extent=bounds, alpha=0.5, aspect='auto')
        grid = grids.get(var)
        if grid is not None:
            # Compute colorscale from start zone only when mask available
            # — outside pixels have thin/patchy snow with extreme values
            if start_zone_mask is not None:
                scale_pixels = grid[start_zone_mask & ~np.isnan(grid)]
            else:
                scale_pixels = grid[~np.isnan(grid)]
            if vmin is None and len(scale_pixels):
                vmin = float(np.percentile(scale_pixels, 2))
            if vmax is None and len(scale_pixels):
                vmax = float(np.percentile(scale_pixels, 98))
            im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax,
                           extent=bounds, alpha=0.85, aspect='auto')
            # Dim pixels outside start zone so they don't distract
            if start_zone_mask is not None:
                outside = np.where(~start_zone_mask,
                                    np.ones_like(grid) * 0.6, np.nan)
                ax.imshow(outside, cmap='gray', vmin=0, vmax=1,
                          extent=bounds, alpha=0.35, aspect='auto')
            plt.colorbar(im, ax=ax, shrink=0.75, label=label)
            if threshold is not None:
                try:
                    ax.contour(grid, levels=[threshold], colors='black',
                               linewidths=0.6, linestyles='--', extent=bounds)
                except Exception:
                    pass
        if boundaries:
            sz = boundaries.get('start_zone')
            if sz:
                ax.plot(sz[0], sz[1], color='limegreen', linewidth=1.2, alpha=0.85)
            ra = boundaries.get('release_area')
            if ra:
                ax.plot(ra[0], ra[1], color='red', linewidth=1.8, alpha=0.9)
            for xs, ys in boundaries.get('release_area_multi', []):
                ax.plot(xs, ys, color='red', linewidth=1.8, alpha=0.9)
        ax.set_title(label, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    depth_str = "stability at WL interface"
    fig.suptitle(
        f"Little Professor  |  {timestamp.strftime('%Y-%m-%d')}  |  "
        f"{panel_set_name}  |  {depth_str}{title_suffix}",
        fontsize=11, y=1.01,
        color='darkred' if is_event else 'black')
    plt.tight_layout()
    fig.savefig(str(output_path), dpi=120, bbox_inches='tight')
    plt.close(fig)


# =====================================================================
# HTML flipbook — three tabs
# =====================================================================

def write_html_flipbook(base_dir: Path, frame_names: list, min_depth: float):
    """Four-tab flipbook: loading / stability / propagation / structure."""
    files_js = '[' + ', '.join(f'"{f}"' for f in frame_names) + ']'
    n = len(frame_names)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Little Professor — SNOWPACK</title>
<style>
  body {{ font-family: sans-serif; background:#1a1a1a; color:#eee;
          display:flex; flex-direction:column; align-items:center; padding:16px; }}
  h2   {{ margin-bottom:6px; font-size:16px; }}
  .tabs {{ display:flex; gap:8px; margin:10px 0; flex-wrap:wrap; justify-content:center; }}
  .tab  {{ padding:6px 18px; font-size:13px; cursor:pointer;
            background:#333; color:#ccc; border:1px solid #555;
            border-radius:4px; }}
  .tab.active {{ background:#555; color:#fff; border-color:#888; }}
  img  {{ max-width:95vw; border:1px solid #555; margin:6px 0; }}
  .controls {{ display:flex; align-items:center; gap:10px; margin:8px 0; }}
  button {{ padding:5px 16px; font-size:13px; cursor:pointer;
             background:#444; color:#eee; border:1px solid #888;
             border-radius:4px; }}
  button:hover {{ background:#666; }}
  input[type=range] {{ width:380px; }}
  #date-label {{ font-size:14px; min-width:110px; text-align:center; }}
  .event-banner {{ background:#8b0000; color:#fff; padding:4px 16px;
                   border-radius:4px; font-size:13px; display:none; }}
  .legend {{ font-size:11px; color:#aaa; margin:4px 0; }}
</style>
</head>
<body>
<h2>Little Professor — SNOWPACK Stability Analysis</h2>
<div style="font-size:12px;color:#aaa;margin-bottom:4px;">
  Stability at WL interface | Jan 18 = skier-triggered D2 event
  <span class="legend"> | <span style="color:limegreen">&#9646;</span> start zone
  | <span style="color:red">&#9646;</span> release area</span>
</div>
<div class="tabs">
  <button class="tab active" onclick="setPanel('loading',this)">&#9601; Loading (HS / dHS/dt)</button>
  <button class="tab" onclick="setPanel('stability',this)">&#9734; Stability (Sk38 / SSI / Sn38)</button>
  <button class="tab" onclick="setPanel('propagation',this)">&#x2605; Propagation (TG / ATG / SDR)</button>
  <button class="tab" onclick="setPanel('structure',this)">&#x2736; Structure (WL / Slab)</button>
  <button class="tab" onclick="setPanel('meloche',this)">&#x2605; Meloche (E / Λ / τ_g)</button>
</div>
<div class="event-banner" id="event-banner">&#9888; JAN 18 EVENT WINDOW &#9888;</div>
<img id="frame" src="loading/{frame_names[0]}" alt="frame">
<div class="controls">
  <button onclick="prev()">&#9664; Prev</button>
  <button onclick="togglePlay()" id="play-btn">&#9654; Play</button>
  <button onclick="next()">Next &#9654;</button>
  <span id="date-label">{frame_names[0].replace('.png','')}</span>
</div>
<div class="controls">
  <input type="range" id="slider" min="0" max="{n-1}"
         value="0" oninput="goTo(parseInt(this.value))">
</div>
<div class="controls">
  <span style="font-size:12px">Speed:</span>
  <button onclick="setSpeed(600)">Slow</button>
  <button onclick="setSpeed(200)">Normal</button>
  <button onclick="setSpeed(80)">Fast</button>
</div>
<script>
const frames = {files_js};
let idx=0, playing=false, interval=null, delay=200, panel='loading';

function show(i) {{
  idx = (i+frames.length)%frames.length;
  const f = frames[idx];
  document.getElementById('frame').src = panel+'/'+f;
  document.getElementById('date-label').textContent = f.replace('.png','');
  document.getElementById('slider').value = idx;
  const d = f.replace('.png','');
  const near = d >= '20260117' && d <= '20260119';
  document.getElementById('event-banner').style.display = near ? 'block' : 'none';
}}
function next()  {{ show(idx+1); }}
function prev()  {{ show(idx-1); }}
function goTo(i) {{ show(i); }}
function setSpeed(ms) {{ delay=ms; if(playing){{ stop(); start(); }} }}
function start() {{ interval=setInterval(next,delay); playing=true;
                    document.getElementById('play-btn').textContent='&#9646;&#9646; Pause'; }}
function stop()  {{ clearInterval(interval); playing=false;
                    document.getElementById('play-btn').textContent='&#9654; Play'; }}
function togglePlay() {{ playing ? stop() : start(); }}
function setPanel(p, btn) {{
  panel=p;
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  btn.classList.add('active');
  show(idx);
}}
document.addEventListener('keydown', e=>{{
  if(e.key==='ArrowRight') next();
  if(e.key==='ArrowLeft')  prev();
  if(e.key===' ') {{ e.preventDefault(); togglePlay(); }}
  if(e.key==='1') setPanel('loading',     document.querySelectorAll('.tab')[0]);
  if(e.key==='2') setPanel('stability',   document.querySelectorAll('.tab')[1]);
  if(e.key==='3') setPanel('propagation', document.querySelectorAll('.tab')[2]);
  if(e.key==='4') setPanel('structure',   document.querySelectorAll('.tab')[3]);
  if(e.key==='5') setPanel('meloche',     document.querySelectorAll('.tab')[4]);
}});
</script>
</body>
</html>"""

    out = base_dir / "index.html"
    out.write_text(html)
    print(f"HTML flipbook -> {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate daily SNOWPACK frames — three thematic panel sets")
    parser.add_argument('--project-dir', default='.')
    parser.add_argument('--pro-dir',
                        default='/home/ron/snowpack/little_prof/output')
    parser.add_argument('--end-date', default=DEFAULT_END_DATE)
    parser.add_argument('--zarr-path', type=Path,
                        default=Path('/home/ron/snowpack/little_prof/output/slope_snowpack.zarr'),
                        help='Path to Zarr cache (created on first run)')
    parser.add_argument('--min-depth', type=float, default=DEFAULT_MIN_DEPTH)
    parser.add_argument('--force', action='store_true',
                        help='Regenerate all frames even if cached')
    parser.add_argument('--wl-method', choices=['simple', 'grain_type'],
                        default='simple',
                        help='WL/slab split method: simple=bottom 20%% of HS '
                             '(fast), grain_type=FC/DH grain detection (slow)')
    parser.add_argument('--max-depth', type=float, default=DEFAULT_MAX_DEPTH)
    args = parser.parse_args()

    cfg = ProjectConfig(project_dir=Path(args.project_dir))
    cfg.ensure_dirs()

    base_dir = cfg.plots_dir / "daily_frames"
    base_dir.mkdir(parents=True, exist_ok=True)
    for panel_name in PANEL_SETS:
        (base_dir / panel_name).mkdir(exist_ok=True)

    import rasterio
    with rasterio.open(str(cfg.resampled_dir / "dem_1m.tif")) as src:
        dem = src.read(1).astype(np.float32)
        dem[dem == src.nodata] = np.nan
        transform = src.transform

    cluster_map = np.load(str(cfg.analysis_dir / "cluster_map.npy"))

    # Load boundary overlays (KML + GeoJSON)
    try:
        boundaries = load_boundaries(cfg.project_dir)
        print(f"Boundaries loaded: {list(boundaries.keys())}")
    except Exception as e:
        print(f"Warning: could not load boundaries ({e})")
        boundaries = {}

    # Rasterize start zone KML for colorscale masking
    start_zone_mask_raster = None
    try:
        from analyze_release_zone import kml_to_mask
        start_zone_mask_raster = kml_to_mask(
            cfg.start_zone_kml, dem.shape, transform)
        print(f"Start zone mask: {start_zone_mask_raster.sum()} cells")
    except Exception as e:
        print(f"Warning: could not build start zone mask ({e})")

    bounds = [transform[2],
              transform[2] + dem.shape[1] * transform[0],
              transform[5] + dem.shape[0] * transform[4],
              transform[5]]

    fill_dem  = np.where(np.isnan(dem), np.nanmean(dem), dem)
    hillshade = LightSource(azdeg=315, altdeg=45).hillshade(fill_dem, dx=1.0, dy=1.0)

    ds = load_all_clusters(Path(args.pro_dir), zarr_path=args.zarr_path)
    location_names = ds.coords['location'].values

    all_times = pd.DatetimeIndex(ds.coords['time'].values)
    end_date  = min(pd.Timestamp(args.end_date), all_times.max())

    daily_noons = [
        t for t in pd.date_range(all_times.min().normalize() + pd.Timedelta(days=1),
                                  end_date, freq='1D')
                    .map(lambda d: d.replace(hour=NOON_HOUR))
        if all_times.min() <= t <= all_times.max()
    ]

    print(f"Generating {len(daily_noons)} daily frames "
          f"({daily_noons[0].date()} -> {daily_noons[-1].date()})...")

    frame_names = []
    prev_hs_state = {}

    for i, ts in enumerate(daily_noons):
        date_str  = ts.strftime('%Y%m%d')
        frame_names.append(f"{date_str}.png")

        # Check if all panel set frames already exist
        all_exist = all(
            (base_dir / pname / f"{date_str}.png").exists()
            for pname in PANEL_SETS
        )
        if all_exist and not args.force:
            print(f"  [{i+1:3d}/{len(daily_noons)}] {date_str}: cached")
            # Still need to advance prev_hs_state
            if not prev_hs_state:
                scalars, prev_hs_state = reduce_at_time(
                    ds, ts, {}, args.min_depth, args.max_depth)
            continue

        scalars, prev_hs_state = reduce_at_time(
            ds, ts, prev_hs_state, args.min_depth, args.max_depth,
            wl_method=args.wl_method)

        grids = {var: scalars_to_grid(scalars[var], location_names, cluster_map)
                 for var in scalars}

        for panel_name, panel_defs in PANEL_SETS.items():
            out_path = base_dir / panel_name / f"{date_str}.png"
            plot_frame(grids, dem, hillshade, bounds, ts,
                       panel_name, panel_defs, out_path, args.min_depth,
                       boundaries=boundaries,
                       start_zone_mask=start_zone_mask_raster)

        print(f"  [{i+1:3d}/{len(daily_noons)}] {date_str}: written")

    # Always rebuild flipbook from ALL frames on disk, not just
    # those generated in this run — stays correct across partial runs.
    all_frames = sorted(f.name for f in (base_dir / 'loading').glob('*.png'))
    write_html_flipbook(base_dir, all_frames or frame_names, args.min_depth)
    print(f"\nDone. Open: {base_dir / 'index.html'}")
    print("Keyboard: arrows=prev/next, space=play/pause, 1/2/3=panel tabs")


if __name__ == '__main__':
    main()
    
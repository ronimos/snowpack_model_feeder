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

# Jan 18 event date for annotation
EVENT_DATE = pd.Timestamp("2026-01-18")

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
}


# =====================================================================
# Load all clusters — lazy via Dask
# =====================================================================

def load_all_clusters(pro_dir: Path,
                      zarr_path: Path | None = None) -> xr.Dataset:
    if zarr_path and zarr_path.exists():
        print(f"Loading dataset from Zarr: {zarr_path}...")
        dr = xr.open_zarr(str(zarr_path))
        ds = xsnow.xsnowDataset(dr)
        print(f"  Dataset dims: {dict(ds.dims)}")
        return ds
    pro_files = list(pro_dir.glob("cluster_*.pro"))
    if not pro_files:
        raise FileNotFoundError(f"No cluster .pro files in {pro_dir}")
    print(f"Loading {len(pro_files)} .pro files from {pro_dir}...")
    ds = xsnow.read(str(pro_dir))
    if zarr_path and isinstance(ds, xsnow.xsnowDataset):
        print(f"  Saving to Zarr: {zarr_path}...")
        ds.data.to_zarr(str(zarr_path), mode='w')
        print(f"  Zarr saved.")
    print(f"  Dataset dims: {dict(ds.dims)}")
    return ds


# =====================================================================
# Per-frame reduction
# =====================================================================

def reduce_at_time(ds: xr.Dataset,
                   timestamp: pd.Timestamp,
                   prev_hs: dict,
                   min_depth_cm: float,
                   max_depth_cm: float) -> dict:
    """
    Reduce all variables to per-cluster scalars at one timestep.

    prev_hs: dict mapping location -> HS value at previous timestep,
             used to compute dHS/dt.
    """
    ds_t = ds.sel(time=timestamp, method='nearest')

    z   = ds_t['z']
    in_depth = (z <= -min_depth_cm) & (z >= -max_depth_cm) & (~np.isnan(z))

    def layer_min(var):
        return (ds_t[var].where(in_depth)
                         .min(dim='layer')
                         .squeeze(['slope','realization'])
                         .compute().values)

    def layer_mean(var):
        return (ds_t[var].where(in_depth)
                         .mean(dim='layer')
                         .squeeze(['slope','realization'])
                         .compute().values)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        hs       = ds_t['HS'].squeeze(['slope','realization']).compute().values
        sk38_min = layer_min('sk38')
        ssi_min  = layer_min('ssi')
        sn38_min = layer_min('sn38')
        tg_min   = layer_min('temperature_gradient')
        atg_mean = layer_mean('accumulated_temperature_gradient')
        sdr_min  = layer_min('stab_deformation_rate')

    # dHS/dt in cm/day from previous timestep
    locs = ds.coords['location'].values
    if prev_hs:
        dt_days = (timestamp - prev_hs['time']).total_seconds() / 86400
        if dt_days > 0:
            dhs_dt = (hs - prev_hs['hs']) / dt_days
        else:
            dhs_dt = np.zeros_like(hs)
    else:
        dhs_dt = np.zeros_like(hs)

    return {
        'HS':       hs,
        'dhs_dt':   dhs_dt,
        'sk38_min': sk38_min,
        'ssi_min':  ssi_min,
        'sn38_min': sn38_min,
        'tg_min':   np.abs(tg_min),   # magnitude — SNOWPACK TG can be signed
        'atg_mean': atg_mean,
        'sdr_min':  sdr_min,
    }, {'hs': hs, 'time': timestamp}


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
               panel_set_name, panel_defs, output_path, min_depth_cm):
    n = len(panel_defs)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5))
    if n == 1:
        axes = [axes]

    is_event = abs((timestamp - EVENT_DATE).days) <= 1
    title_suffix = "  *** JAN 18 EVENT ***" if is_event else ""

    for ax, (var, label, cmap, vmin, vmax, threshold, _) in zip(axes, panel_defs):
        ax.imshow(hillshade, cmap='gray', extent=bounds, alpha=0.5, aspect='auto')
        grid = grids.get(var)
        if grid is not None:
            im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax,
                           extent=bounds, alpha=0.85, aspect='auto')
            plt.colorbar(im, ax=ax, shrink=0.75, label=label)
            if threshold is not None:
                try:
                    ax.contour(grid, levels=[threshold], colors='black',
                               linewidths=0.6, linestyles='--', extent=bounds)
                except Exception:
                    pass
        ax.set_title(label, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    depth_str = f"buried >{min_depth_cm:.0f}cm"
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
    """Three-tab flipbook: loading / stability / propagation."""
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
  .tabs {{ display:flex; gap:8px; margin:10px 0; }}
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
</style>
</head>
<body>
<h2>Little Professor — SNOWPACK Stability Analysis</h2>
<div style="font-size:12px;color:#aaa;margin-bottom:4px;">
  buried layers &gt;{min_depth:.0f}cm | Jan 18 = skier-triggered D2 event
</div>
<div class="tabs">
  <button class="tab active" onclick="setPanel('loading',this)">Loading (HS / dHS/dt)</button>
  <button class="tab" onclick="setPanel('stability',this)">Stability (Sk38 / SSI / Sn38)</button>
  <button class="tab" onclick="setPanel('propagation',this)">Propagation (TG / ATG / SDR)</button>
</div>
<div class="event-banner" id="event-banner">JAN 18 EVENT WINDOW</div>
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
const eventDate = '20260118';
let idx=0, playing=false, interval=null, delay=200, panel='loading';

function show(i) {{
  idx = (i+frames.length)%frames.length;
  const f = frames[idx];
  document.getElementById('frame').src = panel+'/'+f;
  document.getElementById('date-label').textContent = f.replace('.png','');
  document.getElementById('slider').value = idx;
  const near = f.replace('.png','') >= '20260117' && f.replace('.png','') <= '20260119';
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
  if(e.key==='1') setPanel('loading', document.querySelectorAll('.tab')[0]);
  if(e.key==='2') setPanel('stability', document.querySelectorAll('.tab')[1]);
  if(e.key==='3') setPanel('propagation', document.querySelectorAll('.tab')[2]);
}});
</script>
</body>
</html>"""

    out = base_dir / "index.html"
    out.write_text(html)
    print(f"HTML flipbook -> {out}")


# =====================================================================
# Main
# =====================================================================

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
        if all_exist:
            print(f"  [{i+1:3d}/{len(daily_noons)}] {date_str}: cached")
            # Still need to advance prev_hs_state
            if not prev_hs_state:
                scalars, prev_hs_state = reduce_at_time(
                    ds, ts, {}, args.min_depth, args.max_depth)
            continue

        scalars, prev_hs_state = reduce_at_time(
            ds, ts, prev_hs_state, args.min_depth, args.max_depth)

        grids = {var: scalars_to_grid(scalars[var], location_names, cluster_map)
                 for var in scalars}

        for panel_name, panel_defs in PANEL_SETS.items():
            out_path = base_dir / panel_name / f"{date_str}.png"
            plot_frame(grids, dem, hillshade, bounds, ts,
                       panel_name, panel_defs, out_path, args.min_depth)

        print(f"  [{i+1:3d}/{len(daily_noons)}] {date_str}: written")

    write_html_flipbook(base_dir, frame_names, args.min_depth)
    print(f"\nDone. Open: {base_dir / 'index.html'}")
    print("Keyboard: arrows=prev/next, space=play/pause, 1/2/3=panel tabs")


if __name__ == '__main__':
    main()
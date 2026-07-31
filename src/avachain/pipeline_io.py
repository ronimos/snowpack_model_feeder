"""
Shared I/O helpers for the distributed SNOWPACK forcing pipeline.

Centralises the repeated patterns for loading the reference DEM,
transport metadata, survey grids, and weather data so that each
pipeline step does not have to re-implement them.
"""

import datetime
import json
import re
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import rasterio

from config import ProjectConfig
from smet_writer import load_and_convert


# =====================================================================
# Survey discovery
# =====================================================================

def parse_survey_date(filename: str) -> datetime.date:
    """Extract date from survey filename like '260114_Professor_PTC_snowHeight.tif'."""
    match = re.match(r'^(\d{6})', filename)
    if not match:
        raise ValueError(f"Cannot parse date from: {filename}")
    ds = match.group(1)
    return datetime.date(2000 + int(ds[:2]), int(ds[2:4]), int(ds[4:6]))


def discover_surveys(cfg: ProjectConfig) -> list:
    """Find all survey files and return sorted (date, path) list."""
    survey_files = sorted(cfg.survey_dir.glob(cfg.survey_glob))
    surveys = [(parse_survey_date(f.name), f) for f in survey_files]
    surveys.sort(key=lambda x: x[0])
    print(f"Found {len(surveys)} surveys:")
    for d, f in surveys:
        print(f"  {d.isoformat()}  {f.name}")
    return surveys


# =====================================================================
# DEM
# =====================================================================

def load_dem(cfg: ProjectConfig) -> Tuple[np.ndarray, object, object]:
    """
    Load the 1m reference DEM.

    Returns
    -------
    dem : 2D float32 array with nodata → NaN
    transform : rasterio Affine transform
    crs : rasterio CRS
    """
    ref_dem_path = cfg.resampled_dir / "dem_1m.tif"
    if not ref_dem_path.exists():
        raise FileNotFoundError(
            f"Reference DEM not found at {ref_dem_path}. Run 'resample' first.")
    with rasterio.open(str(ref_dem_path)) as src:
        dem = src.read(1).astype(np.float32)
        dem[dem == src.nodata] = np.nan
        transform = src.transform
        crs = src.crs
    return dem, transform, crs


# =====================================================================
# Survey grids
# =====================================================================

def load_survey_grids(cfg: ProjectConfig,
                      surveys: list) -> dict:
    """
    Load resampled survey HS grids.

    Parameters
    ----------
    surveys : list of (date, path) from discover_surveys()

    Returns
    -------
    dict mapping datetime.date → 2D float32 HS array

    Raises
    ------
    FileNotFoundError if any expected .npy file is missing.
    """
    grids = {}
    missing = []
    for date, _ in surveys:
        npy = cfg.resampled_dir / f"hs_{date.isoformat()}.npy"
        if npy.exists():
            grids[date] = np.load(str(npy))
        else:
            missing.append(str(npy))

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} survey grid(s) not found — run 'resample' first:\n"
            + "\n".join(f"  {p}" for p in missing))
    return grids


# =====================================================================
# Transport metadata
# =====================================================================

def load_transport_meta(cfg: ProjectConfig) -> list:
    """
    Load transport_metadata.json produced by step_transport.

    Returns
    -------
    list of period dicts

    Raises
    ------
    FileNotFoundError if file does not exist.
    """
    meta_path = cfg.analysis_dir / "transport_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Transport metadata not found at {meta_path}. Run 'transport' first.")
    with open(str(meta_path)) as f:
        return json.load(f)


# =====================================================================
# Weather
# =====================================================================

def load_weather(cfg: ProjectConfig) -> pd.DataFrame:
    """Load and convert weather CSV to SI units with UTC timestamps."""
    wx = load_and_convert(str(cfg.weather_csv), tz_output="UTC")
    print(f"Weather: {wx.index[0]} → {wx.index[-1]}, {len(wx)} hours")
    return wx


def station_hs_at(wx: pd.DataFrame, date: datetime.date,
                  hour_utc: int = 18) -> float:
    """
    Get station HS (in meters) at survey flight time.

    Falls back to the nearest value within a 6-hour window if
    the exact timestamp is missing.
    """
    ts = pd.Timestamp(datetime.datetime.combine(date, datetime.time(hour_utc)))
    val = wx['HS'].asof(ts)
    if pd.isna(val):
        window = wx['HS'].loc[ts - pd.Timedelta(hours=6): ts + pd.Timedelta(hours=6)]
        if len(window) > 0:
            val = window.iloc[len(window) // 2]
    if pd.isna(val):
        print(f"  WARNING: No station HS available for {date}")
        return np.nan
    return val / 100.0  # cm → m


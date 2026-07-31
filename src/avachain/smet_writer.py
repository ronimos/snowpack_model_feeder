"""
Convert CAIC weather station CSV data to SMET format for SNOWPACK forcing.

Handles unit conversions from CAIC raw sensor units to SMET/SI:
  - Temperature: tenths of °F → °C (stored in SMET, offset 273.15 for K)
  - RH: % → % (stored in SMET, multiplier 0.01 for fraction)
  - Wind speed: tenths of mph → m/s
  - Wind direction: degrees (no conversion)
  - ISWR: tenths of W/m² → W/m²
  - Snow height: cm → cm (stored in SMET, multiplier 0.01 for m)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple


# --- Unit conversion functions ---

def tenths_f_to_celsius(val: pd.Series) -> pd.Series:
    """Tenths of °F → °C"""
    return (val / 10.0 - 32.0) * 5.0 / 9.0


def tenths_mph_to_ms(val: pd.Series) -> pd.Series:
    """Tenths of mph → m/s"""
    return val / 10.0 * 0.44704


def tenths_wm2_to_wm2(val: pd.Series) -> pd.Series:
    """Tenths of W/m² → W/m²"""
    return val / 10.0


def tenths_inch_to_cm(val: pd.Series) -> pd.Series:
    """Tenths of inches → cm"""
    return val / 10.0 * 2.54


# --- ILWR parameterization (Brutsaert clear-sky + Unsworth-Monteith cloud correction) ---

def estimate_ilwr(ta_celsius: pd.Series, rh_percent: pd.Series,
                   iswr_wm2: pd.Series, altitude_m: float,
                   latitude: float, longitude: float,
                   timestamps: pd.DatetimeIndex) -> pd.Series:
    """
    Estimate incoming longwave radiation from TA, RH, and ISWR.
    
    Uses Brutsaert (1975) clear-sky emissivity with cloud correction
    based on ratio of measured to potential clear-sky ISWR.
    """
    SIGMA = 5.67e-8  # Stefan-Boltzmann
    ta_k = ta_celsius + 273.15
    
    # Vapor pressure from TA and RH (Magnus formula)
    e_sat = 6.108 * np.exp(17.27 * ta_celsius / (ta_celsius + 237.3))  # hPa
    e_a = e_sat * rh_percent / 100.0  # hPa
    
    # Brutsaert clear-sky emissivity
    eps_clear = 1.24 * (e_a / ta_k) ** (1.0 / 7.0)
    
    # Solar geometry for cloud correction
    doy = timestamps.dayofyear
    hour_utc = timestamps.hour + timestamps.minute / 60.0
    
    # Solar declination
    decl = np.radians(23.45 * np.sin(np.radians((284 + doy) * 360 / 365)))
    lat_rad = np.radians(latitude)
    
    # Hour angle (longitude correction for solar time)
    # longitude=-105.87 → solar time ~7.06 hours behind UTC
    solar_time = hour_utc + longitude / 15.0  # approximate equation of time ignored
    hour_angle = np.radians(15.0 * (solar_time - 12.0))
    
    # Cosine of solar zenith angle
    cos_z = (np.sin(lat_rad) * np.sin(decl) + 
             np.cos(lat_rad) * np.cos(decl) * np.cos(hour_angle))
    cos_z = np.clip(cos_z, 0, 1)
    
    # Clear-sky ISWR at surface (Ineichen-Perez simplified)
    solar_constant = 1361
    # Altitude-corrected atmospheric transmittance
    air_mass = np.where(cos_z > 0.01, 1.0 / cos_z, 50.0)
    air_mass = np.clip(air_mass, 1, 50)
    p_ratio = (1 - 2.25577e-5 * altitude_m) ** 5.25588
    tau = 0.75 * p_ratio  # simplified broadband transmittance
    iswr_clear = solar_constant * cos_z * tau ** air_mass
    iswr_clear = np.maximum(iswr_clear, 0)
    
    # Cloud fraction from clearness index (only when sun is up)
    is_daytime = cos_z > 0.05
    clearness = np.where(
        is_daytime & (iswr_clear > 10),
        np.clip(iswr_wm2 / np.maximum(iswr_clear, 1), 0, 1.2),
        np.nan
    )
    
    # Fill nighttime cloud fraction with interpolation from surrounding daytime
    clearness_series = pd.Series(clearness, index=timestamps)
    clearness_series = clearness_series.interpolate(method='time', limit=12)
    clearness_series = clearness_series.fillna(0.5)  # fallback
    cloud_fraction = np.clip(1.0 - clearness_series.values, 0, 1)
    
    # Cloud correction (Unsworth & Monteith 1975)
    eps_all = eps_clear * (1 + 0.22 * cloud_fraction ** 2.75)
    eps_all = np.clip(eps_all, eps_clear, 1.0)
    
    ilwr = eps_all * SIGMA * ta_k ** 4
    return pd.Series(ilwr, index=ta_celsius.index)


# --- Station configuration ---

@dataclass
class StationConfig:
    station_id: str
    station_name: str
    latitude: float
    longitude: float
    altitude_m: float
    slope_angle: float = 0.0
    slope_azi: float = 0.0
    tz_offset: float = 0.0  # 0 for UTC, -7 for MST
    nodata: float = -999.0


SUMMIT_STATION = StationConfig(
    station_id="CAABT",
    station_name="A-Basin SA-Summit",
    latitude=39.6424,
    longitude=-105.8718,
    altitude_m=3798.3,  # 12462 ft
)

BASE_STATION = StationConfig(
    station_id="CAABM",
    station_name="A-Basin SA-Base",
    latitude=39.6424,
    longitude=-105.8718,
    altitude_m=3554.0,  # 11660 ft
)


def load_and_convert(csv_path: str, tz_output: str = "UTC") -> pd.DataFrame:
    """
    Load CAIC weather CSV and convert to SI/metric units.
    
    Parameters
    ----------
    csv_path : path to weather_data.csv
    tz_output : "UTC" or "MST" for output timestamps
    
    Returns
    -------
    DataFrame with columns: TA (°C), RH (%), VW (m/s), DW (°), ISWR (W/m²), HS (cm)
    """
    df = pd.read_csv(csv_path, parse_dates=["time"], index_col="time")
    
    out = pd.DataFrame(index=df.index)
    out["TA"] = tenths_f_to_celsius(df["temp"])
    out["RH"] = df["rh"].astype(float)
    out["VW"] = tenths_mph_to_ms(df["wspd"])
    out["DW"] = df["wdir"].astype(float)
    out["ISWR"] = tenths_wm2_to_wm2(df["swin"])
    out["HS"] = tenths_inch_to_cm(df["depth"])
    
    # Estimate ILWR from TA, RH, ISWR
    out["ILWR"] = estimate_ilwr(
        out["TA"], out["RH"], out["ISWR"],
        altitude_m=SUMMIT_STATION.altitude_m,
        latitude=SUMMIT_STATION.latitude,
        longitude=SUMMIT_STATION.longitude,
        timestamps=out.index
    )
    
    # CAIC SQL database stores timestamps in UTC.
    # Verified: ISWR=0 at hours 03-14 (= 8 PM - 7 AM MST = nighttime)
    # No conversion needed if tz_output is UTC.
    if tz_output == "MST":
        out.index = out.index - pd.Timedelta(hours=7)
    
    return out


def write_smet(df: pd.DataFrame, output_path: str, 
               config: StationConfig,
               fields: Optional[list] = None,
               ilwr_source: str = "parameterized"):
    """
    Write a SMET 1.1 forcing file for SNOWPACK.
    
    Parameters
    ----------
    df : DataFrame with columns matching SMET field names (TA in °C, RH in %, etc.)
    output_path : path to write .smet file
    config : StationConfig with station metadata
    fields : list of field names to write (default: TA, RH, VW, DW, ISWR, ILWR, HS)
    ilwr_source : description for header comment
    """
    if fields is None:
        fields = ["TA", "RH", "VW", "DW", "ISWR", "ILWR", "HS"]
    
    # Build units_offset and units_multiplier
    # SMET convention: stored_value * multiplier + offset = SI_value
    field_units = {
        "TA":   {"offset": 273.15, "multiplier": 1.0,  "unit": "K"},
        "RH":   {"offset": 0.0,    "multiplier": 0.01, "unit": "-"},
        "VW":   {"offset": 0.0,    "multiplier": 1.0,  "unit": "m/s"},
        "DW":   {"offset": 0.0,    "multiplier": 1.0,  "unit": "°"},
        "ISWR": {"offset": 0.0,    "multiplier": 1.0,  "unit": "W/m2"},
        "ILWR": {"offset": 0.0,    "multiplier": 1.0,  "unit": "W/m2"},
        "HS":   {"offset": 0.0,    "multiplier": 0.01, "unit": "m"},
    }
    
    offsets = " ".join(["0"] + [str(field_units[f]["offset"]) for f in fields])
    multipliers = " ".join(["1"] + [str(field_units[f]["multiplier"]) for f in fields])
    units = " ".join(["-"] + [field_units[f]["unit"] for f in fields])
    
    nodata = config.nodata
    
    with open(output_path, "w") as f:
        f.write("SMET 1.1 ASCII\n")
        f.write("[HEADER]\n")
        f.write(f"station_id       = {config.station_id}\n")
        f.write(f"station_name     = {config.station_name}\n")
        f.write(f"latitude         = {config.latitude:.6f}\n")
        f.write(f"longitude        = {config.longitude:.6f}\n")
        f.write(f"altitude         = {config.altitude_m:.1f}\n")
        f.write(f"nodata           = {nodata}\n")
        f.write(f"tz               = {config.tz_offset}\n")
        f.write(f"units_offset     = {offsets}\n")
        f.write(f"units_multiplier = {multipliers}\n")
        f.write(f"slope_angle      = {config.slope_angle}\n")
        f.write(f"slope_azi        = {config.slope_azi}\n")
        f.write(f"comment          = ILWR {ilwr_source}; generated from CAIC station data\n")
        f.write(f"fields           = timestamp {' '.join(fields)}\n")
        f.write("[DATA]\n")
        
        for ts, row in df.iterrows():
            ts_str = ts.strftime("%Y-%m-%dT%H:%M:%S")
            vals = []
            for field in fields:
                if field in row and not pd.isna(row[field]):
                    vals.append(f"{row[field]:.3f}")
                else:
                    vals.append(f"{nodata:.3f}")
            f.write(f"{ts_str}  {'  '.join(vals)}\n")


def write_grid_smet(df_base: pd.DataFrame, 
                    grid_hs: dict,
                    output_dir: str,
                    base_config: StationConfig,
                    grid_metadata: dict):
    """
    Write per-cell SMET files for distributed SNOWPACK.
    
    Each cell gets the same meteorological forcing (TA, RH, VW, DW, ISWR, ILWR)
    but a different HS time series based on the spatial distribution model.
    
    Parameters
    ----------
    df_base : base DataFrame with meteorological forcing
    grid_hs : dict mapping cell_id -> pd.Series of HS values (cm)
    output_dir : directory for output SMET files
    base_config : StationConfig template (lat/lon/alt will be overridden per cell)
    grid_metadata : dict mapping cell_id -> {lat, lon, alt, slope, aspect}
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for cell_id, hs_series in grid_hs.items():
        cell_df = df_base.copy()
        cell_df["HS"] = hs_series
        
        meta = grid_metadata[cell_id]
        cell_config = StationConfig(
            station_id=str(cell_id),
            station_name=f"grid_{cell_id}",
            latitude=meta["lat"],
            longitude=meta["lon"],
            altitude_m=meta["alt"],
            slope_angle=meta.get("slope", 0.0),
            slope_azi=meta.get("aspect", 0.0),
            tz_offset=base_config.tz_offset,
        )
        
        output_path = out_dir / f"{cell_id}.smet"
        write_smet(cell_df, str(output_path), cell_config)


# =====================================================================
# SMET reader and NWP refill augmentation
# =====================================================================

def read_smet(path: str) -> Tuple[dict, pd.DataFrame]:
    """
    Parse a SMET 1.1 ASCII file into a header dict and timestamped DataFrame.

    Values are returned as stored in the file — not converted to SI units.
    nodata sentinels are replaced with NaN.

    Returns
    -------
    header : dict of all header key=value pairs plus synthesised keys:
        'fields_list'          : list of field names (timestamp excluded)
        'units_offset_list'    : list of float offsets  (timestamp excluded)
        'units_multiplier_list': list of float multipliers (timestamp excluded)
        '_header_order'        : field keys in original file order
    df : DataFrame indexed by timestamp, one column per field
    """
    header: dict = {}
    header_order: list = []
    data_rows: list = []
    in_header = False
    in_data = False

    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped == '[HEADER]':
                in_header, in_data = True, False
                continue
            if stripped == '[DATA]':
                in_header, in_data = False, True
                continue
            if in_header and '=' in stripped:
                key, _, val = stripped.partition('=')
                key = key.strip()
                header[key] = val.strip()
                header_order.append(key)
            elif in_data:
                data_rows.append(stripped.split())

    header['_header_order'] = header_order

    all_fields = header.get('fields', '').split()
    ts_skip = 1 if all_fields and all_fields[0] == 'timestamp' else 0
    data_fields = all_fields[ts_skip:]
    n = len(data_fields)

    def _parse_units(key: str, default: float) -> list:
        raw = [float(x) for x in header.get(key, '').split()]
        # SMET files include a leading placeholder for the timestamp column
        if len(raw) == n + 1:
            raw = raw[1:]
        elif len(raw) > n + 1:
            raw = raw[1:n + 1]
        while len(raw) < n:
            raw.append(default)
        return raw[:n]

    header['fields_list'] = data_fields
    header['units_offset_list'] = _parse_units('units_offset', 0.0)
    header['units_multiplier_list'] = _parse_units('units_multiplier', 1.0)

    nodata = float(header.get('nodata', -999))
    timestamps: list = []
    rows: list = []

    for parts in data_rows:
        if not parts:
            continue
        try:
            ts = pd.Timestamp(parts[0])
        except Exception:
            continue
        vals = []
        for v in parts[1:n + 1]:
            try:
                fv = float(v)
                vals.append(np.nan if fv == nodata else fv)
            except ValueError:
                vals.append(np.nan)
        while len(vals) < n:
            vals.append(np.nan)
        timestamps.append(ts)
        rows.append(vals)

    df = pd.DataFrame(rows, columns=data_fields,
                      index=pd.DatetimeIndex(timestamps, name='timestamp'))
    return header, df


def augment_smet_with_refill(smet_path: str,
                              refill_header: dict,
                              refill_df: pd.DataFrame,
                              output_path: Optional[str] = None,
                              fields: Optional[list] = None) -> list:
    """
    Augment a station-derived SMET with missing fields from an NWP refill SMET.

    Reads the station SMET, identifies which refill fields are absent, resamples
    the NWP data (typically 6-hourly) to station timestamps via linear time
    interpolation, and rewrites the SMET with the merged fields and updated header.

    Parameters
    ----------
    smet_path     : path to the station SMET to augment
    refill_header : header dict from read_smet() on the NWP refill file
    refill_df     : DataFrame from read_smet() on the NWP refill file
    output_path   : destination path (default: overwrite smet_path in place)
    fields        : explicit list of refill fields to add; default adds all
                    fields present in refill but absent from the station SMET

    Returns
    -------
    list of field names that were added (empty if nothing new to add)
    """
    output_path = output_path or smet_path

    station_header, station_df = read_smet(smet_path)

    station_field_set = set(station_header['fields_list'])
    candidates = fields if fields is not None else refill_header['fields_list']
    # HS_meas and HS_mod are excluded — the pipeline's HS field is authoritative.
    # Including HS_meas (always -999 from NWP) causes SNOWPACK to fail when
    # ENFORCE_MEASURED_SNOW_HEIGHTS is TRUE.
    REFILL_EXCLUDE = {'HS_meas', 'HS_mod'}

    add_fields = [f for f in candidates
                  if f in refill_df.columns
                  and f not in station_field_set
                  and f not in REFILL_EXCLUDE]

    if not add_fields:
        return []

    # --- Resample refill (typically 6-hourly) to station timestamps (hourly) ---
    # Build a combined index so that interpolate(method='time') works correctly
    # across both grids, then select only station timestamps.
    # Values outside the refill time range remain NaN and are written as nodata.
    refill_sub = refill_df[add_fields]
    combined_idx = station_df.index.union(refill_sub.index).sort_values()
    refill_up = (refill_sub
                 .reindex(combined_idx)
                 .interpolate(method='time')
                 .reindex(station_df.index))

    # --- Build merged units metadata ---
    rf_fields = refill_header['fields_list']
    rf_offsets = refill_header['units_offset_list']
    rf_mults = refill_header['units_multiplier_list']

    add_offsets = [rf_offsets[rf_fields.index(f)] for f in add_fields]
    add_mults = [rf_mults[rf_fields.index(f)] for f in add_fields]

    merged_fields = station_header['fields_list'] + add_fields
    merged_offsets = station_header['units_offset_list'] + add_offsets
    merged_mults = station_header['units_multiplier_list'] + add_mults

    nodata = float(station_header.get('nodata', -999))

    # Convert to numpy arrays for fast row-by-row write
    station_arr = station_df[station_header['fields_list']].to_numpy()
    refill_arr = refill_up[add_fields].to_numpy()

    def _fmt(v: float) -> str:
        return f"{nodata:.3f}" if np.isnan(v) else f"{v:.3f}"

    with open(output_path, 'w') as f:
        f.write("SMET 1.1 ASCII\n")
        f.write("[HEADER]\n")

        for key in station_header['_header_order']:
            if key == 'fields':
                f.write(f"fields           = timestamp {' '.join(merged_fields)}\n")
            elif key == 'units_offset':
                vals_str = ' '.join(f"{v:g}" for v in merged_offsets)
                f.write(f"units_offset     = 0 {vals_str}\n")
            elif key == 'units_multiplier':
                vals_str = ' '.join(f"{v:g}" for v in merged_mults)
                f.write(f"units_multiplier = 1 {vals_str}\n")
            else:
                f.write(f"{key:<16} = {station_header[key]}\n")

        f.write("[DATA]\n")

        for i, ts in enumerate(station_df.index):
            ts_str = ts.strftime("%Y-%m-%dT%H:%M:%S")
            vals = [_fmt(v) for v in station_arr[i]]
            vals += [_fmt(v) for v in refill_arr[i]]
            f.write(f"{ts_str}  {'  '.join(vals)}\n")

    return add_fields


def fill_smet_gaps_from_refill(smet_path: str,
                                refill_header: dict,
                                refill_df: pd.DataFrame,
                                max_gap_hours: int = 48,
                                output_path: str | None = None) -> dict:
    """
    Fill gaps in an existing SMET file using NWP refill data.

    Two gap types are handled:
      - Missing hourly rows: inserted into a complete hourly index
      - Existing rows where fields are nodata: filled from refill

    HS gaps are filled by linear interpolation between surrounding station
    values — the refill HS_mod is not used since NWP accumulation differs
    significantly from spatially-distributed cluster values.

    All other fields present in both the station SMET and refill are filled
    from the NWP refill, but only across gaps <= max_gap_hours.

    Parameters
    ----------
    max_gap_hours : gaps longer than this are left as nodata

    Returns
    -------
    dict with 'n_rows_inserted' and 'n_rows_filled'
    """
    output_path = output_path or smet_path
    station_header, station_df = read_smet(smet_path)
    nodata = float(station_header.get('nodata', -999))
    station_fields = station_header['fields_list']

    # Reindex to complete hourly sequence — missing rows become all-NaN
    full_idx = pd.date_range(station_df.index[0], station_df.index[-1], freq='1h')
    n_rows_inserted = len(full_idx) - len(station_df)
    station_df = station_df.reindex(full_idx)

    # Fill HS gaps by linear interpolation between surrounding station values.
    # HS_mod from the refill is intentionally not used — NWP accumulation
    # differs significantly from spatially-distributed cluster values.
    if 'HS' in station_fields:
        station_df['HS'] = station_df['HS'].interpolate(
            method='time',
            limit=max_gap_hours,
            limit_direction='forward')

    # Resample refill to hourly at full station index
    combined_idx = full_idx.union(refill_df.index).sort_values()
    refill_up = (refill_df
                 .reindex(combined_idx)
                 .interpolate(method='time')
                 .reindex(full_idx))

    # For all other station fields present in refill, fill NaN rows
    # but only across gaps <= max_gap_hours
    n_rows_filled = 0
    for field in station_fields:
        if field == 'HS':
            continue  # handled above
        if field not in refill_up.columns:
            continue
        col = station_df[field]
        is_gap = col.isna()
        if not is_gap.any():
            continue

        gap_blocks = (is_gap != is_gap.shift()).cumsum()[is_gap]
        for _, block_idx in station_df[is_gap].groupby(gap_blocks).groups.items():
            if len(block_idx) <= max_gap_hours:
                station_df.loc[block_idx, field] = refill_up.loc[block_idx, field]
                n_rows_filled += len(block_idx)

    # Capture array AFTER all fills
    merged_offsets = station_header['units_offset_list']
    merged_mults = station_header['units_multiplier_list']
    station_arr = station_df[station_fields].to_numpy()

    def _fmt(v):
        return f"{nodata:.3f}" if np.isnan(v) else f"{v:.3f}"

    with open(output_path, 'w') as f:
        f.write("SMET 1.1 ASCII\n")
        f.write("[HEADER]\n")
        for key in station_header['_header_order']:
            if key == 'fields':
                f.write(f"fields           = timestamp {' '.join(station_fields)}\n")
            elif key == 'units_offset':
                f.write(f"units_offset     = 0 {' '.join(f'{v:g}' for v in merged_offsets)}\n")
            elif key == 'units_multiplier':
                f.write(f"units_multiplier = 1 {' '.join(f'{v:g}' for v in merged_mults)}\n")
            else:
                f.write(f"{key:<16} = {station_header[key]}\n")
        f.write("[DATA]\n")
        for i, ts in enumerate(station_df.index):
            vals = [_fmt(v) for v in station_arr[i]]
            f.write(f"{ts.strftime('%Y-%m-%dT%H:%M:%S')}  {'  '.join(vals)}\n")

    return {'n_rows_inserted': n_rows_inserted, 'n_rows_filled': n_rows_filled}


# --- Main: convert station CSV to single SMET file ---

if __name__ == "__main__":
    import sys
    
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "weather_data.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "station_forcing.smet"
    
    print(f"Loading {csv_path}...")
    df = load_and_convert(csv_path, tz_output="UTC")
    
    print(f"Converted data shape: {df.shape}")
    print(f"Time range: {df.index[0]} to {df.index[-1]}")
    print(f"\nSample values (SI units):")
    print(df.head().to_string())
    
    # Merge configs: weather from summit, HS from base
    # Use summit config since that's where the forcing weather is measured
    config = StationConfig(
        station_id="ABAS",
        station_name="A-Basin Combined",
        latitude=SUMMIT_STATION.latitude,
        longitude=SUMMIT_STATION.longitude,
        altitude_m=BASE_STATION.altitude_m,  # HS is measured at base
        tz_offset=0,  # UTC
    )
    
    write_smet(df, output_path, config, ilwr_source="Brutsaert+cloud from ISWR")
    print(f"\nWrote {output_path}")

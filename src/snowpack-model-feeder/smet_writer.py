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
from typing import Optional


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
"""
Fetch avalanche observations from CAIC API and match to survey periods.

Queries the CAIC v2 API for observations within the study boundary KML
and maps them to inter-survey periods for transport correction.

Can auto-generate data/boundaries/avalanche_events.json from API data,
or augment an existing manual events file.

Usage:
  # Fetch and generate events file
  python fetch_avalanche_obs.py

  # Fetch with custom date range
  python fetch_avalanche_obs.py --start 2025-12-01 --end 2026-04-01

  # Explore API response structure
  python fetch_avalanche_obs.py --explore

  # Dry run — show what would be written without saving
  python fetch_avalanche_obs.py --dry-run
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests

from config import ProjectConfig

# --- CAIC API ---

CAIC_API_BASE = "https://api.avalanche.state.co.us/api/v2"
OBS_ENDPOINT = f"{CAIC_API_BASE}/avalanche_observations"


def fetch_observations(start_date: datetime, end_date: datetime, page_limit: int = 500) -> list:
    all_matching_data = []
    page = 1
    
    while True:
        # 1. Pass dates to the API so the result set is stable
        params = {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "per_page": page_limit, 
            "page": page   # Correct parameter name
        }
        
        headers = {"Accept": "application/json"}
        response = requests.get(OBS_ENDPOINT, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        # If no more data is returned, we hit the end
        if not data:
            break

        # Check if we've in date range: 
        earliest_obs_date_str = data[-1]['observed_at']
        earliest_obs_date = datetime.fromisoformat(earliest_obs_date_str.replace('Z', ''))
        
                # Check if we've reached the end of the date range
        latest_obs_date_str = data[0]['observed_at']
        latest_obs_date = datetime.fromisoformat(latest_obs_date_str.replace('Z', ''))
        
        print(f"Page {page}: {len(data)} obs, date range {earliest_obs_date.date()} to {latest_obs_date.date()}")
        if start_date <= earliest_obs_date <= end_date or start_date <= latest_obs_date <= end_date:
            all_matching_data.extend(data)

        elif latest_obs_date < start_date:
            break

        # Increment the page number for the next loop
        page += 1

    return all_matching_data

def extract_observation(raw: dict) -> dict:
    """Extract relevant fields from a raw API observation."""
    # Date — try multiple field names
    date_str = None
    for field in ["observed_at", "created_at", "date", "observationDate", "occurred_at",
                  "observation_date", "avalanche_date", "start_date"]:
        val = raw.get(field)
        if val:
            date_str = val
            break

    date_parsed = None
    if date_str and isinstance(date_str, str):
        try:
            date_parsed = datetime.fromisoformat(date_str.replace("Z", "+00:00")).replace(tzinfo=None)
        except (ValueError, TypeError):
            try:
                date_parsed = datetime.strptime(date_str[:10], "%Y-%m-%d")
            except (ValueError, TypeError):
                pass

    return {
        "id": raw.get("id"),
        "date": date_str,
        "date_parsed": date_parsed,
        "latitude": raw.get("latitude"),
        "longitude": raw.get("longitude"),
        "type": raw.get("type"),
        "trigger": raw.get("trigger"),
        "r_size": raw.get("rSize"),
        "d_size": raw.get("dSize"),
        "aspect": raw.get("aspect"),
        "elevation": raw.get("elevation"),
        "elevation_band": raw.get("elevationBand"),
        "area_name": raw.get("areaName"),
    }


# --- Spatial filtering ---

def filter_to_boundary(observations: list, kml_path: str, buffer_m: float = 200) -> list:
    """
    Filter observations to those within or near the KML boundary polygon.

    Uses a buffer around the polygon since reported avalanche locations
    may not be perfectly accurate.
    """
    from clustering import parse_kml_polygon

    try:
        from pyproj import Transformer
        from shapely.geometry import Point, Polygon
        from shapely.ops import transform as shapely_transform
        import functools
    except ImportError:
        print("WARNING: pyproj/shapely not installed, skipping spatial filter")
        return observations

    coords_lonlat = parse_kml_polygon(kml_path)

    # Create polygon in lon/lat
    poly_lonlat = Polygon(coords_lonlat)

    # Buffer in meters requires projection to UTM
    proj_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32613", always_xy=True)
    proj_to_lonlat = Transformer.from_crs("EPSG:32613", "EPSG:4326", always_xy=True)

    poly_utm = shapely_transform(
        functools.partial(lambda t, x, y, z=None: t.transform(x, y), proj_to_utm),
        poly_lonlat)

    # Buffer the polygon
    poly_buffered = poly_utm.buffer(buffer_m)

    # Convert back to lon/lat for point-in-polygon test
    poly_buffered_lonlat = shapely_transform(
        functools.partial(lambda t, x, y, z=None: t.transform(x, y), proj_to_lonlat),
        poly_buffered)

    matched = []
    for obs in observations:
        lat = obs.get("latitude")
        lon = obs.get("longitude")
        if lat is None or lon is None:
            continue
        try:
            pt = Point(float(lon), float(lat))
            if poly_buffered_lonlat.contains(pt):
                matched.append(obs)
        except (ValueError, TypeError):
            continue

    return matched


# --- Map to survey periods ---

def map_to_survey_periods(observations: list, survey_dates: list,
                            flight_hour_utc: int = 18) -> dict:
    """
    Map each observation to the survey period it falls within.

    Returns dict mapping period_id -> list of observations.
    """
    import datetime as dt
    # Build period intervals
    periods = []
    for i in range(len(survey_dates) - 1):
        d_a = survey_dates[i]
        d_b = survey_dates[i + 1]
        t_a = dt.datetime.combine(d_a, dt.time(hour=flight_hour_utc))
        t_b = dt.datetime.combine(d_b, dt.time(hour=flight_hour_utc))
        pair_id = f"{d_a.isoformat()}__{d_b.isoformat()}"
        periods.append((pair_id, t_a, t_b))

    period_obs = {pid: [] for pid, _, _ in periods}

    for obs in observations:
        obs_date = obs.get("date_parsed")
        if obs_date is None:
            continue

        if obs_date.tzinfo is not None:
            obs_date = obs_date.replace(tzinfo=None)
            
        for pid, t_a, t_b in periods:
            if t_a <= obs_date <= t_b:
                period_obs[pid].append(obs)
                break

    return period_obs


def generate_events_json(period_obs: dict, existing_events: list = None) -> list:
    """
    Generate avalanche_events.json entries from API observations.

    Merges with existing manual entries (manual takes precedence).
    """
    existing_periods = set()
    if existing_events:
        existing_periods = {ev.get("period") for ev in existing_events}

    new_events = list(existing_events or [])

    for pair_id, obs_list in period_obs.items():
        if not obs_list:
            continue
        if pair_id in existing_periods:
            continue  # don't overwrite manual entries

        # Use the largest event in this period
        best = max(obs_list, key=lambda o: _size_score(o))

        event = {
            "period": pair_id,
            "timestamp": best["date"] if best.get("date") else None,
            "size": _format_size(best),
            "trigger": best.get("trigger", "unknown"),
            "source": "CAIC API",
            "n_observations": len(obs_list),
            "notes": f"Auto-fetched from CAIC API. "
                     f"{len(obs_list)} observation(s) in this period. "
                     f"Location accuracy approximate.",
        }

        # Include all observations as sub-entries for reference
        if len(obs_list) > 1:
            event["all_observations"] = [
                {
                    "date": o.get("date"),
                    "size": _format_size(o),
                    "trigger": o.get("trigger"),
                    "aspect": o.get("aspect"),
                    "elevation": o.get("elevation"),
                }
                for o in obs_list
            ]

        new_events.append(event)

    return new_events


def _size_score(obs: dict) -> float:
    """Numeric score for sorting by avalanche size."""
    score = 0
    for field, prefix in [("d_size", "D"), ("r_size", "R")]:
        val = obs.get(field)
        if val:
            try:
                score += float(str(val).replace(prefix, ""))
            except (ValueError, TypeError):
                pass
    return score


def _format_size(obs: dict) -> str:
    """Format size string from observation."""
    parts = []
    if obs.get("d_size"):
        parts.append(f"D{obs['d_size']}")
    if obs.get("r_size"):
        parts.append(f"R{obs['r_size']}")
    return ", ".join(parts) if parts else "unknown"


# --- CLI ---

def main():
    parser = argparse.ArgumentParser(
        description="Fetch CAIC avalanche observations for survey periods")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--project-dir", default=".", help="Project root")
    parser.add_argument("--explore", action="store_true",
                        help="Print raw API response structure")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show results without saving")
    parser.add_argument("--buffer", type=float, default=200,
                        help="Spatial buffer around KML polygon (meters, default=200)")
    args = parser.parse_args()

    cfg = ProjectConfig(project_dir=Path(args.project_dir))

    # Date range from survey files or CLI
    import re
    survey_files = sorted(cfg.survey_dir.glob(cfg.survey_glob))
    survey_dates = []
    for f in survey_files:
        match = re.match(r'^(\d{6})', f.name)
        if match:
            ds = match.group(1)
            survey_dates.append(datetime(2000 + int(ds[:2]), int(ds[2:4]), int(ds[4:6])).date())
    survey_dates.sort()

    if args.start:
        start = datetime.strptime(args.start, "%Y-%m-%d")
    elif survey_dates:
        start = datetime.combine(survey_dates[0], datetime.min.time())
    else:
        start = datetime.now() - timedelta(days=90)

    if args.end:
        end = datetime.strptime(args.end, "%Y-%m-%d")
    elif survey_dates:
        end = datetime.combine(survey_dates[-1], datetime.min.time())
    else:
        end = datetime.now()

    print(f"Fetching CAIC observations: {start.date()} to {end.date()}")
    print(f"Survey dates: {len(survey_dates)} ({survey_dates[0]} to {survey_dates[-1]})"
          if survey_dates else "No survey dates found")

    # Fetch
    try:
        raw_obs = fetch_observations(start, end)
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
        sys.exit(1)

    print(f"Fetched {len(raw_obs)} total observations from CAIC API")

    if args.explore:
        if raw_obs:
            print("\n--- Response Structure ---")
            first = raw_obs[0]
            for k, v in first.items():
                val_str = str(v)[:80]
                print(f"  {k}: {val_str}")
        return

    # Extract
    observations = [extract_observation(r) for r in raw_obs]

    # Spatial filter to KML boundary
    kml_path = str(cfg.boundary_kml)
    if Path(kml_path).exists():
        n_before = len(observations)
        observations = filter_to_boundary(observations, kml_path, buffer_m=args.buffer)
        print(f"Spatial filter ({args.buffer}m buffer): {n_before} → {len(observations)} observations")
    else:
        print(f"WARNING: No KML boundary at {kml_path}, skipping spatial filter")

    if not observations:
        print("No observations matched the study area.")
        return

    # Map to survey periods
    period_obs = map_to_survey_periods(observations, survey_dates, cfg.flight_hour_utc)

    print(f"\nObservations by survey period:")
    for pid, obs_list in sorted(period_obs.items()):
        if obs_list:
            sizes = [_format_size(o) for o in obs_list]
            triggers = [o.get("trigger", "?") or "Unknown" for o in obs_list]
            print(f"  {pid}: {len(obs_list)} obs — {', '.join(sizes)} [{', '.join(triggers)}]")

    # Load existing manual events
    events_path = cfg.avalanche_events_path
    existing = []
    if events_path.exists():
        with open(str(events_path)) as f:
            existing = json.load(f)
        print(f"\nLoaded {len(existing)} existing manual event(s)")

    # Generate merged events
    events = generate_events_json(period_obs, existing)
    n_new = len(events) - len(existing)

    if args.dry_run:
        print(f"\n--- Would write {len(events)} events ({n_new} new) ---")
        print(json.dumps(events, indent=2, default=str))
        return

    # Save
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(events_path), 'w') as f:
        json.dump(events, f, indent=2, default=str)
    print(f"\nSaved {len(events)} events ({n_new} new) to {events_path}")
    print("Run 'python batch_process.py avalanche' to delineate and correct transport")


if __name__ == "__main__":
    main()


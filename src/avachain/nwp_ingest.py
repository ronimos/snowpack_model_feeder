"""
nwp_ingest.py — Extend SMET files with CAIC WRF forecast rows to T+72h.

NWP forecast mode (docs/operational_mode_design_concept.md §3):
after the daily forward step has advanced to today (T+0, observed data),
this module appends CAIC WRF forecast rows to the cluster SMET files so
SNOWPACK can run through T+72h and produce three stability snapshots
(T+0, T+1, T+2) for the scenario ensemble.

Data source: CAIC WRF model — the closest WRF grid point to the study site
is already used for the historical SMET forcing, so the forecast extension
is a direct continuation of the same forcing series (no re-downscaling needed).

NWP rows are flagged in the SMET with a comment (NWP_FLAG) so they can be
stripped and replaced when observed data arrives (rolling convergence, §3.3).

TODO (future ensemble work):
  - Add HRRR / NAM / NBM as secondary NWP sources alongside CAIC WRF to
    form a small met-forcing ensemble (widens the uncertainty envelope at T+2).
  - Weight ensemble members by past forecast skill (archived in
    outputs/forecast_verification/).

Usage:
    python nwp_ingest.py                    # fetch latest CAIC WRF, extend all SMETs
    python nwp_ingest.py --lead-hours 48    # only T+0..T+48
    python nwp_ingest.py --dry-run          # print rows, don't write

Called by run_daily.sh after the observed-data SMET append (smet_append.py).

TODO: implement _fetch_caic_wrf() — replace stub with the real CAIC WRF
      API / file pull once endpoint and credentials are confirmed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SMET_DIR_DEFAULT = Path(__file__).parent.parent.parent / "snowpack" / "little_prof" / "input"

# NWP rows appended to SMET are tagged with this marker so prune_nwp_rows()
# can find and strip them when observed data replaces them.
NWP_FLAG = "# NWP_FORECAST CAIC_WRF"


# ---------------------------------------------------------------------------
# CAIC WRF fetch — stub
# ---------------------------------------------------------------------------

def _fetch_caic_wrf(since: pd.Timestamp, lead_hours: int) -> pd.DataFrame:
    """
    TODO: fetch CAIC WRF forecast for the study site's closest grid point.

    The same WRF grid point already used for historical SMET forcing should
    be used here — the forecast is a direct continuation of that series.
    Confirm the grid point coordinates / index with the CAIC WRF output files
    and wire in the actual fetch (file path, API call, or shared NFS mount).

    Returns a DataFrame with columns:
        timestamp (UTC, hourly), TA (°C), RH (%), VW (m/s), DW (°),
        ISWR (W/m²), ILWR (W/m²), PSUM (mm/h)
    Sorted ascending; no HS column (frozen at last observed per cluster).
    """
    print(f"    [nwp_ingest] STUB: would fetch CAIC WRF from {since} +{lead_hours}h")
    return pd.DataFrame(columns=["timestamp", "TA", "RH", "VW", "DW",
                                  "ISWR", "ILWR", "PSUM"])


# ---------------------------------------------------------------------------
# SMET NWP-row management
# ---------------------------------------------------------------------------

def prune_nwp_rows(smet_path: Path) -> int:
    """Remove previously appended NWP forecast rows from a SMET file.

    Rows are identified by NWP_FLAG written as a comment immediately before
    the forecast block. Returns the number of data rows removed.
    """
    with open(smet_path) as f:
        lines = f.readlines()

    clean = []
    in_nwp_block = False
    removed = 0
    for line in lines:
        if NWP_FLAG in line:
            in_nwp_block = True
            continue
        if in_nwp_block:
            if line.strip() and not line.startswith("#"):
                removed += 1
                continue
            else:
                in_nwp_block = False
        clean.append(line)

    if removed:
        with open(smet_path, "w") as f:
            f.writelines(clean)
    return removed


def append_nwp_rows(smet_path: Path,
                    forecast: pd.DataFrame,
                    dry_run: bool = False) -> int:
    """Append NWP forecast rows to a SMET file, flagged for later pruning."""
    if forecast.empty:
        return 0

    from smet_append import _read_smet_last_timestamp, _read_smet_fields, _format_smet_row
    last_ts = _read_smet_last_timestamp(smet_path)
    fields = _read_smet_fields(smet_path)
    if not fields:
        return 0

    fc = forecast.set_index("timestamp").sort_index()
    if last_ts is not None:
        fc = fc[fc.index > last_ts]
    if fc.empty:
        return 0

    if dry_run:
        print(f"    {smet_path.name}: would append {len(fc)} WRF rows")
        return 0

    lines = [f"{NWP_FLAG}\n"]
    for ts, row in fc.iterrows():
        lines.append(_format_smet_row(ts, row.to_dict(), fields) + "\n")

    with open(smet_path, "a") as f:
        f.writelines(lines)
    return len(fc)


def extend_all_smets(smet_dir: Path,
                     forecast: pd.DataFrame,
                     dry_run: bool = False) -> int:
    """Prune stale NWP rows then append the fresh WRF forecast to all SMETs."""
    smet_files = sorted(smet_dir.glob("cluster_*.smet"))
    if not smet_files:
        print(f"    No SMET files in {smet_dir}")
        return 0

    total_pruned = 0
    total_appended = 0
    for p in smet_files:
        pruned = 0 if dry_run else prune_nwp_rows(p)
        appended = append_nwp_rows(p, forecast, dry_run=dry_run)
        total_pruned += pruned
        total_appended += appended

    print(f"    NWP extend: {len(smet_files)} SMETs — "
          f"{total_pruned} stale rows pruned, {total_appended} WRF rows written")
    return total_appended


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extend SMET files with CAIC WRF forecast rows")
    parser.add_argument("--smet-dir", type=Path, default=SMET_DIR_DEFAULT)
    parser.add_argument("--lead-hours", type=int, default=72,
                        help="Forecast horizon in hours (default: 72)")
    parser.add_argument("--since", default=None,
                        help="Forecast start timestamp (default: last SMET row)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    since = (pd.Timestamp(args.since, tz="UTC") if args.since
             else pd.Timestamp("now", tz="UTC").floor("h"))

    forecast = _fetch_caic_wrf(since=since, lead_hours=args.lead_hours)
    extend_all_smets(args.smet_dir, forecast, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

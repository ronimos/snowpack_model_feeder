#!/bin/bash
# run_daily.sh — Daily forward mode: observed data → SNOWPACK → scenarios → com1DFA
#
# Runs each morning after the previous day's station data is available.
# Between UAS surveys the spatial HS distribution stays frozen at the last
# survey; only stratigraphy (sintering, TG metamorphism, new snow loading) evolves.
#
# Three-phase execution:
#   Phase 1 — Observed data (T+0):
#     aws_ingest   → pull latest station hourly records
#     smet_append  → append new rows to cluster SMETs (HS frozen at last survey)
#     SNOWPACK     → advance one day from restart files
#     zarr_append  → add today's timesteps to the Zarr store        [TODO]
#     analyze      → extract Sk38, SSI, Λ, τ_g snapshot
#     scenarios    → trigger selection, release polygons, depth rasters
#     com1DFA      → runout ensemble
#
#   Phase 2 — NWP forecast (T+1, T+2):
#     nwp_ingest   → fetch CAIC WRF, extend SMETs to T+72h
#     SNOWPACK     → run to T+72h from today's restart
#     analyze      → extract snapshots at T+0, T+1, T+2
#     scenarios    → scenario ensembles at each snapshot (wider size_factor at T+2)
#     com1DFA      → runout envelopes at each lead time
#
#   Phase 3 — Hazard summary:
#     post_summary → traffic-light, trend, key driver              [TODO]
#     smoke_test   → verify pipeline health, alert on failure       [TODO]
#
# Cron schedule (add to crontab with: crontab -e):
#   0 8 * * *   /home/ron/snowpack_model_feeder/run_daily.sh >> /var/log/snowpack_daily.log 2>&1
#   0 10 * * *  /home/ron/snowpack_model_feeder/run_daily.sh --forecast-only >> /var/log/snowpack_forecast.log 2>&1
#
# Usage:
#   ./run_daily.sh                          # full daily run
#   ./run_daily.sh --no-forecast            # observed data + scenarios only (skip NWP)
#   ./run_daily.sh --forecast-only          # re-run NWP phase with latest WRF cycle
#   ./run_daily.sh --snapshot 2026-01-18    # override snapshot date (backfill)
#   ./run_daily.sh --dry-run                # ingest checks only, no SNOWPACK
#
# Prerequisites:
#   - run_full_pipeline.sh completed at least once (cluster map + restart files exist)
#   - .env contains any credentials needed by aws_ingest / nwp_ingest

set -euo pipefail

# --- Configuration ---
PROJECT_DIR=/home/ron/snowpack_model_feeder
SLOPE_SCRIPTS=$PROJECT_DIR/slopes/little_prof
SLOPE_DIR=${SLOPE_DIR:-$PROJECT_DIR/snowpack/little_prof}
SRC=$PROJECT_DIR/src/avachain
PIPELINE="python $SRC/forcing_pipeline.py"
ANALYSIS="python $SRC/analysis_pipeline.py"
AWS_INGEST="python $SRC/aws_ingest.py"
SMET_APPEND="python $SRC/smet_append.py"
NWP_INGEST="python $SRC/nwp_ingest.py"
VENV=$PROJECT_DIR/.venv/bin/activate
LOG_DIR=$PROJECT_DIR/outputs/logs

# Scenario defaults (T+0 — tight, mostly observed forcing)
N_TRIGGERS=5
SIZE_FACTORS_T0="0.85 1.00 1.15"
SIZE_FACTORS_T1="0.70 0.85 1.00 1.15 1.30"
SIZE_FACTORS_T2="0.55 0.70 0.85 1.00 1.15 1.30 1.45"
DEPTH_PCTS="10 50 90"

# Flags
RUN_FORECAST=1
FORECAST_ONLY=0
DRY_RUN=0
SNAPSHOT_DATE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-forecast)    RUN_FORECAST=0; shift ;;
        --forecast-only)  FORECAST_ONLY=1; shift ;;
        --dry-run)        DRY_RUN=1; shift ;;
        --snapshot)       SNAPSHOT_DATE="$2"; shift 2 ;;
        *)                echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# --- Setup ---
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="$LOG_DIR/daily_${TIMESTAMP}.log"

exec > >(tee -a "$LOGFILE") 2>&1

TODAY=$(date +%Y-%m-%d)
SNAPSHOT=${SNAPSHOT_DATE:-$TODAY}
SNAPSHOT_ARG="--snapshot-date $SNAPSHOT"

DRY_FLAG=""
[[ $DRY_RUN -eq 1 ]] && DRY_FLAG="--dry-run"

echo "============================================================"
echo "  Daily Forward Pipeline"
echo "  Date:          $SNAPSHOT"
echo "  Forecast:      $RUN_FORECAST"
echo "  Forecast-only: $FORECAST_ONLY"
echo "  Dry-run:       $DRY_RUN"
echo "  Log:           $LOGFILE"
echo "  Started:       $(date)"
echo "============================================================"
echo ""

cd "$PROJECT_DIR"
source "$VENV"

# Verify cluster map exists
if [[ ! -f outputs/analysis/cluster_map.npy ]]; then
    echo "ERROR: cluster_map.npy not found — run run_full_pipeline.sh first." >&2
    exit 1
fi

SECONDS=0

# ============================================================
# Phase 1 — Observed data (T+0)
# ============================================================

if [[ $FORECAST_ONLY -eq 0 ]]; then

    echo ">>> Phase 1: Observed data (T+0 = $SNAPSHOT)"
    echo ""

    # Step 1a: Pull latest station data
    echo "  [1a] AWS ingest"
    t=$SECONDS
    $AWS_INGEST $DRY_FLAG
    echo "       $(( SECONDS - t ))s"
    echo ""

    # Step 1b: Append new rows to cluster SMETs
    echo "  [1b] SMET append"
    t=$SECONDS
    $SMET_APPEND --until "${SNAPSHOT}T18:00" $DRY_FLAG
    echo "       $(( SECONDS - t ))s"
    echo ""

    # Step 1c: Run SNOWPACK incrementally from restart files
    if [[ $DRY_RUN -eq 0 ]]; then
        echo "  [1c] SNOWPACK incremental (→ ${SNAPSHOT}T18:00)"
        t=$SECONDS
        bash "$SLOPE_SCRIPTS/run_snowpack.sh" "" "${SNAPSHOT}T18:00"
        echo "       $(( SECONDS - t ))s"
        echo ""
    else
        echo "  [1c] SNOWPACK — skipped (dry-run)"
        echo ""
    fi

    # Step 1d: Zarr append
    # TODO: replace full rebuild with incremental append
    # (see docs/TODO.md "Zarr append" and "Fix build_zarr_chunked.py resume logic")
    # For now, the full rebuild is a safe fallback.
    if [[ $DRY_RUN -eq 0 ]]; then
        echo "  [1d] Zarr update  [TODO: switch to append-only]"
        t=$SECONDS
        python "$SRC/build_zarr_chunked.py"
        echo "       $(( SECONDS - t ))s"
        echo ""
    fi

    # Step 1e: Extract stability snapshot
    echo "  [1e] Analyze (snapshot $SNAPSHOT)"
    t=$SECONDS
    [[ $DRY_RUN -eq 0 ]] && $ANALYSIS analyze $SNAPSHOT_ARG
    echo "       $(( SECONDS - t ))s"
    echo ""

    # Step 1f: Generate scenarios
    echo "  [1f] Scenarios T+0 (size_factors: $SIZE_FACTORS_T0)"
    t=$SECONDS
    if [[ $DRY_RUN -eq 0 ]]; then
        $ANALYSIS scenarios $SNAPSHOT_ARG \
            --n-triggers "$N_TRIGGERS" \
            --size-factors $SIZE_FACTORS_T0 \
            --depth-pcts $DEPTH_PCTS
    fi
    echo "       $(( SECONDS - t ))s"
    echo ""

    # Step 1g: com1DFA
    # TODO: wrap com1DFA call here once the com1DFA invocation is scripted
    echo "  [1g] com1DFA T+0   [TODO: wire com1DFA CLI call]"
    echo ""

fi  # end Phase 1

# ============================================================
# Phase 2 — NWP forecast (T+1, T+2)
# ============================================================

if [[ $RUN_FORECAST -eq 1 ]]; then

    echo ">>> Phase 2: NWP forecast (CAIC WRF, T+1 / T+2)"
    echo ""

    T1=$(date -d "$SNAPSHOT + 1 day" +%Y-%m-%d 2>/dev/null || \
         python3 -c "from datetime import date,timedelta; \
                     print((date.fromisoformat('$SNAPSHOT')+timedelta(1)).isoformat())")
    T2=$(date -d "$SNAPSHOT + 2 days" +%Y-%m-%d 2>/dev/null || \
         python3 -c "from datetime import date,timedelta; \
                     print((date.fromisoformat('$SNAPSHOT')+timedelta(2)).isoformat())")

    # Step 2a: Fetch CAIC WRF and extend SMETs to T+72h
    echo "  [2a] NWP ingest (CAIC WRF → ${SNAPSHOT}T18:00 +72h)"
    t=$SECONDS
    $NWP_INGEST --since "${SNAPSHOT}T18:00" --lead-hours 72 $DRY_FLAG
    echo "       $(( SECONDS - t ))s"
    echo ""

    # Step 2b: SNOWPACK T+0 → T+72h
    if [[ $DRY_RUN -eq 0 ]]; then
        echo "  [2b] SNOWPACK forecast run (→ ${T2}T18:00)"
        t=$SECONDS
        bash "$SLOPE_SCRIPTS/run_snowpack.sh" "" "${T2}T18:00"
        echo "       $(( SECONDS - t ))s"
        echo ""
    else
        echo "  [2b] SNOWPACK forecast — skipped (dry-run)"
        echo ""
    fi

    # Step 2c/d: Analyze + scenarios at T+1 and T+2
    for LEAD_DATE in "$T1" "$T2"; do
        if [[ "$LEAD_DATE" == "$T1" ]]; then
            SF=$SIZE_FACTORS_T1; LEAD="T+1"
        else
            SF=$SIZE_FACTORS_T2; LEAD="T+2"
        fi

        echo "  [2c/d] Analyze + scenarios $LEAD ($LEAD_DATE, size_factors: $SF)"
        t=$SECONDS
        if [[ $DRY_RUN -eq 0 ]]; then
            $ANALYSIS analyze --snapshot-date "$LEAD_DATE"
            $ANALYSIS scenarios --snapshot-date "$LEAD_DATE" \
                --n-triggers "$N_TRIGGERS" \
                --size-factors $SF \
                --depth-pcts $DEPTH_PCTS
        fi
        echo "       $(( SECONDS - t ))s"
        echo ""

        # TODO: com1DFA per lead time
        echo "  [2e] com1DFA $LEAD    [TODO: wire com1DFA CLI call]"
        echo ""
    done

fi  # end Phase 2

# ============================================================
# Phase 3 — Hazard summary + smoke test
# ============================================================

echo ">>> Phase 3: Hazard summary + smoke test"
echo ""
echo "  [3a] Post hazard summary   [TODO: implement post_summary.py]"
echo "  [3b] Smoke test            [TODO: implement smoke_test.py]"
echo ""

# ============================================================
# Done
# ============================================================

total=$SECONDS
echo "============================================================"
echo "  Daily pipeline complete"
echo "  Finished: $(date)"
echo "  Total:    $(( total / 3600 ))h $(( total % 3600 / 60 ))m $(( total % 60 ))s"
echo ""
echo "  Outputs:"
echo "    Scenarios:  $PROJECT_DIR/outputs/scenarios/"
echo "    Logs:       $LOGFILE"
echo "============================================================"

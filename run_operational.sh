#!/bin/bash
# run_operational.sh — Operational pipeline after a new UAS survey.
#
# Skips steps that only change when the DEM changes:
#   transport, features, train, avalanche, cluster
#
# What it runs:
#   resample → gap_fill → smet → SNOWPACK → analyze → scenarios
#
# SNOWPACK uses restart files from the previous run, so it only
# simulates from where it left off — not the full season.
#
# Usage:
#   ./run_operational.sh                                    # defaults
#   ./run_operational.sh --survey data/surveys/20260315.tif # specific survey
#   ./run_operational.sh --snapshot 2026-03-14              # analysis date
#   ./run_operational.sh --n-triggers 3                     # scenario params
#
#   # With avalanche reinitialization:
#   ./run_operational.sh --snapshot 2026-01-17 \
#       --reinit --event-date 2026-01-18 \
#       --date-before 2026-01-14 --date-after 2026-01-20
#
#   # With hand-drawn release boundary:
#   ./run_operational.sh --snapshot 2026-01-17 \
#       --reinit --event-date 2026-01-18 \
#       --date-before 2026-01-14 --date-after 2026-01-20 \
#       --reinit-geojson data/boundaries/avalanche_release_area.geojson
#
# Prerequisites:
#   - Full pipeline has been run at least once (cluster map exists)
#   - New survey .tif is in data/surveys/
#
# Runtime: ~1-2 hours (~2-3 with --reinit two-pass SNOWPACK).

set -euo pipefail

# --- Configuration ---
PROJECT_DIR=/home/ron/snowpack_model_feeder
SNOWPACK_DIR=$PROJECT_DIR/snowpack/little_prof
PIPELINE="python $PROJECT_DIR/src/avachain/forcing_pipeline.py"
ANALYSIS="python $PROJECT_DIR/src/avachain/analysis_pipeline.py"
VENV=$PROJECT_DIR/.venv/bin/activate
LOG_DIR=$PROJECT_DIR/outputs/logs

# Defaults
SURVEY_FILE=""
SNAPSHOT_DATE=""
N_TRIGGERS=5
SIZE_FACTORS="0.70 0.85 1.00 1.15 1.30"
DEPTH_PCTS="10 50 90"
REINIT=0
EVENT_DATE=""
DATE_BEFORE=""
DATE_AFTER=""
REINIT_GEOJSON=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --survey)          SURVEY_FILE="$2"; shift 2 ;;
        --snapshot)        SNAPSHOT_DATE="$2"; shift 2 ;;
        --n-triggers)      N_TRIGGERS="$2"; shift 2 ;;
        --size-factors)    SIZE_FACTORS="$2"; shift 2 ;;
        --depth-pcts)      DEPTH_PCTS="$2"; shift 2 ;;
        --reinit)          REINIT=1; shift ;;
        --event-date)      EVENT_DATE="$2"; shift 2 ;;
        --date-before)     DATE_BEFORE="$2"; shift 2 ;;
        --date-after)      DATE_AFTER="$2"; shift 2 ;;
        --reinit-geojson)  REINIT_GEOJSON="$2"; shift 2 ;;
        *)                 echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Validate reinit args
if [[ $REINIT -eq 1 ]]; then
    if [[ -z "$EVENT_DATE" || -z "$DATE_BEFORE" || -z "$DATE_AFTER" ]]; then
        echo "ERROR: --reinit requires --event-date, --date-before, --date-after" >&2
        exit 1
    fi
fi

# --- Setup ---
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="$LOG_DIR/operational_${TIMESTAMP}.log"

exec > >(tee -a "$LOGFILE") 2>&1

echo "============================================================"
echo "  Operational Pipeline Update"
echo "  Started: $(date)"
echo "  Reinit:  $REINIT"
if [[ $REINIT -eq 1 ]]; then
echo "  Event:   $EVENT_DATE ($DATE_BEFORE → $DATE_AFTER)"
fi
echo "  Log:     $LOGFILE"
echo "============================================================"
echo ""

cd "$PROJECT_DIR"
source "$VENV"

# Verify prerequisites
for f in outputs/analysis/cluster_map.npy outputs/analysis/cluster_map.tif; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: $f not found. Run run_full_pipeline.sh first." >&2
        exit 1
    fi
done

SECONDS=0

SNAPSHOT_ARG=""
if [[ -n "$SNAPSHOT_DATE" ]]; then
    SNAPSHOT_ARG="--snapshot-date $SNAPSHOT_DATE"
fi

# --- Step 1: Resample new survey ---
echo ">>> Step 1: Resample"
step1_start=$SECONDS

if [[ -n "$SURVEY_FILE" ]]; then
    echo "    Survey: $SURVEY_FILE"
fi
$PIPELINE resample

step1_elapsed=$((SECONDS - step1_start))
echo "    Done: $(($step1_elapsed / 60))m $(($step1_elapsed % 60))s"
echo ""

# --- Step 2: Gap-fill + SMET ---
echo ">>> Step 2: Gap-fill + SMET generation"
step2_start=$SECONDS

$PIPELINE gap_fill
$PIPELINE smet

step2_elapsed=$((SECONDS - step2_start))
echo "    Done: $(($step2_elapsed / 60))m $(($step2_elapsed % 60))s"
echo ""

# --- Step 3: SNOWPACK ---
echo ">>> Step 3: SNOWPACK simulation"
step3_start=$SECONDS

if [[ $REINIT -eq 1 ]]; then
    # --- Two-pass SNOWPACK with avalanche reinitialization ---

    REINIT_SNAPSHOT="$SNAPSHOT_DATE"
    if [[ -z "$REINIT_SNAPSHOT" ]]; then
        REINIT_SNAPSHOT=$(date -d "$EVENT_DATE - 1 day" +%Y-%m-%d 2>/dev/null || \
                          python3 -c "from datetime import datetime, timedelta; print((datetime.strptime('$EVENT_DATE','%Y-%m-%d')-timedelta(1)).strftime('%Y-%m-%d'))")
    fi

    echo "    Pass 1: incremental → ${EVENT_DATE}T18:00"
    bash "$SNOWPACK_DIR/run_snowpack.sh" "" "${EVENT_DATE}T18:00"

    echo ""
    echo "    Analyzing snowpack at $REINIT_SNAPSHOT for slab thickness..."
    $ANALYSIS analyze --snapshot-date "$REINIT_SNAPSHOT"

    echo ""
    echo "    Reinit: scouring release clusters for $EVENT_DATE event"

    REINIT_CMD="$PIPELINE reinit"
    REINIT_CMD="$REINIT_CMD --event-date $EVENT_DATE"
    REINIT_CMD="$REINIT_CMD --date-before $DATE_BEFORE"
    REINIT_CMD="$REINIT_CMD --date-after $DATE_AFTER"
    REINIT_CMD="$REINIT_CMD --snapshot-date $REINIT_SNAPSHOT"
    if [[ -n "$REINIT_GEOJSON" ]]; then
        REINIT_CMD="$REINIT_CMD --release-geojson $REINIT_GEOJSON"
    fi
    $REINIT_CMD

    echo ""
    echo "    Pass 2: $EVENT_DATE → end of season (scoured clusters only)"
    REINIT_STATS="$PROJECT_DIR/outputs/analysis/reinit_stats_${EVENT_DATE}.json"
    if [[ -f "$REINIT_STATS" ]]; then
        bash "$SNOWPACK_DIR/run_snowpack.sh" "" "" "$REINIT_STATS"
    else
        echo "    WARNING: reinit stats not found, rerunning all clusters"
        bash "$SNOWPACK_DIR/run_snowpack.sh"
    fi
else
    # --- Single-pass SNOWPACK (incremental from restart) ---
    echo "    Incremental from restart files"
    bash "$SNOWPACK_DIR/run_snowpack.sh"
fi

step3_elapsed=$((SECONDS - step3_start))
echo "    Done: $(($step3_elapsed / 60))m $(($step3_elapsed % 60))s"
echo ""

# --- Step 4: Analysis + scenarios ---
echo ">>> Step 4: Analysis + scenario generation"
step4_start=$SECONDS

$ANALYSIS analyze $SNAPSHOT_ARG
$ANALYSIS scenarios $SNAPSHOT_ARG \
    --n-triggers "$N_TRIGGERS" \
    --size-factors $SIZE_FACTORS \
    --depth-pcts $DEPTH_PCTS

step4_elapsed=$((SECONDS - step4_start))
echo "    Done: $(($step4_elapsed / 60))m $(($step4_elapsed % 60))s"
echo ""

# --- Summary ---
total_elapsed=$SECONDS
echo "============================================================"
echo "  Operational update complete"
echo "  Finished: $(date)"
echo "  Total runtime: $(($total_elapsed / 3600))h $(($total_elapsed % 3600 / 60))m $(($total_elapsed % 60))s"
echo ""
echo "  Step 1 (resample):   $(($step1_elapsed / 60))m $(($step1_elapsed % 60))s"
echo "  Step 2 (gap+smet):   $(($step2_elapsed / 60))m $(($step2_elapsed % 60))s"
echo "  Step 3 (SNOWPACK):   $(($step3_elapsed / 60))m $(($step3_elapsed % 60))s"
if [[ $REINIT -eq 1 ]]; then
echo "    (two-pass: run → reinit scour → rerun)"
fi
echo "  Step 4 (analysis):   $(($step4_elapsed / 60))m $(($step4_elapsed % 60))s"
echo ""
echo "  Outputs:"
echo "    Scenarios: $PROJECT_DIR/outputs/scenarios/"
echo "    Log:       $LOGFILE"
echo "============================================================"

#!/bin/bash
# run_full_pipeline.sh — Full end-to-end pipeline rebuild.
#
# Runs everything from raw UAS surveys through to scenario generation.
# Use this when: new DEM, first setup, or major config change.
# For routine operations after a new survey, use run_operational.sh instead.
#
# Usage:
#   ./run_full_pipeline.sh                         # full pipeline
#   ./run_full_pipeline.sh --clean                 # wipe SNOWPACK restart files
#   ./run_full_pipeline.sh --snapshot 2026-01-17   # custom analysis snapshot
#
#   # With avalanche reinitialization (two-pass SNOWPACK):
#   ./run_full_pipeline.sh --clean \
#       --reinit --event-date 2026-01-18 \
#       --date-before 2026-01-14 --date-after 2026-01-20 \
#       --snapshot 2026-01-17
#
# Flags:
#   --clean           Remove SNOWPACK restart files so simulation starts
#                     from bare ground. Required after a DEM change.
#   --reinit          Enable post-avalanche reinitialization (two-pass SNOWPACK).
#   --event-date      Avalanche event date YYYY-MM-DD
#   --date-before     Pre-event survey date YYYY-MM-DD
#   --date-after      Post-event survey date YYYY-MM-DD
#   --reinit-geojson  Hand-drawn release GeoJSON (bypasses auto-detection)
#
# With --reinit, SNOWPACK runs in two passes:
#   Pass 1:  Season start → event date
#   Reinit:  Scour release cluster .sno files (auto-detect or GeoJSON)
#   Pass 2:  Event date → end of season
#
# Runtime: ~4-6 hours (~6-8 with --reinit two-pass SNOWPACK).

set -euo pipefail

# --- Configuration ---
PROJECT_DIR=/home/ron/snowpack_model_feeder
SNOWPACK_DIR=$PROJECT_DIR/snowpack/little_prof
PIPELINE="python $PROJECT_DIR/src/snowpack-model-feeder/forcing_pipeline.py"
ANALYSIS="python $PROJECT_DIR/src/snowpack-model-feeder/analysis_pipeline.py"
VENV=$PROJECT_DIR/.venv/bin/activate
LOG_DIR=$PROJECT_DIR/outputs/logs

# Defaults
SNAPSHOT_DATE=""
CLEAN=0
REINIT=0
EVENT_DATE=""
DATE_BEFORE=""
DATE_AFTER=""
REINIT_GEOJSON=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)           CLEAN=1; shift ;;
        --snapshot)        SNAPSHOT_DATE="$2"; shift 2 ;;
        --reinit)          REINIT=1; shift ;;
        --event-date)      EVENT_DATE="$2"; shift 2 ;;
        --date-before)     DATE_BEFORE="$2"; shift 2 ;;
        --date-after)      DATE_AFTER="$2"; shift 2 ;;
        --reinit-geojson)  REINIT_GEOJSON="$2"; shift 2 ;;
        *)                 shift ;;
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
LOGFILE="$LOG_DIR/full_pipeline_${TIMESTAMP}.log"

exec > >(tee -a "$LOGFILE") 2>&1

echo "============================================================"
echo "  Full Pipeline Rebuild"
echo "  Started: $(date)"
echo "  Clean:   $CLEAN"
echo "  Reinit:  $REINIT"
if [[ $REINIT -eq 1 ]]; then
echo "  Event:   $EVENT_DATE ($DATE_BEFORE → $DATE_AFTER)"
fi
echo "  Log:     $LOGFILE"
echo "============================================================"
echo ""

cd "$PROJECT_DIR"
source "$VENV"

SECONDS=0

SNAPSHOT_ARG=""
if [[ -n "$SNAPSHOT_DATE" ]]; then
    SNAPSHOT_ARG="--snapshot-date $SNAPSHOT_DATE"
fi

# --- Phase 1: Data preparation ---
echo ">>> Phase 1: Data preparation pipeline"
echo "    resample → transport → features → train → avalanche → cluster → gap_fill → smet"
echo ""

phase1_start=$SECONDS

$PIPELINE resample
$PIPELINE transport
$PIPELINE features
$PIPELINE train
$PIPELINE avalanche
$PIPELINE cluster
$PIPELINE gap_fill
$PIPELINE smet

phase1_elapsed=$((SECONDS - phase1_start))
echo ""
echo "    Phase 1 complete: $(($phase1_elapsed / 60))m $(($phase1_elapsed % 60))s"
echo ""

# --- Phase 2: SNOWPACK simulation ---
echo ">>> Phase 2: SNOWPACK simulation"

if [[ $CLEAN -eq 1 ]]; then
    echo "    --clean: removing restart files, .pro, and per-cluster .sno"
    rm -f "$SNOWPACK_DIR"/output/*.sno
    rm -f "$SNOWPACK_DIR"/output/*.pro
    rm -f "$SNOWPACK_DIR"/input/snow/cluster_*.sno
    echo "    Cleaned. SNOWPACK will start from template.sno."
fi

phase2_start=$SECONDS

if [[ $REINIT -eq 1 ]]; then
    # --- Two-pass SNOWPACK with avalanche reinitialization ---

    # Determine snapshot date for slab thickness lookup
    REINIT_SNAPSHOT="$SNAPSHOT_DATE"
    if [[ -z "$REINIT_SNAPSHOT" ]]; then
        # Default: day before event
        REINIT_SNAPSHOT=$(date -d "$EVENT_DATE - 1 day" +%Y-%m-%d 2>/dev/null || \
                          python3 -c "from datetime import datetime, timedelta; print((datetime.strptime('$EVENT_DATE','%Y-%m-%d')-timedelta(1)).strftime('%Y-%m-%d'))")
    fi

    echo ""
    echo "    Pass 1: Season start → ${EVENT_DATE}T18:00"
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
    # --- Single-pass SNOWPACK ---
    echo ""
    bash "$SNOWPACK_DIR/run_snowpack.sh"
fi

phase2_elapsed=$((SECONDS - phase2_start))
echo ""
echo "    Phase 2 complete: $(($phase2_elapsed / 60))m $(($phase2_elapsed % 60))s"
echo ""

# --- Phase 3: Analysis + scenarios ---
echo ">>> Phase 3: Analysis + scenario generation"
echo ""

phase3_start=$SECONDS

$ANALYSIS analyze $SNAPSHOT_ARG
$ANALYSIS scenarios $SNAPSHOT_ARG

phase3_elapsed=$((SECONDS - phase3_start))
echo ""
echo "    Phase 3 complete: $(($phase3_elapsed / 60))m $(($phase3_elapsed % 60))s"
echo ""

# --- Summary ---
total_elapsed=$SECONDS
echo "============================================================"
echo "  Pipeline complete"
echo "  Finished: $(date)"
echo "  Total runtime: $(($total_elapsed / 3600))h $(($total_elapsed % 3600 / 60))m $(($total_elapsed % 60))s"
echo ""
echo "  Phase 1 (data prep):     $(($phase1_elapsed / 60))m $(($phase1_elapsed % 60))s"
echo "  Phase 2 (SNOWPACK):      $(($phase2_elapsed / 60))m $(($phase2_elapsed % 60))s"
echo "  Phase 3 (analysis):      $(($phase3_elapsed / 60))m $(($phase3_elapsed % 60))s"
if [[ $REINIT -eq 1 ]]; then
echo "    (includes two-pass SNOWPACK + reinit scour)"
fi
echo ""
echo "  Outputs:"
echo "    SMET files:    $PROJECT_DIR/outputs/smet/"
echo "    SNOWPACK:      $SNOWPACK_DIR/output/"
echo "    Scenarios:     $PROJECT_DIR/outputs/scenarios/"
echo "    Log:           $LOGFILE"
echo "============================================================"

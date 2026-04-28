#!/bin/bash
# run_full_pipeline.sh — Full end-to-end pipeline rebuild.
#
# Runs everything from raw UAS surveys through to scenario generation.
# Use this when: new DEM, first setup, or major config change.
# For routine operations after a new survey, use run_operational.sh instead.
#
# Usage:
#   ./run_full_pipeline.sh                         # full pipeline, default dates
#   ./run_full_pipeline.sh --end-date 2026-03-31   # custom SNOWPACK end date
#   ./run_full_pipeline.sh --snapshot 2026-01-17    # custom analysis snapshot
#
# Runtime: ~4-6 hours depending on cluster count and SNOWPACK season length.

set -euo pipefail

# --- Configuration ---
PROJECT_DIR=/home/ron/snowpack_model_feeder
SNOWPACK_DIR=/home/ron/snowpack/little_prof
PIPELINE="python $PROJECT_DIR/src/snowpack-model-feeder/pipeline.py"
ANALYSIS="python $PROJECT_DIR/src/snowpack-model-feeder/analysis_pipeline.py"
VENV=$PROJECT_DIR/.venv/bin/activate
LOG_DIR=$PROJECT_DIR/outputs/logs
SNOWPACK_END="${1:---end-date}"
SNAPSHOT_DATE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --end-date)   SNOWPACK_END="$2"; shift 2 ;;
        --snapshot)   SNAPSHOT_DATE="$2"; shift 2 ;;
        *)            shift ;;
    esac
done

# --- Setup ---
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="$LOG_DIR/full_pipeline_${TIMESTAMP}.log"

exec > >(tee -a "$LOGFILE") 2>&1

echo "============================================================"
echo "  Full Pipeline Rebuild"
echo "  Started: $(date)"
echo "  Log:     $LOGFILE"
echo "============================================================"
echo ""

cd "$PROJECT_DIR"
source "$VENV"

SECONDS=0

# --- Phase 1: Data preparation (pipeline.py) ---
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
echo ""

phase2_start=$SECONDS

bash "$SNOWPACK_DIR/run_snowpack.sh"

phase2_elapsed=$((SECONDS - phase2_start))
echo ""
echo "    Phase 2 complete: $(($phase2_elapsed / 60))m $(($phase2_elapsed % 60))s"
echo ""

# --- Phase 3: Analysis + scenarios (analysis_pipeline.py) ---
echo ">>> Phase 3: Analysis + scenario generation"
echo ""

phase3_start=$SECONDS

SNAPSHOT_ARG=""
if [[ -n "$SNAPSHOT_DATE" ]]; then
    SNAPSHOT_ARG="--snapshot-date $SNAPSHOT_DATE"
fi

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
echo ""
echo "  Outputs:"
echo "    SMET files:    $PROJECT_DIR/outputs/smet/"
echo "    SNOWPACK:      $SNOWPACK_DIR/output/"
echo "    Scenarios:     $PROJECT_DIR/outputs/scenarios/"
echo "    Log:           $LOGFILE"
echo "============================================================"


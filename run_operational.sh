#!/bin/bash
# run_operational.sh — Operational pipeline after a new UAS survey.
#
# Skips steps that only change when the DEM changes:
#   transport, features, train, avalanche, cluster
#
# What it runs:
#   resample (new survey only) → gap_fill → smet → SNOWPACK → zarr → analyze → scenarios
#
# SNOWPACK uses restart files (_res.sno) from the previous run, so it only
# simulates from where it left off — not the full season.
#
# Usage:
#   ./run_operational.sh                                    # defaults
#   ./run_operational.sh --survey data/surveys/20260315.tif # specific survey
#   ./run_operational.sh --snapshot 2026-03-14              # analysis date
#   ./run_operational.sh --n-triggers 3                     # scenario params
#
# Prerequisites:
#   - Full pipeline has been run at least once (cluster map, transport model exist)
#   - New survey .tif is in data/surveys/
#
# Runtime: ~1-2 hours (dominated by SNOWPACK incremental run).

set -euo pipefail

# --- Configuration ---
PROJECT_DIR=/home/ron/snowpack_model_feeder
SNOWPACK_DIR=/home/ron/snowpack/little_prof
PIPELINE="python $PROJECT_DIR/src/snowpack-model-feeder/pipeline.py"
ANALYSIS="python $PROJECT_DIR/src/snowpack-model-feeder/analysis_pipeline.py"
VENV=$PROJECT_DIR/.venv/bin/activate
LOG_DIR=$PROJECT_DIR/outputs/logs

# Defaults
SURVEY_FILE=""
SNAPSHOT_DATE=""
N_TRIGGERS=5
SIZE_FACTORS="0.70 0.85 1.00 1.15 1.30"
DEPTH_PCTS="10 50 90"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --survey)       SURVEY_FILE="$2"; shift 2 ;;
        --snapshot)     SNAPSHOT_DATE="$2"; shift 2 ;;
        --n-triggers)   N_TRIGGERS="$2"; shift 2 ;;
        --size-factors) SIZE_FACTORS="$2"; shift 2 ;;
        --depth-pcts)   DEPTH_PCTS="$2"; shift 2 ;;
        *)              echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# --- Setup ---
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="$LOG_DIR/operational_${TIMESTAMP}.log"

exec > >(tee -a "$LOGFILE") 2>&1

echo "============================================================"
echo "  Operational Pipeline Update"
echo "  Started: $(date)"
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

# --- Step 3: SNOWPACK (incremental from restart files) ---
echo ">>> Step 3: SNOWPACK simulation (incremental)"
step3_start=$SECONDS

bash "$SNOWPACK_DIR/run_snowpack.sh"

step3_elapsed=$((SECONDS - step3_start))
echo "    Done: $(($step3_elapsed / 60))m $(($step3_elapsed % 60))s"
echo ""

# --- Step 4: Analysis + scenarios ---
echo ">>> Step 4: Analysis + scenario generation"
step4_start=$SECONDS

SNAPSHOT_ARG=""
if [[ -n "$SNAPSHOT_DATE" ]]; then
    SNAPSHOT_ARG="--snapshot-date $SNAPSHOT_DATE"
fi

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
echo "  Step 4 (analysis):   $(($step4_elapsed / 60))m $(($step4_elapsed % 60))s"
echo ""
echo "  Outputs:"
echo "    Scenarios: $PROJECT_DIR/outputs/scenarios/"
echo "    Log:       $LOGFILE"
echo "============================================================"


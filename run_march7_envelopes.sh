#!/bin/bash
# run_march7_envelopes.sh — Three-day forecast envelopes centered on 2026-03-07.
#
# Builds the zarr from scratch, then generates analyze + scenario outputs
# for T+0 (Mar 7), T+1 (Mar 6), and T+2 (Mar 5) in parallel.
#
# Size factor spreads widen with forecast horizon — higher uncertainty further
# from the observed state:
#   T_2 (Mar 5): 0.55 – 1.00 – 1.70   (widest)
#   T_1 (Mar 6): 0.70 – 1.00 – 1.30
#   T_0 (Mar 7): 0.70 0.85 – 1.00   (narrowest)
#
# max-slab-thickness=3.0 targets natural trigger scenarios.
#
# Usage:
#   ./run_march7_envelopes.sh
#
# Runtime: ~30–60 min depending on cluster count.

set -euo pipefail

# --- Configuration ---
PROJECT_DIR=/home/ron/snowpack_model_feeder
ANALYSIS="$PROJECT_DIR/.venv/bin/python $PROJECT_DIR/src/avachain/analysis_pipeline.py"
VENV=$PROJECT_DIR/.venv/bin/activate
LOG_DIR=$PROJECT_DIR/outputs/logs

N_TRIGGERS=5
DEPTH_PCTS="10 50 90"

# --- Setup ---
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="$LOG_DIR/march7_envelopes_${TIMESTAMP}.log"

exec > >(tee -a "$LOGFILE") 2>&1

echo "============================================================"
echo "  March 7 Forecast Envelopes"
echo "  Started: $(date)"
echo "  Log:     $LOGFILE"
echo "============================================================"
echo ""

cd "$PROJECT_DIR"
source "$VENV"

SECONDS=0

# --- Step 1: Build zarr (must complete before analysis) ---
echo ">>> Step 1: Build zarr"
step1_start=$SECONDS

$ANALYSIS zarr_build

step1_elapsed=$((SECONDS - step1_start))
echo "    Done: $(($step1_elapsed / 60))m $(($step1_elapsed % 60))s"
echo ""

# --- Step 2: Analyze each snapshot date (parallel) ---
echo ">>> Step 2: Analyze snapshot dates (parallel)"
step2_start=$SECONDS

#$ANALYSIS analyze --snapshot-date 2026-03-05 &
#$ANALYSIS analyze --snapshot-date 2026-03-06 &
$ANALYSIS analyze --snapshot-date 2026-03-07 &
p_analyze=$!
# Wait only for the Python jobs — bare `wait` would also wait for the
# tee process substitution (>(tee ...)) and deadlock.
wait $p_analyze

step2_elapsed=$((SECONDS - step2_start))
echo "    Done: $(($step2_elapsed / 60))m $(($step2_elapsed % 60))s"
echo ""

# --- Step 3: Generate scenarios (parallel, each reads its own date's CSVs) ---
echo ">>> Step 3: Generate scenarios (parallel)"
step3_start=$SECONDS

echo ">>> Generate scenarios for T_2 (0.55 1.00 1.45 1.70) 2026-03-07"

$ANALYSIS scenarios \
    --snapshot-date 2026-03-07 --forecast-horizon T_2 \
    --size-factors 0.55 1.00 1.45 1.70 \
    --n-triggers $N_TRIGGERS \
    --depth-pcts $DEPTH_PCTS \
    --max-slab-thickness 3.0 &
p_t2=$!

echo ">>> Generate scenarios for T_1 (0.70 1.00 1.30) 2026-03-07"

$ANALYSIS scenarios \
    --snapshot-date 2026-03-07 --forecast-horizon T_1 \
    --size-factors 0.70 1.00 1.30 \
    --n-triggers $N_TRIGGERS \
    --depth-pcts $DEPTH_PCTS \
    --max-slab-thickness 3.0 &
p_t1=$!

echo ">>> Generate scenarios for T_0 (0.70 0.85 1.00) 2026-03-07"
$ANALYSIS scenarios \
    --snapshot-date 2026-03-07 --forecast-horizon T_0 \
    --size-factors 0.70 0.85 1.00 \
    --n-triggers $N_TRIGGERS \
    --depth-pcts $DEPTH_PCTS \
    --max-slab-thickness 3.0 &
p_t0=$!

wait $p_t2 $p_t1 $p_t0

step3_elapsed=$((SECONDS - step3_start))
echo "    Done: $(($step3_elapsed / 60))m $(($step3_elapsed % 60))s"
echo ""

# --- Summary ---
total_elapsed=$SECONDS
echo "============================================================"
echo "  Envelopes complete"
echo "  Finished: $(date)"
echo "  Total runtime: $(($total_elapsed / 3600))h $(($total_elapsed % 3600 / 60))m $(($total_elapsed % 60))s"
echo ""
echo "  Step 1 (zarr build):  $(($step1_elapsed / 60))m $(($step1_elapsed % 60))s"
echo "  Step 2 (analyze):     $(($step2_elapsed / 60))m $(($step2_elapsed % 60))s"
echo "  Step 3 (scenarios):   $(($step3_elapsed / 60))m $(($step3_elapsed % 60))s"
echo ""
echo "  Outputs:"
echo "    Scenarios: $PROJECT_DIR/outputs/scenarios/2026-03-*/T_{0,1,2}/"
echo "    Log:       $LOGFILE"
echo "============================================================"

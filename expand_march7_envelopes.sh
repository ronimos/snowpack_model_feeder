#!/bin/bash
# expand_march7_envelopes.sh — Expand T_0 envelope to T_1 and T_2 using
# the 2026-03-07 snowpack state (same analysis CSVs as T_0).
#
# Envelopes are cumulative (nested) — each horizon adds two outer size factors:
#
#   T_0 (Mar 7, T+0): 0.85 1.00 1.15                          →  45 scenarios
#   T_1 (Mar 6, T+1): 0.70 0.85 1.00 1.15 1.30                →  75 scenarios
#   T_2 (Mar 5, T+2): 0.55 0.70 0.85 1.00 1.15 1.30 1.45      → 105 scenarios
#
# T_0 is already in outputs/scenarios/2026-03-07/T_0/.
# This script generates T_1 and T_2 from the same March 7 snowpack state
# and moves them to the forecast-date-appropriate paths:
#
#   → outputs/scenarios/2026-03-06/T_1/
#   → outputs/scenarios/2026-03-05/T_2/
#
# Prerequisite: 2026-03-07 analyze step must already have run
# (all_start_zone_features_2026-03-07.csv must exist).

set -euo pipefail

PROJECT_DIR=/home/ron/snowpack_model_feeder
ANALYSIS="$PROJECT_DIR/.venv/bin/python $PROJECT_DIR/src/avachain/analysis_pipeline.py"
VENV=$PROJECT_DIR/.venv/bin/activate
LOG_DIR=$PROJECT_DIR/outputs/logs

mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="$LOG_DIR/expand_march7_envelopes_${TIMESTAMP}.log"

exec > >(tee -a "$LOGFILE") 2>&1

echo "============================================================"
echo "  Expand March 7 Envelopes (T_1, T_2)"
echo "  Started: $(date)"
echo "  Log:     $LOGFILE"
echo "============================================================"
echo ""

cd "$PROJECT_DIR"
source "$VENV"

FEAT_CSV_ALL="$PROJECT_DIR/outputs/analysis/all_start_zone_features_2026-03-07.csv"
FEAT_CSV_GRP="$PROJECT_DIR/outputs/analysis/release_zone_features_2026-03-07.csv"
if [[ ! -f "$FEAT_CSV_ALL" && ! -f "$FEAT_CSV_GRP" ]]; then
    echo "ERROR: no features CSV found for 2026-03-07." >&2
    echo "       Run 'analyze --snapshot-date 2026-03-07' first." >&2
    exit 1
fi
if [[ ! -f "$FEAT_CSV_ALL" ]]; then
    echo "WARNING: using group-level features (release_zone_features_2026-03-07.csv)."
    echo "         Rerun analyze to get full start zone coverage."
    echo ""
fi

SECONDS=0

# --- Generate T_1 and T_2 using March 7 snowpack state (parallel) ---
echo ">>> Generating T_1 and T_2 scenarios from 2026-03-07 snowpack state"
echo "    T_0: size factors 0.85 1.00 1.15  (45 scenarios)"
echo "    T_1: size factors 0.70 0.85 1.00 1.15 1.30  (75 scenarios)"
echo "    T_2: size factors 0.55 0.70 0.85 1.00 1.15 1.30 1.45  (105 scenarios)"
echo ""

$ANALYSIS scenarios \
    --snapshot-date 2026-03-07 --forecast-horizon T_0 \
    --size-factors 0.85 1.00 1.15 \
    --n-triggers 5 \
    --depth-pcts 10 50 90 \
    --max-slab-thickness 3.0 &


$ANALYSIS scenarios \
    --snapshot-date 2026-03-07 --forecast-horizon T_1 \
    --size-factors 0.70 0.85 1.00 1.15 1.30 \
    --n-triggers 5 \
    --depth-pcts 10 50 90 \
    --max-slab-thickness 3.0 &

$ANALYSIS scenarios \
    --snapshot-date 2026-03-07 --forecast-horizon T_2 \
    --size-factors 0.55 0.70 0.85 1.00 1.15 1.30 1.45 \
    --n-triggers 5 \
    --depth-pcts 10 50 90 \
    --max-slab-thickness 3.0 &

wait
gen_elapsed=$SECONDS
echo ""
echo "    Generation done: $(($gen_elapsed / 60))m $(($gen_elapsed % 60))s"
echo ""

# --- Move outputs to forecast-date-appropriate paths ---
echo ">>> Moving outputs to forecast-horizon paths"

SRC_T1="$PROJECT_DIR/outputs/scenarios/2026-03-07/T_1"
DST_T1="$PROJECT_DIR/outputs/scenarios/2026-03-06/T_1"
SRC_T2="$PROJECT_DIR/outputs/scenarios/2026-03-07/T_2"
DST_T2="$PROJECT_DIR/outputs/scenarios/2026-03-05/T_2"

mkdir -p "$(dirname "$DST_T1")" "$(dirname "$DST_T2")"

if [[ -d "$DST_T1" ]]; then
    echo "    Removing existing $DST_T1"
    rm -rf "$DST_T1"
fi
mv "$SRC_T1" "$DST_T1"
echo "    $SRC_T1 → $DST_T1"

if [[ -d "$DST_T2" ]]; then
    echo "    Removing existing $DST_T2"
    rm -rf "$DST_T2"
fi
mv "$SRC_T2" "$DST_T2"
echo "    $SRC_T2 → $DST_T2"

echo ""

# --- Summary ---
total_elapsed=$SECONDS
echo "============================================================"
echo "  Done"
echo "  Finished: $(date)"
echo "  Total runtime: $(($total_elapsed / 60))m $(($total_elapsed % 60))s"
echo ""
echo "  Outputs:"
echo "    T_1 (75 scenarios):  $DST_T1"
echo "    T_2 (105 scenarios): $DST_T2"
echo "    Log: $LOGFILE"
echo "============================================================"

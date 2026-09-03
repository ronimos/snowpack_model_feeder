#!/bin/bash
# run_reinit_comparison.sh — Compare no-reinit vs avalanche-aware SNOWPACK runs.
#
# Runs two complete seasons from a clean slate:
#   1. no_reinit:   full season, no .sno modifications
#   2. with_reinit: phase 1 to event, reinit .sno files, phase 2 to end
#
# Builds a separate Zarr for each, then generates comparison plots including
# HS, slab thickness, Sk38, tau_g, and their differences.
#
# Usage:
#   ./run_reinit_comparison.sh [event_date] [end_date]
#
#   event_date  default: 2026-01-18   (Jan 18 avalanche)
#   end_date    default: 2026-03-31
#
# Survey dates bracketing the Jan 18 event (used for reinit):
#   date_before: 2026-01-14  (last UAS survey before event)
#   date_after:  2026-01-20  (first UAS survey after event)

set -euo pipefail

# ─── Paths ────────────────────────────────────────────────────────────────────
REPO_DIR=/home/ron/snowpack_model_feeder
SNOWPACK_BIN=/home/caic/caic/rtsys/snowpack/exe/snowpack
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/home/caic/caic/rtsys/snowpack/src/snowpack/lib
PYTHON=$REPO_DIR/.venv/bin/python3

SLOPE_DIR=$REPO_DIR/snowpack/little_prof
SMET_DIR=$REPO_DIR/outputs/smet
MASTER_CFG=$REPO_DIR/slopes/little_prof/config/master_config.ini
TEMPLATE_SNO=$SLOPE_DIR/input/snow/template.sno

# ─── Parameters ───────────────────────────────────────────────────────────────
EVENT_DATE="${1:-2026-01-18}"
EDATE_ARG="${2:-2026-03-31}"
EDATE="${EDATE_ARG}T18:00"
DATE_BEFORE="2026-01-14"
DATE_AFTER="2026-01-20"
RELEASE_GEOJSON=$REPO_DIR/data/boundaries/avalanche_release_area.geojson
MAX_JOBS=30

# Comparison output directories
CMP_BASE=$SLOPE_DIR/comparison
NO_REINIT_DIR=$CMP_BASE/no_reinit
WITH_REINIT_DIR=$CMP_BASE/with_reinit

# ─── Season start from first SMET ─────────────────────────────────────────────
first_smet=("$SMET_DIR"/cluster_*.smet)
BDATE=$(awk '/^\[DATA\]/{found=1; next} found{print $1; exit}' "${first_smet[0]}")

echo "============================================"
echo "  Reinit Comparison Run"
echo "  Season:      $BDATE  →  $EDATE"
echo "  Event date:  $EVENT_DATE"
echo "  Date before: $DATE_BEFORE  |  Date after: $DATE_AFTER"
echo "  No-reinit:   $NO_REINIT_DIR"
echo "  With-reinit: $WITH_REINIT_DIR"
echo "  Started:     $(date -u)"
echo "============================================"
echo ""

# ─── Helper: run SNOWPACK for all clusters into OUT_DIR ───────────────────────
# Usage: run_snowpack OUT_DIR B_DATE E_DATE
# Reads restart .sno from OUT_DIR/output if present, else expands template.
# Does NOT build Zarr (caller does that explicitly).
run_snowpack() {
    local out_dir="$1"
    local b_date="$2"
    local e_date="$3"
    local snow_in="$out_dir/input/snow"
    local output="$out_dir/output"

    mkdir -p "$output" "$snow_in"
    [[ ! -f "$snow_in/template.sno" ]] && cp "$TEMPLATE_SNO" "$snow_in/template.sno"

    local smets=("$SMET_DIR"/cluster_*.smet)
    echo "    Clusters: ${#smets[@]}  |  $b_date → $e_date"

    for smet in "${smets[@]}"; do
        local cid lat lon alt res_sno cluster_sno cluster_ini
        cid=$(basename "$smet" .smet)
        lat=$(awk -F'=' '/^latitude/{gsub(/ /,"",$2); print $2}' "$smet")
        lon=$(awk -F'=' '/^longitude/{gsub(/ /,"",$2); print $2}' "$smet")
        alt=$(awk -F'=' '/^altitude/{gsub(/ /,"",$2); print $2}' "$smet")

        # Prefer restart .sno from a prior phase; fall back to template expansion
        res_sno="$output/${cid}_${cid}.sno"
        cluster_sno="$snow_in/${cid}.sno"

        if [[ -f "$res_sno" ]]; then
            cp "$res_sno" "$cluster_sno"
        else
            sed -e "s/WRFPT/$cid/g"  \
                -e "s/WRFNAME/$cid/g" \
                -e "s/WRFLAT/$lat/g"  \
                -e "s/WRFLON/$lon/g"  \
                -e "s/WRFELEV/$alt/g" \
                -e "s/WRFDATE/${b_date%T*}/g" \
                "$snow_in/template.sno" > "$cluster_sno"
        fi

        cluster_ini="$output/${cid}.ini"
        cat > "$cluster_ini" << EOF
IMPORT_BEFORE = $MASTER_CFG

[Input]
METEOPATH  = $SMET_DIR
SNOWPATH   = $snow_in
METEOFILE1 = $cid
SNOWFILE1  = $cid

[Output]
METEOPATH  = $output
SNOWPATH   = $output
EXPERIMENT = $cid
EOF
        "$SNOWPACK_BIN" -c "$cluster_ini" -b "$b_date" -e "$e_date" \
            > "$output/${cid}.log" 2>&1 &

        local ct
        ct=$(ps aux | grep "[s]nowpack.*exe" | wc -l)
        while [[ $ct -ge $MAX_JOBS ]]; do
            sleep 1
            ct=$(ps aux | grep "[s]nowpack.*exe" | wc -l)
        done
    done
    wait

    local n_ok=0 n_fail=0
    for log in "$output"/cluster_*.log; do
        grep -q "done!" "$log" 2>/dev/null && n_ok=$((n_ok+1)) || n_fail=$((n_fail+1))
    done
    echo "    Done: $n_ok ok, $n_fail failed"
    [[ $n_fail -gt 0 ]] && echo "    (check $output/cluster_*.log for failures)"
}

# ─── 1. No-reinit: single full-season run ─────────────────────────────────────
echo "=== [1/5] No-reinit full-season run ==="
run_snowpack "$NO_REINIT_DIR" "$BDATE" "$EDATE"

echo ""
echo "=== [2/5] Building no-reinit Zarr ==="
$PYTHON "$REPO_DIR/src/avachain/build_zarr_chunked.py" \
    --pro-dir "$NO_REINIT_DIR/output" \
    --zarr-out "$NO_REINIT_DIR/output/slope_snowpack.zarr"
echo "    → $NO_REINIT_DIR/output/slope_snowpack.zarr"

# ─── 2. With-reinit: phase 1 → reinit → phase 2 ──────────────────────────────
echo ""
echo "=== [3/5] With-reinit phase 1: season start → event date ==="
# Phase 1 ends at event date so SNOWPACK writes restart .sno files at that state
run_snowpack "$WITH_REINIT_DIR" "$BDATE" "${EVENT_DATE}T18:00"

echo ""
echo "=== [4/5] Applying reinit at $EVENT_DATE ==="
# reinitialize_snowpack.py reads restart .sno from --sno-dir, modifies them
# in-place, and writes backup .sno.bak files alongside each modified file.
#
# It uses the post-event UAS survey HS (date_after) as the scour target when
# station dHS between event and survey is < 5 cm (clean window); otherwise
# falls back to modeled slab_thickness.
$PYTHON "$REPO_DIR/src/avachain/reinitialize_snowpack.py" \
    --project-dir "$REPO_DIR" \
    --date-before  "$DATE_BEFORE" \
    --date-after   "$DATE_AFTER" \
    --event-date   "$EVENT_DATE" \
    --snapshot-date "$EVENT_DATE" \
    --sno-dir      "$WITH_REINIT_DIR/output" \
    --release-geojson "$RELEASE_GEOJSON" \
    --no-backup

echo ""
echo "=== [5/5] With-reinit phase 2: event date → end ==="
# Phase 2 picks up from the reinit'd .sno files (restart path in run_snowpack)
run_snowpack "$WITH_REINIT_DIR" "${EVENT_DATE}T18:00" "$EDATE"

echo ""
echo "=== Building with-reinit Zarr ==="
# Phase 2 .pro files cover event_date→end; phase 1 .pro files cover start→event_date.
# SNOWPACK appends to .pro on restart, so each cluster's .pro has the full season.
$PYTHON "$REPO_DIR/src/avachain/build_zarr_chunked.py" \
    --pro-dir "$WITH_REINIT_DIR/output" \
    --zarr-out "$WITH_REINIT_DIR/output/slope_snowpack.zarr"
echo "    → $WITH_REINIT_DIR/output/slope_snowpack.zarr"

# ─── 3. Comparison plots ──────────────────────────────────────────────────────
echo ""
echo "=== Generating comparison plots ==="
$PYTHON "$REPO_DIR/src/avachain/compare_reinit_runs.py" \
    --zarr-no-reinit   "$NO_REINIT_DIR/output/slope_snowpack.zarr" \
    --zarr-with-reinit "$WITH_REINIT_DIR/output/slope_snowpack.zarr" \
    --smet-dir         "$SMET_DIR" \
    --dem-tif          "$REPO_DIR/outputs/resampled_1m/dem_1m.tif" \
    --release-geojson  "$RELEASE_GEOJSON" \
    --crown-geojson    "$REPO_DIR/data/boundaries/avalanche_release_area_top.geojson" \
    --event-date       "$EVENT_DATE" \
    --out-dir          "$REPO_DIR/outputs/plots/comparison_v2"

echo ""
echo "============================================"
echo "  Comparison complete: $(date -u)"
echo "  Plots: outputs/plots/comparison_v2/"
echo "============================================"

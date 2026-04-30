#!/bin/bash
# Run SNOWPACK on all Little Professor cluster .smet files.
#
# Usage:
#   ./run_snowpack.sh                          # full season (default dates)
#   ./run_snowpack.sh 2025-11-26 2026-04-15   # explicit start/end dates (YYYY-MM-DD)
#
# On first run:  each cluster gets a copy of template.sno as initial conditions.
# On reruns:     each cluster restarts from its own _res.sno (written by SNOWPACK).

# --- Paths ---
SNOWPACK_BIN=/home/caic/caic/rtsys/snowpack/exe/snowpack
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/home/caic/caic/rtsys/snowpack/src/snowpack/lib

PROJECT_DIR=/home/ron/snowpack_model_feeder/snowpack/little_prof
SMET_DIR=/home/ron/snowpack_model_feeder/outputs/smet
SNOW_IN_DIR=$PROJECT_DIR/input/snow
OUTPUT_DIR=$PROJECT_DIR/output
MASTER_CFG=$PROJECT_DIR/config/master_config.ini
TEMPLATE_SNO=$SNOW_IN_DIR/template.sno

# --- Date range ---
# Get first data timestamp from first cluster SMET
first_smet=("$SMET_DIR"/cluster_*.smet)
BDATE=$(awk '/^\[DATA\]/{found=1; next} found{print $1; exit}' "${first_smet[0]}")
EDATE="${2:-2026-03-31}T18:00"

# --- Concurrency ---
# Mirror the operational script's ps-count throttle.
MAX_JOBS=30

# --- Sanity checks ---
for f in "$SNOWPACK_BIN" "$MASTER_CFG" "$TEMPLATE_SNO"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: required file not found: $f" >&2
        exit 1
    fi
done

smets=( "$SMET_DIR"/cluster_*.smet )
if [[ ${#smets[@]} -eq 0 ]]; then
    echo "ERROR: no cluster .smet files found in $SMET_DIR" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "=== Little Professor SNOWPACK run ==="
echo "    Start:    $BDATE"
echo "    End:      $EDATE"
echo "    Clusters: ${#smets[@]}"
echo "    Started:  $(date -u)"
echo ""

for smet in "${smets[@]}"; do
    cid=$(basename "$smet" .smet)

    # Extract cluster coordinates from SMET header for .sno substitution
    lat=$(awk -F'=' '/^latitude/{gsub(/ /,"",$2); print $2}' "$smet")
    lon=$(awk -F'=' '/^longitude/{gsub(/ /,"",$2); print $2}' "$smet")
    alt=$(awk -F'=' '/^altitude/{gsub(/ /,"",$2); print $2}' "$smet")

    # Initial conditions: prefer a previous run's restart file, else substitute
    # the coordinate placeholders in the template and write a per-cluster .sno
    res_sno="$OUTPUT_DIR/${cid}_res.sno"
    cluster_sno="$SNOW_IN_DIR/${cid}.sno"
    if [[ -f "$res_sno" ]]; then
        cp "$res_sno" "$cluster_sno"
    else
        sed -e "s/WRFPT/$cid/g"  \
            -e "s/WRFNAME/$cid/g" \
            -e "s/WRFLAT/$lat/g"  \
            -e "s/WRFLON/$lon/g"  \
            -e "s/WRFELEV/$alt/g" \
            -e "s/WRFDATE/${BDATE%T*}/g" \
            "$TEMPLATE_SNO" > "$cluster_sno"
    fi

    # Write per-cluster ini to output dir — kept after run for reproducibility
    cluster_ini="$OUTPUT_DIR/${cid}.ini"
    cat > "$cluster_ini" << EOF
IMPORT_BEFORE = $MASTER_CFG

[Input]
METEOPATH  = $SMET_DIR
SNOWPATH   = $SNOW_IN_DIR
METEOFILE1 = $cid
SNOWFILE1  = $cid

[Output]
METEOPATH  = $OUTPUT_DIR
SNOWPATH   = $OUTPUT_DIR
EXPERIMENT = $cid
EOF

    "$SNOWPACK_BIN" -c "$cluster_ini" -b "$BDATE" -e "$EDATE" \
        > "$OUTPUT_DIR/${cid}.log" 2>&1 &

    # Throttle: match the operational script's ps-count approach
    ct=$(ps aux | grep "[s]nowpack.*exe" | wc -l)
    while [[ $ct -ge $MAX_JOBS ]]; do
        sleep 1
        ct=$(ps aux | grep "[s]nowpack.*exe" | wc -l)
    done
done

wait
echo "=== Run complete: $(date -u) ==="

# Quick summary: count successes vs failures
n_ok=0
n_fail=0
for log in "$OUTPUT_DIR"/cluster_*.log; do
    if grep -q "done!" "$log" 2>/dev/null; then
        n_ok=$((n_ok + 1))
    else
        n_fail=$((n_fail + 1))
    fi
done
echo "    OK: $n_ok   Failed: $n_fail"
if [[ $n_fail -gt 0 ]]; then
    echo "    Failed clusters:"
    for log in "$OUTPUT_DIR"/cluster_*.log; do
        if ! grep -q "done!" "$log" 2>/dev/null; then
            echo "      $(basename "$log" .log)"
        fi
    done
fi

wait
echo "=== Run complete: $(date -u) ==="

# Quick summary: count successes vs failures
n_ok=0
n_fail=0
for log in "$OUTPUT_DIR"/cluster_*.log; do
    if grep -q "done!" "$log" 2>/dev/null; then
        n_ok=$((n_ok + 1))
    else
        n_fail=$((n_fail + 1))
    fi
done
echo "    OK: $n_ok   Failed: $n_fail"

if [[ $n_fail -gt 0 ]]; then
    echo "    Failed clusters:"
    for log in "$OUTPUT_DIR"/cluster_*.log; do
        if ! grep -q "done!" "$log" 2>/dev/null; then
            echo "      $(basename "$log" .log)"
        fi
    done
fi

# --- NEW: Build Zarr Store ---

echo ""
echo "=== Aggregating results to Zarr ==="
/home/ron/snowpack_model_feeder/.venv/bin/python3 /home/ron/snowpack_model_feeder/snowpack/little_prof/build_zarr_chunked.py
echo "=== Zarr Build Complete ==="

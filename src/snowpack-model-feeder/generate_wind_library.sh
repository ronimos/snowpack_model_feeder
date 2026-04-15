#!/bin/bash
# Generate WindNinja wind library for Little Professor transport model.
#
# 16 directions x 6 speeds = 96 runs
# 6 parallel batches x 8 threads = 48 threads (leaving 16 free)
#
# Output: /home/ron/snowpack/little_prof/windninja/library/
#   dir_{DIR}_spd_{SPD}_vel.asc
#   dir_{DIR}_spd_{SPD}_ang.asc
#
# Usage:
#   ./generate_wind_library.sh
#   ./generate_wind_library.sh --dry-run   # print commands without running

set -uo pipefail

WINDNINJA=/home/ron/miniforge/envs/windninja/bin/WindNinja_cli
DEM=/home/ron/snowpack/little_prof/windninja/dem_utm.tif
LIB_DIR=/home/ron/snowpack/little_prof/windninja/library
LOG_DIR=/home/ron/snowpack/little_prof/windninja/logs
THREADS_PER_RUN=8
MAX_PARALLEL=6

export PROJ_DATA=/home/ron/miniforge/envs/windninja/share/proj
export GDAL_DATA=/home/ron/miniforge/envs/windninja/share/gdal

DIRECTIONS=(0 22.5 45 67.5 90 112.5 135 157.5 180 202.5 225 247.5 270 292.5 315 337.5)
SPEEDS=(3 8 15 25 40 60)

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

mkdir -p "$LIB_DIR" "$LOG_DIR"

total=$(( ${#DIRECTIONS[@]} * ${#SPEEDS[@]} ))
echo "=== WindNinja library generation ==="
echo "    DEM:        $DEM"
echo "    Directions: ${#DIRECTIONS[@]}"
echo "    Speeds:     ${#SPEEDS[@]} (${SPEEDS[*]} m/s)"
echo "    Total runs: $total"
echo "    Parallel:   $MAX_PARALLEL x ${THREADS_PER_RUN} threads"
echo ""

run_one() {
    local dir=$1
    local spd=$2

    local dir_tag spd_tag
    dir_tag=$(printf "%05.1f" "$dir")
    spd_tag=$(printf "%02d" "$spd")

    local vel_out="$LIB_DIR/dir_${dir_tag}_spd_${spd_tag}_vel.asc"
    local log="$LOG_DIR/dir_${dir_tag}_spd_${spd_tag}.log"

    if [[ -f "$vel_out" ]]; then
        echo "  SKIP dir=${dir} spd=${spd} (exists)"
        return 0
    fi

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  DRYRUN: WindNinja_cli --input_direction $dir --input_speed $spd ..."
        return 0
    fi

    "$WINDNINJA" \
        --initialization_method domainAverageInitialization \
        --input_speed "$spd" \
        --input_speed_units mps \
        --input_direction "$dir" \
        --input_wind_height 10 \
        --units_input_wind_height m \
        --output_wind_height 10 \
        --units_output_wind_height m \
        --vegetation grass \
        --mesh_resolution 1 \
        --units_mesh_resolution m \
        --output_speed_units mps \
        --write_ascii_output true \
        --num_threads "$THREADS_PER_RUN" \
        --elevation_file "$DEM" \
        --output_path "$LIB_DIR" \
        > "$log" 2>&1

    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        echo "  FAILED dir=${dir} spd=${spd} -- see $log"
        return 1
    fi

    # WindNinja rounds .5 directions (22.5->23, 202.5->203 etc) in filenames.
    # Try exact match first, then rounded integer fallback.
    local dir_rounded
    # python round() uses banker's rounding; WindNinja always rounds .5 up
    dir_rounded=$(python3 -c "import math; print(math.floor(${dir}+0.5))")
    local wn_base=""
    for candidate in "$LIB_DIR/dem_utm_${dir}_${spd}_1m" \
                     "$LIB_DIR/dem_utm_${dir_rounded}_${spd}_1m"; do
        if [[ -f "${candidate}_vel.asc" ]]; then
            wn_base="$candidate"
            break
        fi
    done

    if [[ -z "$wn_base" ]]; then
        echo "  WARNING: output not found for dir=${dir} spd=${spd}"
        return 1
    fi

    mv "${wn_base}_vel.asc" "$vel_out"
    mv "${wn_base}_ang.asc" "$LIB_DIR/dir_${dir_tag}_spd_${spd_tag}_ang.asc"
    rm -f "${wn_base}_cld.asc" "${wn_base}_cld.prj" \
          "${wn_base}_vel.prj" "${wn_base}_ang.prj"

    local sim_time
    sim_time=$(grep "Total simulation time" "$log" | awk '{print $(NF-1), $NF}')
    echo "  OK  dir=${dir} spd=${spd} (${sim_time})"
}

export -f run_one
export WINDNINJA DEM LIB_DIR LOG_DIR THREADS_PER_RUN DRY_RUN PROJ_DATA GDAL_DATA

echo "Started: $(date -u)"

pids=()
jobs_running=0

for dir in "${DIRECTIONS[@]}"; do
    for spd in "${SPEEDS[@]}"; do
        run_one "$dir" "$spd" &
        pids+=($!)
        jobs_running=$(( jobs_running + 1 ))

        if [[ $jobs_running -ge $MAX_PARALLEL ]]; then
            wait "${pids[0]}"
            pids=("${pids[@]:1}")
            jobs_running=$(( jobs_running - 1 ))
        fi
    done
done

for pid in "${pids[@]}"; do
    wait "$pid"
done

echo ""
echo "Finished: $(date -u)"
n_ok=$(ls "$LIB_DIR"/*_vel.asc 2>/dev/null | wc -l)
echo "Library: $n_ok / $total velocity grids written to $LIB_DIR"

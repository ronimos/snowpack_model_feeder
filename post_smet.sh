#!/bin/bash
# post_smet.sh — run after pipeline.py smet

python - << 'EOF'
from smet_writer import read_smet, fill_smet_gaps_from_refill
from pathlib import Path

smet_dir     = Path("/home/ron/snowpack_model_feeder/outputs/smet")
refill_path  = Path("/home/ron/snowpack_model_feeder/data/weather/refill.smet")

refill_header, refill_df = read_smet(str(refill_path))
n_total = 0
for smet_file in sorted(smet_dir.glob("cluster_*.smet")):
    result = fill_smet_gaps_from_refill(
        str(smet_file), refill_header, refill_df, max_gap_hours=48)
    if result['n_rows_inserted'] or result['n_rows_filled']:
        n_total += 1
print(f"Gap-filled {n_total} SMETs")
EOF

# In post_smet.sh, replace the rm line with:
rm -f /home/ron/snowpack/little_prof/output/cluster_*_cluster_*.sno \
      /home/ron/snowpack/little_prof/output/cluster_*_cluster_*.pro \
      /home/ron/snowpack/little_prof/output/cluster_*_cluster_*.haz \
      /home/ron/snowpack/little_prof/output/cluster_*_cluster_*.ini \
      /home/ron/snowpack/little_prof/input/snow/cluster_*.sno \
      /home/ron/snowpack/little_prof/output/*.zarr

cd /home/ron/snowpack/little_prof && ./run_snowpack.sh


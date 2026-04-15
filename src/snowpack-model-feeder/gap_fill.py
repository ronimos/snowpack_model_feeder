from pathlib import Path, sys

base_dir = Path("/home/ron/snowpack_model_feeder/outputs/plots/daily_frames")

frame_names = sorted(f.name for f in (base_dir / "loading").glob("*.png"))
print(f"Found {len(frame_names)} frames: {frame_names[0]} -> {frame_names[-1]}")

import sys
sys.path.insert(0, "/home/ron/snowpack_model_feeder/src/snowpack-model-feeder")
from visualize_snowpack import write_html_flipbook
write_html_flipbook(base_dir, frame_names, min_depth=30.0)
print("Done")
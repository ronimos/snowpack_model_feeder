#!/bin/bash
# One-time setup for Little Professor SNOWPACK project.
# Run from anywhere; paths are absolute.

PROJECT_DIR=/home/ron/snowpack_model_feeder/snowpack/little_prof

mkdir -p "$PROJECT_DIR/config"
mkdir -p "$PROJECT_DIR/input/snow"
mkdir -p "$PROJECT_DIR/output"

echo "Created directory structure under $PROJECT_DIR"
echo ""
echo "Next steps:"
echo "  1. Copy your master_config.ini   → $PROJECT_DIR/config/"
echo "  2. Copy your template.sno        → $PROJECT_DIR/input/snow/"
echo "     Set SlopeAzi = 90 in template.sno for virtual east aspect"
echo "  3. Copy run_snowpack.sh          → $PROJECT_DIR/"
echo "  4. chmod +x $PROJECT_DIR/run_snowpack.sh"


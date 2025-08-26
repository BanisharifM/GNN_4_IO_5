#!/bin/bash

# Run Darshan Parser for GNN I/O Optimization Project
# Place this script in: /work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/scripts/

# Set up paths
PROJECT_ROOT="/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5"
PARSER_SCRIPT="$PROJECT_ROOT/scripts/preprocessing/darshan_parser.py"
CONFIG_FILE="$PROJECT_ROOT/configs/darshan_features.json"
DARSHAN_LOG_DIR="$PROJECT_ROOT/darshan_log"
OUTPUT_DIR="$PROJECT_ROOT/data"

# Navigate to project root
cd $PROJECT_ROOT

# Example 1: Parse a single E2E Darshan log
echo "Parsing E2E ultra optimized log..."
python $PARSER_SCRIPT \
    "$DARSHAN_LOG_DIR/darshan_log_E2E/e2e_ultra_optimized_11643472_64procs_32stripes_32mb.darshan" \
    "$OUTPUT_DIR/e2e_ultra_optimized_features.csv" \
    --config $CONFIG_FILE

# Example 2: Parse a single IOR Darshan log
echo "Parsing IOR baseline log..."
python $PARSER_SCRIPT \
    "$DARSHAN_LOG_DIR/darshan_log_ior/11464139_baseline.darshan" \
    "$OUTPUT_DIR/ior_baseline_features.csv" \
    --config $CONFIG_FILE

# Example 3: Batch process all IOR logs
echo "Batch processing all IOR logs..."
python $PARSER_SCRIPT \
    "$DARSHAN_LOG_DIR/darshan_log_ior/" \
    "$OUTPUT_DIR/ior_all_features.csv" \
    --config $CONFIG_FILE \
    --batch

# Example 4: Batch process all E2E logs
echo "Batch processing all E2E logs..."
python $PARSER_SCRIPT \
    "$DARSHAN_LOG_DIR/darshan_log_E2E/" \
    "$OUTPUT_DIR/e2e_all_features.csv" \
    --config $CONFIG_FILE \
    --batch

echo "Parsing complete! Check the output files in $OUTPUT_DIR"
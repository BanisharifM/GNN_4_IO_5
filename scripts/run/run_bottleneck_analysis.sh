#!/bin/bash
#
# Run IO Bottleneck Analysis
#
# Usage: ./run_bottleneck_analysis.sh <test_file> [options]

# Default paths - update these for your environment
DEFAULT_MODEL="/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/data/1M/best_model.pt"
DEFAULT_DATA_DIR="/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/data/1M"
DEFAULT_OUTPUT="/work/hdd/bdau/mbanisharifdehkordi/GNN_4_IO_5/benchmark_evaluation/E2E/results/Study7"

# Check if test file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <test_file> [additional_options]"
    echo "Example: $0 benchmark_evaluation/E2E/test_sample.csv"
    exit 1
fi

TEST_FILE=$1
shift  # Remove first argument, keep rest for passing to Python script

# Create output directory if it doesn't exist
mkdir -p $DEFAULT_OUTPUT

# Run analysis
python -m io_bottleneck_analyzer.run_analysis \
    "$TEST_FILE" \
    --model-path "$DEFAULT_MODEL" \
    --data-dir "$DEFAULT_DATA_DIR" \
    --output-dir "$DEFAULT_OUTPUT" \
    "$@"  # Pass any additional arguments
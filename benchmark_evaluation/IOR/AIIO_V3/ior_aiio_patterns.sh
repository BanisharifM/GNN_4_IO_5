#!/bin/bash
#SBATCH --job-name=ior_aiio_patterns
#SBATCH --nodes=2
#SBATCH --ntasks=256
#SBATCH --time=02:00:00
#SBATCH --partition=cpu
#SBATCH --account=bdau-delta-cpu
#SBATCH --output=IOR_Bench/AIIO_V3/logs/ior_patterns_%j.out
#SBATCH --error=IOR_Bench/AIIO_V3/logs/ior_patterns_%j.err

# ============================================
# LOAD SYSTEM MODULES (like E2E)
# ============================================
module load gcc/11.4.0 openmpi/4.1.6

# ============================================
# CONFIGURATION
# ============================================
BASE_DIR="/work/hdd/bdau/mbanisharifdehkordi/E2E"
IOR_BENCH_DIR="$BASE_DIR/IOR_Bench/AIIO_V3"
DARSHAN_LOG_BASE="$BASE_DIR/darshan_logs"
DARSHAN_HOME="$HOME/darshan-fresh"
PARSER_SCRIPT="$BASE_DIR/evaluation/parser.py"
SAMPLE_CSV="$BASE_DIR/evaluation/sample_train_100.csv"

# IOR from conda, but use SYSTEM mpirun
IOR_BIN="$HOME/.conda/envs/ior_env/bin/ior"
LIBDARSHAN="$DARSHAN_HOME/lib/libdarshan.so"

# Python environment
PYTHON_ENV="/u/mbanisharifdehkordi/.conda/envs/gnn4_env/bin/python"

# Export Darshan paths
export PATH="$DARSHAN_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$DARSHAN_HOME/lib:$LD_LIBRARY_PATH"
export DARSHAN_HOME

# ============================================
# SETUP DIRECTORIES
# ============================================
mkdir -p "$IOR_BENCH_DIR"/{Pattern1,Pattern1_Optimized,Pattern2,Pattern3,Pattern4,Pattern5,Pattern6,logs}
mkdir -p "$DARSHAN_LOG_BASE/$(date +%Y)/$(date +%-m)/$(date +%-d)"

echo "=================================================="
echo "IOR AIIO Pattern Benchmark Suite"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: 2"
echo "Tasks: 256"
echo "Base Directory: $IOR_BENCH_DIR"
echo "Darshan: $DARSHAN_LOG_BASE"
echo "Date: $(date)"
echo "=================================================="

# ============================================
# HELPER FUNCTION: Process Darshan Log
# ============================================
process_darshan_log() {
    local pattern_name=$1
    local pattern_dir=$2
    local ior_config=$3
    
    echo "Processing Darshan log for $pattern_name..."
    
    # Find most recent Darshan log
    local darshan_file=$(find $DARSHAN_LOG_BASE -name "*.darshan" -type f -mmin -5 2>/dev/null | head -1)
    
    if [ -n "$darshan_file" ]; then
        # Copy to pattern directory
        local final_log="$pattern_dir/ior_${pattern_name}_${SLURM_JOB_ID}.darshan"
        cp "$darshan_file" "$final_log"
        
        # Parse to text
        $DARSHAN_HOME/bin/darshan-parser --show-incomplete "$final_log" > "${final_log%.darshan}_parsed.txt"
        
        # Extract key metrics
        echo "=== Performance Metrics for $pattern_name ===" > "$pattern_dir/metrics.txt"
        echo "IOR Config: $ior_config" >> "$pattern_dir/metrics.txt"
        echo "" >> "$pattern_dir/metrics.txt"
        $DARSHAN_HOME/bin/darshan-parser "$final_log" | \
            grep -E "agg_perf_by_slowest|POSIX_WRITES|POSIX_READS|POSIX_SIZE|POSIX_STRIDE|POSIX_FILE_NOT_ALIGNED|POSIX_SEEKS" \
            >> "$pattern_dir/metrics.txt" 2>/dev/null
        
        # Convert to CSV using existing parser
        local temp_dir="/tmp/darshan_parse_${SLURM_JOB_ID}_${pattern_name}"
        mkdir -p "$temp_dir"
        
        DARSHAN_HOME="$DARSHAN_HOME" $PYTHON_ENV "$PARSER_SCRIPT" \
            "$pattern_dir" \
            "$pattern_dir/parsed.csv" \
            "$SAMPLE_CSV" \
            "$temp_dir" 2>&1 | tee "$pattern_dir/parse.log"
        
        rm -rf "$temp_dir"
        rm "$darshan_file"  # Clean up original
        
        echo "✓ Processed: $final_log"
        echo "✓ CSV: $pattern_dir/parsed.csv"
    else
        echo "✗ WARNING: No Darshan log found for $pattern_name"
        echo "  Searched in: $DARSHAN_LOG_BASE"
        find $DARSHAN_LOG_BASE -name "*.darshan" -mmin -10 -ls
    fi
    
    sleep 2
}

# ============================================
# PATTERN 1: Sequential Writing with Small I/O (BASELINE)
# ============================================
echo ""
echo "=================================================="
echo "PATTERN 1: Sequential Writing with Small I/O (BASELINE)"
echo "=================================================="
echo "Configuration: ior -w -t 1k -b 1m -Y"
echo "Expected Issue: Small write operations"
echo "Expected Performance: ~1-2 MiB/s (poor)"
echo "AIIO Paper: Fig 7(a)"

cd "$IOR_BENCH_DIR/Pattern1"

# Setup Lustre striping (minimal for poor performance)
lfs setstripe . -c 1 -S 1M

# Configure Darshan for this run
export DARSHAN_LOG_DIR="$DARSHAN_LOG_BASE"
export DARSHAN_LOGPATH="$DARSHAN_LOG_DIR"
export DXT_ENABLE_IO_TRACE=1
export DARSHAN_DISABLE_SHARED_REDUCTION=1

# Force POSIX-only, disable optimizations
export ROMIO_HINTS="romio_no_indep_rw=false"
export HDF5_USE_FILE_LOCKING=FALSE

echo "Running IOR..."
LD_PRELOAD="$LIBDARSHAN" mpirun -np 256 $IOR_BIN \
    -a POSIX \
    -w \
    -t 1k \
    -b 1m \
    -Y \
    -o ./ior_pattern1_file \
    -v

sleep 5
process_darshan_log "Pattern1" "$IOR_BENCH_DIR/Pattern1" "ior -w -t 1k -b 1m -Y"

# ============================================
# PATTERN 1 OPTIMIZED: Sequential Writing with LARGE I/O
# ============================================
echo ""
echo "=================================================="
echo "PATTERN 1 OPTIMIZED: Sequential Writing with LARGE I/O"
echo "=================================================="
echo "Configuration: ior -w -t 1M -b 1m -Y"
echo "Expected: Much better performance (~150-200 MiB/s)"
echo "This demonstrates the fix for Pattern 1's small I/O issue"
echo "AIIO Paper: Fig 7(b)"

cd "$IOR_BENCH_DIR/Pattern1_Optimized"
lfs setstripe . -c 1 -S 1M

echo "Running IOR with large transfer size..."
LD_PRELOAD="$LIBDARSHAN" mpirun -np 256 $IOR_BIN \
    -a POSIX \
    -w \
    -t 1M \
    -b 1m \
    -Y \
    -o ./ior_pattern1_opt_file \
    -v

sleep 5
process_darshan_log "Pattern1_Optimized" "$IOR_BENCH_DIR/Pattern1_Optimized" "ior -w -t 1M -b 1m -Y"

# ============================================
# PATTERN 2: Sequential Reading with Small Requests
# ============================================
echo ""
echo "=================================================="
echo "PATTERN 2: Sequential Reading with Small Requests"
echo "=================================================="
echo "Configuration: ior -r -t 1k -b 1m"
echo "Expected Issue: Excessive seeks for sequential reads"
echo "Expected Performance: ~400-500 MiB/s"
echo "AIIO Paper: Fig 8(a)"

cd "$IOR_BENCH_DIR/Pattern2"
lfs setstripe . -c 1 -S 1M

# First write the file - ADD -k FLAG TO KEEP FILE
echo "Creating test file..."
LD_PRELOAD="$LIBDARSHAN" mpirun -np 256 $IOR_BIN \
    -a POSIX \
    -w \
    -t 1k \
    -b 1m \
    -k \
    -o ./ior_pattern2_file \
    -v

# Wait and clear Darshan log from write
sleep 5
find $DARSHAN_LOG_BASE -name "*.darshan" -type f -mmin -5 -delete

# Now do the read test
echo "Running read test..."
LD_PRELOAD="$LIBDARSHAN" mpirun -np 256 $IOR_BIN \
    -a POSIX \
    -r \
    -t 1k \
    -b 1m \
    -o ./ior_pattern2_file \
    -v

sleep 5
process_darshan_log "Pattern2" "$IOR_BENCH_DIR/Pattern2" "ior -r -t 1k -b 1m"

# ============================================
# PATTERN 3: Noncontiguous Writing with Fixed Stride
# ============================================
echo ""
echo "=================================================="
echo "PATTERN 3: Noncontiguous Writing with Fixed Stride"
echo "=================================================="
echo "Configuration: ior -w -t 1k -b 1k -s 1024 -Y"
echo "Expected Issue: Stride pattern, small writes"
echo "Expected Performance: ~1-2 MiB/s (very poor)"
echo "AIIO Paper: Fig 9"

cd "$IOR_BENCH_DIR/Pattern3"
lfs setstripe . -c 1 -S 1M

echo "Running IOR..."
LD_PRELOAD="$LIBDARSHAN" mpirun -np 256 $IOR_BIN \
    -a POSIX \
    -w \
    -t 1k \
    -b 1k \
    -s 1024 \
    -Y \
    -o ./ior_pattern3_file \
    -v

sleep 5
process_darshan_log "Pattern3" "$IOR_BENCH_DIR/Pattern3" "ior -w -t 1k -b 1k -s 1024 -Y"

# ============================================
# PATTERN 4: Noncontiguous Reading with Fixed Stride
# ============================================
echo ""
echo "=================================================="
echo "PATTERN 4: Noncontiguous Reading with Fixed Stride"
echo "=================================================="
echo "Configuration: ior -r -t 1k -b 1k -s 1024"
echo "Expected Issue: Excessive seeks, misalignment"
echo "Expected Performance: ~60-70 MiB/s"
echo "AIIO Paper: Fig 10"

cd "$IOR_BENCH_DIR/Pattern4"
lfs setstripe . -c 1 -S 1M

# Write file first - ADD -k FLAG TO KEEP FILE
echo "Creating test file..."
LD_PRELOAD="$LIBDARSHAN" mpirun -np 256 $IOR_BIN \
    -a POSIX \
    -w \
    -t 1k \
    -b 1k \
    -s 1024 \
    -k \
    -o ./ior_pattern4_file \
    -v

sleep 5
find $DARSHAN_LOG_BASE -name "*.darshan" -type f -mmin -5 -delete

# Read test
echo "Running read test..."
LD_PRELOAD="$LIBDARSHAN" mpirun -np 256 $IOR_BIN \
    -a POSIX \
    -r \
    -t 1k \
    -b 1k \
    -s 1024 \
    -o ./ior_pattern4_file \
    -v

sleep 5
process_darshan_log "Pattern4" "$IOR_BENCH_DIR/Pattern4" "ior -r -t 1k -b 1k -s 1024"

# ============================================
# PATTERN 5: Writing with Random Offset
# ============================================
echo ""
echo "=================================================="
echo "PATTERN 5: Writing with Random Offset"
echo "=================================================="
echo "Configuration: ior -w -t 1k -b 1m -z -Y"
echo "Expected Issue: Random access, misalignment, strides"
echo "Expected Performance: ~1-2 MiB/s (very poor)"
echo "AIIO Paper: Fig 11"

cd "$IOR_BENCH_DIR/Pattern5"
lfs setstripe . -c 1 -S 1M

echo "Running IOR..."
LD_PRELOAD="$LIBDARSHAN" mpirun -np 256 $IOR_BIN \
    -a POSIX \
    -w \
    -t 1k \
    -b 1m \
    -z \
    -Y \
    -o ./ior_pattern5_file \
    -v

sleep 5
process_darshan_log "Pattern5" "$IOR_BENCH_DIR/Pattern5" "ior -w -t 1k -b 1m -z -Y"

# ============================================
# PATTERN 6: Reading with Random Offset
# ============================================
echo ""
echo "=================================================="
echo "PATTERN 6: Reading with Random Offset"
echo "=================================================="
echo "Configuration: ior -a POSIX -r -t 1k -b 1m -z"
echo "Expected Issue: Random access, multiple strides"
echo "Expected Performance: ~90-100 MiB/s"
echo "AIIO Paper: Fig 12"

cd "$IOR_BENCH_DIR/Pattern6"
lfs setstripe . -c 1 -S 1M

# Write file first - ADD -k FLAG TO KEEP FILE
echo "Creating test file..."
LD_PRELOAD="$LIBDARSHAN" mpirun -np 256 $IOR_BIN \
    -a POSIX \
    -w \
    -t 1k \
    -b 1m \
    -z \
    -k \
    -o ./ior_pattern6_file \
    -v

sleep 5
find $DARSHAN_LOG_BASE -name "*.darshan" -type f -mmin -5 -delete

# Read test
echo "Running read test..."
LD_PRELOAD="$LIBDARSHAN" mpirun -np 256 $IOR_BIN \
    -a POSIX \
    -r \
    -t 1k \
    -b 1m \
    -z \
    -o ./ior_pattern6_file \
    -v

sleep 5
process_darshan_log "Pattern6" "$IOR_BENCH_DIR/Pattern6" "ior -a POSIX -r -t 1k -b 1m -z"

# ============================================
# GENERATE SUMMARY
# ============================================
echo ""
echo "=================================================="
echo "GENERATING SUMMARY"
echo "=================================================="

cat > "$IOR_BENCH_DIR/summary.txt" <<EOF
IOR AIIO Pattern Benchmark Summary
===================================
Job ID: $SLURM_JOB_ID
Date: $(date)
Nodes: 2
Tasks: 256

Pattern Descriptions (AIIO Paper Table 3):
------------------------------------------
Pattern1:           Sequential Writing with Small I/O - BASELINE (Fig 7a: ior -w -t 1k -b 1m -Y)
Pattern1_Optimized: Sequential Writing with Large I/O - OPTIMIZED (Fig 7b: ior -w -t 1M -b 1m -Y)
Pattern2:           Sequential Reading with Small Requests (Fig 8a: ior -r -t 1k -b 1m)
Pattern3:           Noncontiguous Writing with Fixed Stride (Fig 9: ior -w -t 1k -b 1k -s 1024 -Y)
Pattern4:           Noncontiguous Reading with Fixed Stride (Fig 10: ior -r -t 1k -b 1k -s 1024)
Pattern5:           Writing with Random Offset (Fig 11: ior -w -t 1k -b 1m -z -Y)
Pattern6:           Reading with Random Offset (Fig 12: ior -a POSIX -r -t 1k -b 1m -z)

Note: Fig 8(b) requires IOR source code modification (skipped)

Performance Results:
--------------------
EOF

# Extract performance from each pattern
for pattern in Pattern1 Pattern1_Optimized Pattern2 Pattern3 Pattern4 Pattern5 Pattern6; do
    echo "" >> "$IOR_BENCH_DIR/summary.txt"
    echo "$pattern:" >> "$IOR_BENCH_DIR/summary.txt"
    if [ -f "$IOR_BENCH_DIR/$pattern/metrics.txt" ]; then
        grep "agg_perf_by_slowest" "$IOR_BENCH_DIR/$pattern/metrics.txt" | head -1 >> "$IOR_BENCH_DIR/summary.txt"
        
        # Also extract key counters
        grep -E "POSIX_WRITES:|POSIX_READS:|POSIX_FILE_NOT_ALIGNED:|POSIX_STRIDE" \
            "$IOR_BENCH_DIR/$pattern/metrics.txt" | head -5 >> "$IOR_BENCH_DIR/summary.txt"
    else
        echo "  No metrics found" >> "$IOR_BENCH_DIR/summary.txt"
    fi
done

# Calculate improvement ratio for Pattern1 vs Pattern1_Optimized
echo "" >> "$IOR_BENCH_DIR/summary.txt"
echo "Pattern1 Optimization Analysis:" >> "$IOR_BENCH_DIR/summary.txt"
if [ -f "$IOR_BENCH_DIR/Pattern1/metrics.txt" ] && [ -f "$IOR_BENCH_DIR/Pattern1_Optimized/metrics.txt" ]; then
    baseline_perf=$(grep "agg_perf_by_slowest" "$IOR_BENCH_DIR/Pattern1/metrics.txt" | head -1 | awk '{print $(NF-1)}')
    optimized_perf=$(grep "agg_perf_by_slowest" "$IOR_BENCH_DIR/Pattern1_Optimized/metrics.txt" | head -1 | awk '{print $(NF-1)}')
    
    if [ -n "$baseline_perf" ] && [ -n "$optimized_perf" ]; then
        echo "  Baseline (1k):  $baseline_perf MiB/s" >> "$IOR_BENCH_DIR/summary.txt"
        echo "  Optimized (1M): $optimized_perf MiB/s" >> "$IOR_BENCH_DIR/summary.txt"
        improvement=$(echo "scale=2; $optimized_perf / $baseline_perf" | bc)
        echo "  Improvement: ${improvement}x faster (should be ~100x per AIIO paper)" >> "$IOR_BENCH_DIR/summary.txt"
    fi
fi

cat >> "$IOR_BENCH_DIR/summary.txt" <<EOF

Files Generated:
----------------
EOF

# List all generated files
for pattern in Pattern1 Pattern1_Optimized Pattern2 Pattern3 Pattern4 Pattern5 Pattern6; do
    echo "$pattern:" >> "$IOR_BENCH_DIR/summary.txt"
    echo "  - Darshan: $(ls $IOR_BENCH_DIR/$pattern/*.darshan 2>/dev/null)" >> "$IOR_BENCH_DIR/summary.txt"
    echo "  - CSV: $(ls $IOR_BENCH_DIR/$pattern/parsed*.csv 2>/dev/null)" >> "$IOR_BENCH_DIR/summary.txt"
    echo "  - Metrics: $IOR_BENCH_DIR/$pattern/metrics.txt" >> "$IOR_BENCH_DIR/summary.txt"
done

echo ""
echo "=================================================="
echo "BENCHMARK COMPLETE"
echo "=================================================="
echo "Results directory: $IOR_BENCH_DIR"
echo "Summary: $IOR_BENCH_DIR/summary.txt"
echo ""
cat "$IOR_BENCH_DIR/summary.txt"
echo "=================================================="
#!/bin/bash
#SBATCH --job-name=io500_ion_config_ALL
#SBATCH --account=bdau-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=4
#SBATCH --ntasks=256
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=/work/hdd/bdau/mbanisharifdehkordi/E2E/IO500_Bench/ION_V1/logs/io500_all_%j.out
#SBATCH --error=/work/hdd/bdau/mbanisharifdehkordi/E2E/IO500_Bench/ION_V1/logs/io500_all_%j.err

# ============================================
# LOAD SYSTEM MODULES (same as IOR script)
# ============================================
module load gcc/11.4.0 openmpi/4.1.6

# ============================================
# CONFIGURATION
# ============================================
BASE_DIR="/work/hdd/bdau/mbanisharifdehkordi/E2E"
IO500_DIR="/work/hdd/bdau/mbanisharifdehkordi/io500"
RESULTS_BASE="$BASE_DIR/IO500_Bench/ION_V1"
DARSHAN_LOG_BASE="$BASE_DIR/darshan_logs"
DARSHAN_HOME="$HOME/darshan-fresh"
PARSER_SCRIPT="$BASE_DIR/evaluation/parser.py"
SAMPLE_CSV="$BASE_DIR/evaluation/sample_train_100.csv"
PYTHON_ENV="$HOME/.conda/envs/gnn4_env/bin/python"

LIBDARSHAN="$DARSHAN_HOME/lib/libdarshan.so"

# Export Darshan paths (same as IOR script)
export PATH="$DARSHAN_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$DARSHAN_HOME/lib:$LD_LIBRARY_PATH"
export DARSHAN_HOME

# ============================================
# SETUP DIRECTORIES
# ============================================
mkdir -p "$RESULTS_BASE"/{Config1,Config2,Config3,Config4,Config5,Config6,logs}
mkdir -p "$DARSHAN_LOG_BASE/$(date +%Y)/$(date +%-m)/$(date +%-d)"

echo "=================================================="
echo "IO500 ION Benchmark Suite"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks: $SLURM_NTASKS"
echo "Base Directory: $RESULTS_BASE"
echo "Darshan: $DARSHAN_LOG_BASE"
echo "Date: $(date)"
echo "=================================================="

# ============================================
# HELPER FUNCTION: Process Darshan Log
# (Same approach as IOR script)
# ============================================
process_darshan_log() {
    local config_num=$1
    local config_name=$2
    
    echo ""
    echo "Processing Darshan log for Config${config_num}..."
    
    local config_dir="$RESULTS_BASE/Config${config_num}"
    
    # Wait for Darshan to finalize
    sleep 5
    
    # Find most recent Darshan log (same as IOR script)
    local darshan_file=$(find $DARSHAN_LOG_BASE -name "*.darshan" -type f -mmin -5 2>/dev/null | head -1)
    
    if [ -n "$darshan_file" ]; then
        local final_log="$config_dir/Config${config_num}_${SLURM_JOB_ID}.darshan"
        
        # Copy to config directory
        cp "$darshan_file" "$final_log"
        
        echo "✓ Darshan log found and copied to: $final_log"
        
        # Parse to text
        $DARSHAN_HOME/bin/darshan-parser --show-incomplete "$final_log" > "${final_log%.darshan}_parsed.txt"
        echo "✓ Parsed to: ${final_log%.darshan}_parsed.txt"
        
        # Extract key metrics
        cat > "$config_dir/darshan_metrics.txt" <<METRICS
=== Darshan Metrics for Config${config_num}: $config_name ===
Job ID: $SLURM_JOB_ID
Date: $(date)

Performance Summary:
METRICS
        
        # Extract aggregate performance
        $DARSHAN_HOME/bin/darshan-parser "$final_log" | \
            grep -E "agg_perf_by|total_bytes" | head -20 >> "$config_dir/darshan_metrics.txt"
        
        echo "" >> "$config_dir/darshan_metrics.txt"
        echo "Key I/O Counters:" >> "$config_dir/darshan_metrics.txt"
        
        # Extract key POSIX counters
        $DARSHAN_HOME/bin/darshan-parser "$final_log" | \
            grep -E "POSIX.*WRITES|POSIX.*READS|POSIX.*SIZE|POSIX.*STRIDE|POSIX.*FILE_NOT_ALIGNED|POSIX.*SEEKS" | \
            head -30 >> "$config_dir/darshan_metrics.txt"
        
        echo "✓ Metrics extracted to: $config_dir/darshan_metrics.txt"
        
        # Convert to CSV using existing parser (same as IOR script)
        if [ -f "$PARSER_SCRIPT" ] && [ -f "$SAMPLE_CSV" ]; then
            echo "Converting to CSV..."
            local temp_dir="/tmp/darshan_${SLURM_JOB_ID}_${config_num}"
            mkdir -p "$temp_dir"
            
            DARSHAN_HOME="$DARSHAN_HOME" $PYTHON_ENV "$PARSER_SCRIPT" \
                "$config_dir" \
                "$config_dir/Config${config_num}_parsed.csv" \
                "$SAMPLE_CSV" \
                "$temp_dir" &> "$config_dir/csv_conversion.log"
            
            if [ -f "$config_dir/Config${config_num}_parsed.csv" ]; then
                echo "✓ CSV created: $config_dir/Config${config_num}_parsed.csv"
            else
                echo "✗ CSV conversion failed (check $config_dir/csv_conversion.log)"
            fi
            
            rm -rf "$temp_dir"
        fi
        
        # Clean up original Darshan log
        rm "$darshan_file"
        
    else
        echo "✗ WARNING: No Darshan log found for Config${config_num}"
        echo "  Searched in: $DARSHAN_LOG_BASE"
        find $DARSHAN_LOG_BASE -name "*.darshan" -mmin -10 -ls
    fi
    
    # Copy IO500 results
    if [ -d "$IO500_DIR/results" ]; then
        echo "Copying IO500 results..."
        cp -r "$IO500_DIR/results"/* "$config_dir/" 2>/dev/null
    fi
    
    sleep 2
}

# ============================================
# FUNCTION: Run Single Config
# ============================================
run_config() {
    local config_num=$1
    local config_file=$2
    local config_name=$3
    
    echo ""
    echo "=================================================="
    echo "CONFIG ${config_num}: ${config_name}"
    echo "=================================================="
    echo "Config file: $config_file"
    echo "Start time: $(date)"
    
    cd "$IO500_DIR"
    
    # Clean previous run
    rm -rf ./datafiles/* ./results/* 2>/dev/null
    
    # Clean any recent Darshan logs to avoid confusion
    find $DARSHAN_LOG_BASE -name "*.darshan" -type f -mmin -5 -delete 2>/dev/null
    
    # Verify config file exists
    if [ ! -f "$config_file" ]; then
        echo "✗ ERROR: Config file not found: $config_file"
        return 1
    fi
    
    # Configure Darshan for this run (same as IOR script)
    export DARSHAN_LOG_DIR="$DARSHAN_LOG_BASE"
    export DARSHAN_LOGPATH="$DARSHAN_LOG_DIR"
    export DXT_ENABLE_IO_TRACE=1
    export DARSHAN_DISABLE_SHARED_REDUCTION=1
    
    # Force POSIX-only optimizations off (same as IOR script)
    export ROMIO_HINTS="romio_no_indep_rw=false"
    export HDF5_USE_FILE_LOCKING=FALSE
    
    # Run IO500 with LD_PRELOAD (same approach as IOR script)
    echo "Running: LD_PRELOAD=$LIBDARSHAN ./io500.sh $config_file"
    LD_PRELOAD="$LIBDARSHAN" ./io500.sh "$config_file" 2>&1 | tee "$RESULTS_BASE/Config${config_num}/io500_run.log"
    
    echo "End time: $(date)"
    
    # Process Darshan log
    process_darshan_log "$config_num" "$config_name"
    
    # Clean up large data files
    rm -rf ./datafiles/* 2>/dev/null
    
    sleep 3
}

# ============================================
# RUN ALL 6 CONFIGS SEQUENTIALLY
# ============================================

run_config 1 \
    "$IO500_DIR/configs_ion/config-ion-1_Small_IO.ini" \
    "Small I/O (2KB, shared file)"

run_config 2 \
    "$IO500_DIR/configs_ion/config-ion-2_Aligned_IO.ini" \
    "Aligned I/O (1MB, shared file)"

run_config 3 \
    "$IO500_DIR/configs_ion/config-ion-3_Optimized.ini" \
    "Optimized (1MB, file-per-proc)"

run_config 4 \
    "$IO500_DIR/configs_ion/config-ion-4_Random_Small.ini" \
    "Random Small I/O (ior-hard)"

run_config 5 \
    "$IO500_DIR/configs_ion/config-ion-5_4K_Random.ini" \
    "4KB Random I/O"

run_config 6 \
    "$IO500_DIR/configs_ion/config-ion-6_Metadata.ini" \
    "Metadata Operations"

# ============================================
# GENERATE SUMMARY
# ============================================
echo ""
echo "=================================================="
echo "GENERATING SUMMARY"
echo "=================================================="

cat > "$RESULTS_BASE/FINAL_SUMMARY.txt" <<EOF
IO500 ION Benchmark Suite - Final Results
==========================================
Job ID: $SLURM_JOB_ID
Date: $(date)
System: Delta HPC
Nodes: $SLURM_JOB_NUM_NODES
Tasks: $SLURM_NTASKS

Configuration Descriptions:
---------------------------
Config1: Small I/O (2KB, shared file)
Config2: Aligned I/O (1MB, shared file)
Config3: Optimized (1MB, file-per-proc)
Config4: Random Small I/O (ior-hard)
Config5: 4KB Random I/O
Config6: Metadata Operations

Results Directory Structure:
-----------------------------
EOF

for i in {1..6}; do
    echo "" >> "$RESULTS_BASE/FINAL_SUMMARY.txt"
    echo "Config${i}:" >> "$RESULTS_BASE/FINAL_SUMMARY.txt"
    
    if [ -f "$RESULTS_BASE/Config${i}/Config${i}_${SLURM_JOB_ID}.darshan" ]; then
        echo "  ✓ Darshan log: Config${i}_${SLURM_JOB_ID}.darshan" >> "$RESULTS_BASE/FINAL_SUMMARY.txt"
    else
        echo "  ✗ Darshan log: MISSING" >> "$RESULTS_BASE/FINAL_SUMMARY.txt"
    fi
    
    if [ -f "$RESULTS_BASE/Config${i}/Config${i}_${SLURM_JOB_ID}_parsed.txt" ]; then
        echo "  ✓ Parsed text: Config${i}_${SLURM_JOB_ID}_parsed.txt" >> "$RESULTS_BASE/FINAL_SUMMARY.txt"
    else
        echo "  ✗ Parsed text: MISSING" >> "$RESULTS_BASE/FINAL_SUMMARY.txt"
    fi
    
    if [ -f "$RESULTS_BASE/Config${i}/Config${i}_parsed.csv" ]; then
        echo "  ✓ CSV: Config${i}_parsed.csv" >> "$RESULTS_BASE/FINAL_SUMMARY.txt"
    else
        echo "  ✗ CSV: MISSING" >> "$RESULTS_BASE/FINAL_SUMMARY.txt"
    fi
    
    # Extract performance if available
    if [ -f "$RESULTS_BASE/Config${i}/darshan_metrics.txt" ]; then
        echo "  Performance:" >> "$RESULTS_BASE/FINAL_SUMMARY.txt"
        grep "agg_perf_by_slowest" "$RESULTS_BASE/Config${i}/darshan_metrics.txt" | head -1 | \
            sed 's/^/    /' >> "$RESULTS_BASE/FINAL_SUMMARY.txt"
    fi
done

echo ""
echo "=================================================="
echo "BENCHMARK COMPLETE"
echo "=================================================="
echo "Results directory: $RESULTS_BASE"
echo "Summary: $RESULTS_BASE/FINAL_SUMMARY.txt"
echo ""
cat "$RESULTS_BASE/FINAL_SUMMARY.txt"
echo "=================================================="
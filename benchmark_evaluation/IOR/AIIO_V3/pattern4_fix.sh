#!/bin/bash
#SBATCH --job-name=ior_pattern4_fix
#SBATCH --nodes=2
#SBATCH --ntasks=256
#SBATCH --time=00:30:00
#SBATCH --partition=cpu
#SBATCH --account=bdau-delta-cpu
#SBATCH --output=IOR_Bench/AIIO_V3/Pattern4/pattern4_fix_%j.out
#SBATCH --error=IOR_Bench/AIIO_V3/Pattern4/pattern4_fix_%j.err

# ============================================
# CONFIGURATION
# ============================================
BASE_DIR="/work/hdd/bdau/mbanisharifdehkordi/E2E"
PATTERN_DIR="$BASE_DIR/IOR_Bench/AIIO_V3/Pattern4"
DARSHAN_LOG_BASE="$BASE_DIR/darshan_logs"
DARSHAN_HOME="$HOME/darshan-fresh"
PARSER_SCRIPT="$BASE_DIR/evaluation/parser.py"
SAMPLE_CSV="$BASE_DIR/evaluation/sample_train_100.csv"

IOR_BIN="$HOME/.conda/envs/ior_env/bin/ior"
LIBDARSHAN="$DARSHAN_HOME/lib/libdarshan.so"
PYTHON_ENV="/u/mbanisharifdehkordi/.conda/envs/gnn4_env/bin/python"

# ============================================
# LOAD MODULES
# ============================================
module load gcc/11.4.0 openmpi/4.1.6

# ============================================
# SETUP DARSHAN ENVIRONMENT (AGGRESSIVE)
# ============================================
export PATH="$DARSHAN_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$DARSHAN_HOME/lib:$LD_LIBRARY_PATH"
export DARSHAN_HOME

# Create today's log directory
mkdir -p "$DARSHAN_LOG_BASE/$(date +%Y)/$(date +%-m)/$(date +%-d)"

# CRITICAL: Force Darshan to log everything
export DARSHAN_LOG_DIR="$DARSHAN_LOG_BASE"
export DARSHAN_LOGPATH="$DARSHAN_LOG_DIR"
export DARSHAN_DISABLE_SHARED_REDUCTION=1
export DARSHAN_DISABLE_TIMING=0
export DXT_ENABLE_IO_TRACE=1

# Force all modules to be enabled
export DARSHAN_ENABLE_NONMPI=1
export DARSHAN_INTERNAL_TIMING=1

# Disable any optimization that might prevent logging
export ROMIO_HINTS="romio_no_indep_rw=false"
export HDF5_USE_FILE_LOCKING=FALSE

echo "=================================================="
echo "PATTERN 4 FIX: Noncontiguous Reading with Fixed Stride"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "Darshan: $DARSHAN_LOG_BASE"
echo "Pattern Dir: $PATTERN_DIR"
echo ""

cd "$PATTERN_DIR"

# Setup Lustre striping
lfs setstripe . -c 1 -S 1M

# ============================================
# STEP 1: Write the test file (if not exists)
# ============================================
if [ ! -f "./ior_pattern4_file.00000000" ]; then
    echo "Creating test file with strided write..."
    LD_PRELOAD="$LIBDARSHAN" mpirun -np 256 $IOR_BIN \
        -a POSIX \
        -w \
        -t 1k \
        -b 1k \
        -s 1024 \
        -k \
        -o ./ior_pattern4_file \
        -v
    
    echo "Write complete. Waiting and cleaning Darshan write log..."
    sleep 10
    
    # Remove write log
    find $DARSHAN_LOG_BASE -name "*.darshan" -type f -mmin -10 -delete
    
    # List files created
    echo "Files created:"
    ls -lh ./ior_pattern4_file*
else
    echo "Test file already exists, skipping write phase"
fi

echo ""
echo "=================================================="
echo "STEP 2: Run strided READ test"
echo "=================================================="
echo "Configuration: ior -r -t 1k -b 1k -s 1024"
echo "Expected: ~65 MiB/s with POSIX_SEEKS and misalignment"
echo ""

# Clear any old logs
find $DARSHAN_LOG_BASE -name "*.darshan" -type f -mmin -10 -delete

# Verify Darshan is loaded
echo "Verifying Darshan interception:"
ldd $IOR_BIN | grep darshan || echo "WARNING: Darshan not in ldd, but LD_PRELOAD should handle it"
echo "LD_PRELOAD=$LIBDARSHAN"
echo ""

# Run the READ test with Darshan
echo "Running READ test..."
LD_PRELOAD="$LIBDARSHAN" mpirun -np 256 $IOR_BIN \
    -a POSIX \
    -r \
    -t 1k \
    -b 1k \
    -s 1024 \
    -o ./ior_pattern4_file \
    -v

echo ""
echo "Read test complete. Waiting for Darshan to flush..."
sleep 15

# ============================================
# STEP 3: Find and Process Darshan Log
# ============================================
echo ""
echo "=================================================="
echo "STEP 3: Processing Darshan Log"
echo "=================================================="

# Search for Darshan log more aggressively
echo "Searching for Darshan logs..."
DARSHAN_FILE=$(find $DARSHAN_LOG_BASE -name "*.darshan" -type f -mmin -15 2>/dev/null | tail -1)

if [ -z "$DARSHAN_FILE" ]; then
    echo "❌ ERROR: No Darshan log found!"
    echo ""
    echo "Debugging information:"
    echo "1. Recent files in Darshan log directory:"
    find $DARSHAN_LOG_BASE -type f -mmin -15 -ls
    echo ""
    echo "2. Checking if Darshan was preloaded:"
    echo "   LD_PRELOAD was: $LIBDARSHAN"
    echo ""
    echo "3. Darshan library exists:"
    ls -lh "$LIBDARSHAN"
    echo ""
    echo "4. Environment variables:"
    env | grep DARSHAN
    echo ""
    echo "SUGGESTIONS:"
    echo "- Check if Darshan is properly compiled with MPI support"
    echo "- Try running: darshan-parser --help (to verify Darshan installation)"
    echo "- Check: $DARSHAN_LOG_BASE permissions"
    echo "- Consider recompiling IOR with Darshan statically linked"
    exit 1
fi

echo "✓ Found Darshan log: $DARSHAN_FILE"

# Copy to pattern directory
FINAL_LOG="$PATTERN_DIR/ior_Pattern4_${SLURM_JOB_ID}.darshan"
cp "$DARSHAN_FILE" "$FINAL_LOG"

echo "✓ Copied to: $FINAL_LOG"

# Parse to text
echo "Parsing to text format..."
$DARSHAN_HOME/bin/darshan-parser --show-incomplete "$FINAL_LOG" > "${FINAL_LOG%.darshan}_parsed.txt"

# Extract key metrics
echo "Extracting key metrics..."
cat > "$PATTERN_DIR/metrics.txt" <<EOF
=== Performance Metrics for Pattern4 ===
IOR Config: ior -r -t 1k -b 1k -s 1024
Date: $(date)

EOF

$DARSHAN_HOME/bin/darshan-parser "$FINAL_LOG" | \
    grep -E "agg_perf_by_slowest|POSIX_WRITES|POSIX_READS|POSIX_SIZE|POSIX_STRIDE|POSIX_FILE_NOT_ALIGNED|POSIX_FILE_ALIGNMENT|POSIX_SEEKS" \
    >> "$PATTERN_DIR/metrics.txt" 2>/dev/null

echo "Key metrics saved to: $PATTERN_DIR/metrics.txt"

# ============================================
# STEP 4: Convert to CSV
# ============================================
echo ""
echo "=================================================="
echo "STEP 4: Converting to CSV format"
echo "=================================================="

TEMP_DIR="/tmp/darshan_parse_pattern4_${SLURM_JOB_ID}"
mkdir -p "$TEMP_DIR"

echo "Running parser..."
DARSHAN_HOME="$DARSHAN_HOME" $PYTHON_ENV "$PARSER_SCRIPT" \
    "$PATTERN_DIR" \
    "$PATTERN_DIR/parsed.csv" \
    "$SAMPLE_CSV" \
    "$TEMP_DIR" 2>&1 | tee "$PATTERN_DIR/parse.log"

rm -rf "$TEMP_DIR"

# ============================================
# STEP 5: Display Results
# ============================================
echo ""
echo "=================================================="
echo "RESULTS SUMMARY"
echo "=================================================="

if [ -f "$PATTERN_DIR/parsed_raw.csv" ]; then
    echo "✓ Successfully created CSV files"
    echo ""
    echo "Generated files:"
    ls -lh "$PATTERN_DIR"/*.{darshan,csv,txt} 2>/dev/null
    echo ""
    echo "CSV Content (first few lines):"
    head -3 "$PATTERN_DIR/parsed_raw.csv"
    echo ""
    echo "Expected issues to see in CSV:"
    echo "  - High POSIX_SEEKS count"
    echo "  - POSIX_FILE_NOT_ALIGNED > 0"
    echo "  - Small POSIX_SIZE_READ values (0-100 or 100-1K bins)"
    echo "  - Performance (tag) around 65-70 MiB/s"
else
    echo "❌ CSV generation failed. Check parse.log"
fi

echo ""
echo "=================================================="
echo "Pattern 4 Fix Complete!"
echo "=================================================="
#!/bin/bash
#SBATCH --job-name=e2e_decomp_mismatch
#SBATCH --nodes=2
#SBATCH --ntasks=64
#SBATCH --time=00:30:00
#SBATCH --partition=cpu
#SBATCH --account=bdau-delta-cpu
#SBATCH --output=decomp_mismatch_%j.out
#SBATCH --error=decomp_mismatch_%j.err

# Load modules
module load gcc/11.4.0 openmpi/4.1.6 hdf5/1.14.3 netcdf-c/4.9.2

# Use the fresh Darshan build
LIBDARSHAN="$HOME/darshan-fresh/lib/libdarshan.so"

# Set ALL Darshan environment variables (like your working script)
export DARSHAN_LOG_DIR="/work/hdd/bdau/mbanisharifdehkordi/E2E/darshan_logs"
export DARSHAN_LOGPATH="$DARSHAN_LOG_DIR"
export DARSHAN_LOGDIR="$DARSHAN_LOG_DIR"
export DARSHAN_LOG_PATH="$DARSHAN_LOG_DIR"
export DARSHAN_DISABLE_SHARED_REDUCTION=1
export DXT_ENABLE_IO_TRACE=1

# Create the log directory with date structure
mkdir -p "$DARSHAN_LOG_DIR/$(date +%Y)/$(date +%-m)/$(date +%-d)"

cd /work/hdd/bdau/mbanisharifdehkordi/E2E/3d
mkdir -p results_decomp_mismatch
cd results_decomp_mismatch

# Use reasonable Lustre settings (not the bottleneck here)
lfs setstripe . -c 4 -S 1048576  # 4 stripes, 1MB each

echo "=== E2E DECOMPOSITION MISMATCH (Real Application Issue) ==="
echo "Issue: Poor data decomposition causes small, non-contiguous writes"
echo "This matches the ION/AIIO E2E evaluation"
echo "Start time: $(date)"

# Run with problematic decomposition (like ION paper)
# Using standard write_3d_nc4.c, NOT pathological version
LD_PRELOAD="$LIBDARSHAN" time mpirun -np 64 \
    ../write_3d_nc4 decomp_mismatch \
    32 32 16 \
    32 32 32

echo "End time: $(date)"
ls -lah decomp_mismatch.nc4

# Find the Darshan log
echo "Looking for Darshan log..."
DARSHAN_FILE=$(find $DARSHAN_LOG_DIR -name "*write_3d_nc4*" -type f -mmin -2 2>/dev/null | head -1)

if [ -z "$DARSHAN_FILE" ]; then
    DARSHAN_FILE=$(find $DARSHAN_LOG_DIR -type f -name "*.darshan" -mmin -2 2>/dev/null | head -1)
fi

if [ -n "$DARSHAN_FILE" ]; then
    FINAL_LOG="$DARSHAN_LOG_DIR/e2e_decomp_mismatch_${SLURM_JOB_ID}.darshan"
    cp "$DARSHAN_FILE" "$FINAL_LOG"
    echo "Darshan log saved as: $FINAL_LOG"
    
    echo ""
    echo "=== Darshan Log Summary ==="
    $HOME/darshan-fresh/bin/darshan-parser "$FINAL_LOG" | head -50
    
    echo ""
    echo "=== Expected Issues ==="
    echo "- High POSIX_SIZE_WRITE_100_1K (small writes)"
    echo "- High POSIX_FILE_NOT_ALIGNED (misaligned writes)"
    echo "- Many POSIX_WRITES operations"
    
    $HOME/darshan-fresh/bin/darshan-parser "$FINAL_LOG" | \
        grep -E "POSIX_SIZE_WRITE|POSIX_FILE_NOT_ALIGNED|POSIX_WRITES|agg_perf"
    
    echo ""
    echo "=== LUSTRE Configuration ==="
    $HOME/darshan-fresh/bin/darshan-parser "$FINAL_LOG" | grep -A10 "LUSTRE module data"
    
    $HOME/darshan-fresh/bin/darshan-parser "$FINAL_LOG" > "${FINAL_LOG%.darshan}_parsed.txt"
    echo "Parsed output saved to: ${FINAL_LOG%.darshan}_parsed.txt"
    
    if [ -f "$FINAL_LOG" ]; then
        rm "$DARSHAN_FILE"
        echo "Original log removed from dated subdirectory"
    fi
else
    echo "WARNING: No Darshan log found!"
    echo "Checking all recent files in $DARSHAN_LOG_DIR:"
    find $DARSHAN_LOG_DIR -type f -mmin -5 -ls
fi
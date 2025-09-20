#!/bin/bash
#SBATCH --job-name=e2e_decomp_fixed_proper
#SBATCH --nodes=2
#SBATCH --ntasks=64
#SBATCH --time=00:30:00
#SBATCH --partition=cpu
#SBATCH --account=bdau-delta-cpu
#SBATCH --output=decomp_fixed_proper_%j.out
#SBATCH --error=decomp_fixed_proper_%j.err

# Load modules
module load gcc/11.4.0 openmpi/4.1.6 hdf5/1.14.3 netcdf-c/4.9.2

# Use the fresh Darshan build
LIBDARSHAN="$HOME/darshan-fresh/lib/libdarshan.so"

# Set Darshan environment variables
export DARSHAN_LOG_DIR="/work/hdd/bdau/mbanisharifdehkordi/E2E/darshan_logs"
export DARSHAN_LOGPATH="$DARSHAN_LOG_DIR"
export DARSHAN_LOGDIR="$DARSHAN_LOG_DIR"
export DARSHAN_LOG_PATH="$DARSHAN_LOG_DIR"
export DARSHAN_DISABLE_SHARED_REDUCTION=1
export DXT_ENABLE_IO_TRACE=1

# ENABLE PROPER I/O OPTIMIZATIONS
export ROMIO_HINTS="romio_no_indep_rw=true"
export ROMIO_CB_READ="enable"
export ROMIO_CB_WRITE="enable"

# Enable collective I/O for better performance
export OMPI_MCA_fcoll="dynamic_gen2"
export OMPI_MCA_io_romio321_version_request=321

# Create the log directory
mkdir -p "$DARSHAN_LOG_DIR/$(date +%Y)/$(date +%-m)/$(date +%-d)"

cd /work/hdd/bdau/mbanisharifdehkordi/E2E/3d
mkdir -p results_decomp_fixed_proper
cd results_decomp_fixed_proper

# Optimal Lustre striping for parallel I/O
lfs setstripe . -c 16 -S 1048576  # 16 stripes, 1MB stripe size

echo "Start time: $(date)"

# FIXED CONFIGURATION - Properly matched decomposition
# Using 64x64x64 total size with 4x4x4 decomposition for 64 ranks
# This creates perfect alignment: each rank writes a 16x16x16 contiguous block
# 64/4 = 16 per dimension per rank - perfectly divisible
LD_PRELOAD="$LIBDARSHAN" time mpirun -np 64 \
    --mca io romio321 \
    --mca fcoll dynamic_gen2 \
    ../write_3d_nc4 decomp_fixed_proper \
    64 64 64 \
    4 4 4

echo "End time: $(date)"
ls -lah decomp_fixed_proper.nc4

# Find and process the Darshan log
DARSHAN_FILE=$(find $DARSHAN_LOG_DIR -name "*write_3d_nc4*" -type f -mmin -2 2>/dev/null | head -1)
if [ -z "$DARSHAN_FILE" ]; then
    DARSHAN_FILE=$(find $DARSHAN_LOG_DIR -type f -name "*.darshan" -mmin -2 2>/dev/null | head -1)
fi

if [ -n "$DARSHAN_FILE" ]; then
    FINAL_LOG="$DARSHAN_LOG_DIR/e2e_decomp_fixed_proper_${SLURM_JOB_ID}.darshan"
    cp "$DARSHAN_FILE" "$FINAL_LOG"
    
    echo "=== Expected Good Performance Metrics ==="
    echo "Should see: Lower POSIX_WRITES, Larger write sizes, Better alignment"
    
    $HOME/darshan-fresh/bin/darshan-parser "$FINAL_LOG" | \
        grep -E "agg_perf_by_slowest|POSIX_WRITES|POSIX_SIZE_WRITE|POSIX_FILE_NOT_ALIGNED"
    
    $HOME/darshan-fresh/bin/darshan-parser "$FINAL_LOG" > "${FINAL_LOG%.darshan}_parsed.txt"
    
    if [ -f "$FINAL_LOG" ]; then
        rm "$DARSHAN_FILE"
    fi
fi
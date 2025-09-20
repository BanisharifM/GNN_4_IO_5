#!/bin/bash
#SBATCH --job-name=e2e_decomp_problematic
#SBATCH --nodes=2
#SBATCH --ntasks=64
#SBATCH --time=00:30:00
#SBATCH --partition=cpu
#SBATCH --account=bdau-delta-cpu
#SBATCH --output=decomp_problematic_%j.out
#SBATCH --error=decomp_problematic_%j.err

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

# FORCE POOR I/O BEHAVIOR - Disable optimizations
export ROMIO_HINTS="romio_no_indep_rw=false"
export MPICH_MPIIO_HINTS_DISPLAY=1
export MPICH_MPIIO_AGGREGATOR_PLACEMENT_DISPLAY=1
export OMPI_MCA_io_romio321_version_request=321  # Force ROMIO version
export ROMIO_PRINT_HINTS=1

# Disable collective buffering to force individual writes
export OMPI_MCA_fcoll="^dynamic_gen2"
export OMPI_MCA_fcoll="individual"
export ROMIO_CB_READ="disable"
export ROMIO_CB_WRITE="disable"

# Create the log directory
mkdir -p "$DARSHAN_LOG_DIR/$(date +%Y)/$(date +%-m)/$(date +%-d)"

cd /work/hdd/bdau/mbanisharifdehkordi/E2E/3d
mkdir -p results_decomp_problematic
cd results_decomp_problematic

# Use minimal Lustre striping to worsen performance
lfs setstripe . -c 1 -S 65536  # Single stripe, 64KB size (very small)

echo "Start time: $(date)"

# PROBLEMATIC CONFIGURATION - Exactly like ION paper
# This creates severe decomposition mismatch causing small scattered writes
# Using 16x16x16 data with 8x4x2 decomposition for 64 ranks
# This mismatch forces each rank to write tiny non-contiguous pieces
LD_PRELOAD="$LIBDARSHAN" time mpirun -np 64 \
    --mca io romio321 \
    --mca fs ufs \
    --mca fcoll individual \
    ../write_3d_nc4 decomp_problematic \
    16 16 16 \
    8 4 2

echo "End time: $(date)"
ls -lah decomp_problematic.nc4

# Find and process the Darshan log
DARSHAN_FILE=$(find $DARSHAN_LOG_DIR -name "*write_3d_nc4*" -type f -mmin -2 2>/dev/null | head -1)
if [ -z "$DARSHAN_FILE" ]; then
    DARSHAN_FILE=$(find $DARSHAN_LOG_DIR -type f -name "*.darshan" -mmin -2 2>/dev/null | head -1)
fi

if [ -n "$DARSHAN_FILE" ]; then
    FINAL_LOG="$DARSHAN_LOG_DIR/e2e_decomp_problematic_${SLURM_JOB_ID}.darshan"
    cp "$DARSHAN_FILE" "$FINAL_LOG"
    
    echo "=== Expected Poor Performance Metrics ==="
    echo "Should see: High POSIX_WRITES, Small write sizes, Many misaligned operations"
    
    $HOME/darshan-fresh/bin/darshan-parser "$FINAL_LOG" | \
        grep -E "agg_perf_by_slowest|POSIX_WRITES|POSIX_SIZE_WRITE|POSIX_FILE_NOT_ALIGNED"
    
    $HOME/darshan-fresh/bin/darshan-parser "$FINAL_LOG" > "${FINAL_LOG%.darshan}_parsed.txt"
    
    if [ -f "$FINAL_LOG" ]; then
        rm "$DARSHAN_FILE"
    fi
fi
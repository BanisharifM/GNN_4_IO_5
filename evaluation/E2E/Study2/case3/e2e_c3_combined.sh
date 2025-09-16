#!/bin/bash
#SBATCH --job-name=e2e_combined
#SBATCH --nodes=2
#SBATCH --ntasks=64
#SBATCH --time=00:30:00
#SBATCH --partition=cpu
#SBATCH --account=bdau-delta-cpu
#SBATCH --output=combined_%j.out
#SBATCH --error=combined_%j.err

# Load modules
module load gcc/11.4.0 openmpi/4.1.6 hdf5/1.14.3 netcdf-c/4.9.2

# Use the fresh Darshan build
LIBDARSHAN="$HOME/darshan-fresh/lib/libdarshan.so"

# Set ALL Darshan environment variables
export DARSHAN_LOG_DIR="/work/hdd/bdau/mbanisharifdehkordi/E2E/darshan_logs"
export DARSHAN_LOGPATH="$DARSHAN_LOG_DIR"
export DARSHAN_LOGDIR="$DARSHAN_LOG_DIR"
export DARSHAN_LOG_PATH="$DARSHAN_LOG_DIR"
export DARSHAN_DISABLE_SHARED_REDUCTION=1
export DXT_ENABLE_IO_TRACE=1

# Create the log directory with date structure
mkdir -p "$DARSHAN_LOG_DIR/$(date +%Y)/$(date +%-m)/$(date +%-d)"

cd /work/hdd/bdau/mbanisharifdehkordi/E2E/3d
mkdir -p results_combined
cd results_combined

# Default Lustre settings (suboptimal but realistic)
lfs setstripe . -c 1 -S 1048576  # Default: 1 stripe, 1MB - not optimized for this workload

echo "=== E2E COMBINED ISSUES (Multiple Real Application Problems) ==="
echo "Issues: Poor decomposition + Checkpoint pattern + Default stripe config"
echo "This represents a realistic scenario where scientist uses defaults"
echo "Start time: $(date)"

# Checkpoint 1 with bad decomposition and default stripe
echo "Checkpoint X with poor decomposition and default stripe"
LD_PRELOAD="$LIBDARSHAN" mpirun -np 64 ../write_3d_nc4 combined_ckpt1 32 32 16 32 32 32

echo "Simulating computation..."
sleep 2

# Checkpoint 2
echo "Checkpoint 2 with poor decomposition and bad stripe"
LD_PRELOAD="$LIBDARSHAN" mpirun -np 64 ../write_3d_nc4 combined_ckpt2 32 32 16 32 32 32

echo "Simulating computation..."
sleep 2

# Checkpoint 3
echo "Checkpoint 3 with poor decomposition and bad stripe"
LD_PRELOAD="$LIBDARSHAN" mpirun -np 64 ../write_3d_nc4 combined_ckpt3 32 32 16 32 32 32

echo "End time: $(date)"
ls -lah combined_*.nc4

# Find the most recent Darshan log
echo "Looking for Darshan log..."
DARSHAN_FILE=$(find $DARSHAN_LOG_DIR -name "*write_3d_nc4*" -type f -mmin -2 2>/dev/null | tail -1)

if [ -z "$DARSHAN_FILE" ]; then
    DARSHAN_FILE=$(find $DARSHAN_LOG_DIR -type f -name "*.darshan" -mmin -2 2>/dev/null | tail -1)
fi

if [ -n "$DARSHAN_FILE" ]; then
    FINAL_LOG="$DARSHAN_LOG_DIR/e2e_combined_${SLURM_JOB_ID}.darshan"
    cp "$DARSHAN_FILE" "$FINAL_LOG"
    echo "Darshan log saved as: $FINAL_LOG"
    
    echo ""
    echo "=== Darshan Log Summary ==="
    $HOME/darshan-fresh/bin/darshan-parser "$FINAL_LOG" | head -50
    
    echo ""
    echo "=== Expected Multiple Issues ==="
    echo "1. Decomposition: High POSIX_FILE_NOT_ALIGNED, small writes"
    echo "2. Checkpoint: Burst pattern (from 3 runs)"
    echo "3. Stripe: Default LUSTRE_STRIPE_WIDTH (1), could benefit from multiple stripes"
    
    echo ""
    echo "=== POSIX Counters ==="
    $HOME/darshan-fresh/bin/darshan-parser "$FINAL_LOG" | \
        grep -E "POSIX_SIZE_WRITE|POSIX_FILE_NOT_ALIGNED|POSIX_BYTES_WRITTEN|agg_perf"
    
    echo ""
    echo "=== LUSTRE Configuration ==="
    $HOME/darshan-fresh/bin/darshan-parser "$FINAL_LOG" | \
        grep -E "LUSTRE_STRIPE"
    
    $HOME/darshan-fresh/bin/darshan-parser "$FINAL_LOG" > "${FINAL_LOG%.darshan}_parsed.txt"
    echo "Parsed output saved to: ${FINAL_LOG%.darshan}_parsed.txt"
    
    if [ -f "$FINAL_LOG" ]; then
        rm "$DARSHAN_FILE"
        echo "Original log removed from dated subdirectory"
    fi
    
    echo ""
    echo "Your gradient analysis should identify ALL THREE issues:"
    echo "- Application-level: decomposition causing small/misaligned writes"
    echo "- Pattern-level: checkpoint/burst I/O behavior"
    echo "- System-level: default stripe configuration (not optimized)"
else
    echo "WARNING: No Darshan log found!"
    echo "Checking all recent files in $DARSHAN_LOG_DIR:"
    find $DARSHAN_LOG_DIR -type f -mmin -5 -ls
fi
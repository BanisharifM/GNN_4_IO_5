#!/bin/bash
#SBATCH --job-name=e2e_checkpoint
#SBATCH --nodes=2
#SBATCH --ntasks=64
#SBATCH --time=00:30:00
#SBATCH --partition=cpu
#SBATCH --account=bdau-delta-cpu
#SBATCH --output=checkpoint_%j.out
#SBATCH --error=checkpoint_%j.err

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
mkdir -p results_checkpoint
cd results_checkpoint

# Good Lustre settings to isolate the checkpoint issue
lfs setstripe . -c 16 -S 4194304  # 16 stripes, 4MB each

echo "=== E2E CHECKPOINT PATTERN (Burst I/O Issue) ==="
echo "Issue: Simulating checkpoint/restart pattern with burst writes"
echo "Start time: $(date)"

# Simpler approach: Run E2E three times to simulate checkpoints
echo "Checkpoint 1: Writing initial state"
LD_PRELOAD="$LIBDARSHAN" mpirun -np 64 ../write_3d_nc4 checkpoint_1 64 8 8 16 16 16

echo "Simulating computation phase..."
sleep 2

echo "Checkpoint 2: Writing mid-state"
LD_PRELOAD="$LIBDARSHAN" mpirun -np 64 ../write_3d_nc4 checkpoint_2 64 8 8 16 16 16

echo "Simulating computation phase..."
sleep 2

echo "Checkpoint 3: Writing final state"
LD_PRELOAD="$LIBDARSHAN" mpirun -np 64 ../write_3d_nc4 checkpoint_3 64 8 8 16 16 16

echo "End time: $(date)"
ls -lah checkpoint_*.nc4

# Find the most recent Darshan log (will be from checkpoint_3)
echo "Looking for Darshan log..."
DARSHAN_FILE=$(find $DARSHAN_LOG_DIR -name "*write_3d_nc4*" -type f -mmin -2 2>/dev/null | tail -1)

if [ -z "$DARSHAN_FILE" ]; then
    DARSHAN_FILE=$(find $DARSHAN_LOG_DIR -type f -name "*.darshan" -mmin -2 2>/dev/null | tail -1)
fi

if [ -n "$DARSHAN_FILE" ]; then
    FINAL_LOG="$DARSHAN_LOG_DIR/e2e_checkpoint_${SLURM_JOB_ID}.darshan"
    cp "$DARSHAN_FILE" "$FINAL_LOG"
    echo "Darshan log saved as: $FINAL_LOG"
    
    echo ""
    echo "=== Darshan Log Summary ==="
    $HOME/darshan-fresh/bin/darshan-parser "$FINAL_LOG" | head -50
    
    echo ""
    echo "=== Expected Patterns ==="
    echo "- Large POSIX_BYTES_WRITTEN (burst writes)"
    echo "- Low POSIX_RW_SWITCHES (write-only pattern)"
    echo "- High POSIX_SIZE_WRITE_100K_1M (large writes)"
    
    $HOME/darshan-fresh/bin/darshan-parser "$FINAL_LOG" | \
        grep -E "POSIX_BYTES_WRITTEN|POSIX_RW_SWITCHES|POSIX_SIZE_WRITE_100K|agg_perf"
    
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
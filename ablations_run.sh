#!/bin/bash
#SBATCH --job-name=ablations_run
#SBATCH --output=run_ablations_%j.out
#SBATCH --error=run_ablations_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ivan.lee@u.nus.edu

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running All Variations Sequentially"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Create a dedicated log directory
LOG_DIR="runs/ablations_${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"

# Sync dependencies
echo "Syncing dependencies..."
uv sync || echo "uv sync skipped or failed, continuing..."

echo "Starting persistent run..."

while true; do
    echo "--- [$(date)] Polling for updates for all variations ---"
    
    # Run the script WITHOUT arguments. This triggers the Python 'else' block
    # to sequentially process A, B, C, and D safely in one thread.
    uv run run_gpt54_endowus_ablations.py 2>&1 | tee "$LOG_DIR/last_poll.log"
    PY_STATUS=${PIPESTATUS[0]}
    
    # Check if the script finished everything
    if grep -q "All tasks and variations completed successfully" "$LOG_DIR/last_poll.log"; then
        echo "=========================================="
        echo "SUCCESS: All Variations are 100% complete!"
        echo "Finished at $(date)"
        echo "=========================================="
        EXIT_CODE=0
        break
    fi

    # If the Python script crashed entirely (e.g. OOM, Syntax Error), stop looping.
    if [ $PY_STATUS -ne 0 ]; then
        echo "CRITICAL ERROR: Python script crashed with exit code $PY_STATUS. Halting loop."
        EXIT_CODE=$PY_STATUS
        break
    fi
    
    echo "Batches are still pending at OpenAI. Sleeping 15 minutes..."
    sleep 900
done

echo "=========================================="
exit $EXIT_CODE
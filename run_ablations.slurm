#!/bin/bash
#SBATCH --job-name=ablations_run
#SBATCH --output=runs/ablations_%A_%a/run.out
#SBATCH --error=runs/ablations_%A_%a/run.err
#SBATCH --time=72:00:00
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --array=0-3
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ivan.lee@u.nus.edu

# Map array ID to Variation letter (0->A, 1->B, 2->C, 3->D)
VARS=("A" "B" "C" "D")
VAR=${VARS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running Variation: $VAR"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Create output directory for this array task
mkdir -p runs/ablations_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

# Sync dependencies
echo "Syncing dependencies..."
uv sync || echo "uv sync skipped or failed, continuing..."

# Run the ablation for the specific variation with persistent polling
echo "Starting persistent run for Variation $VAR..."

while true; do
    echo "--- [$(date)] Polling for updates for Variation $VAR ---"
    
    # Run the script and capture output to check for completion
    # We use 'tee' so you can still see the output in run.out
    uv run run_gpt54_endowus_ablations.py $VAR 2>&1 | tee runs/ablations_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}/last_poll.log
    
    # Check if the script finished everything
    if grep -q "All tasks and variations completed successfully" runs/ablations_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}/last_poll.log; then
        echo "=========================================="
        echo "SUCCESS: Variation $VAR is 100% complete!"
        echo "Finished at $(date)"
        echo "=========================================="
        break
    fi
    
    echo "Variation $VAR still has pending batches or more steps. Sleeping 15 minutes..."
    sleep 900
done
echo "=========================================="
# Check exit status
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Successfully completed variation $VAR."
else
    echo "✗ Failed variation $VAR with exit code $EXIT_CODE"
fi

# Check exit status
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Successfully completed variation $VAR."
else
    echo "✗ Failed variation $VAR with exit code $EXIT_CODE"
fi

exit $EXIT_CODE

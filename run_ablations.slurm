#!/bin/bash
#SBATCH --job-name=ablations_run
#SBATCH --output=runs/ablations_%A_%a/run.out
#SBATCH --error=runs/ablations_%A_%a/run.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100-96:1
#SBATCH --mem=700G
#SBATCH --array=0-3

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

# Sync dependencies if using uv
echo "Syncing dependencies..."
uv sync || echo "uv sync skipped or failed, continuing..."

# Run the ablations script
echo "Starting ablation variation $VAR..."
uv run python run_ablations.py "$VAR" || python run_ablations.py "$VAR"

# Check exit status
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Successfully completed variation $VAR."
else
    echo "✗ Failed variation $VAR with exit code $EXIT_CODE"
fi

exit $EXIT_CODE

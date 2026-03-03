#!/bin/bash
#SBATCH --job-name=ablations_run
#SBATCH --output=runs/ablations_%A/run.out
#SBATCH --error=runs/ablations_%A/run.err
#SBATCH --time=48:00:00
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ivan.lee@u.nus.edu
#
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running: Multi-Agent Ablations (A, B, C, D)"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Create output directory for this job
mkdir -p runs/ablations_${SLURM_JOB_ID}

# Sync dependencies if using uv
echo "Syncing dependencies..."
uv sync || echo "uv sync skipped or failed, continuing..."

# Run the ablations script
echo "Starting ablations script..."
uv run python run_ablations.py || python run_ablations.py

# Check exit status
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Successfully completed ablations."
else
    echo "✗ Failed ablations with exit code $EXIT_CODE"
fi

exit $EXIT_CODE

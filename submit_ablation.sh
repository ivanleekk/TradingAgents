#!/bin/bash
#SBATCH --job-name=trading_ablation
#SBATCH --output=logs/ablation_%j.out
#SBATCH --error=logs/ablation_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00

# Create logs directory if it doesn't exist
mkdir -p logs

# Set UV cache to local directory to avoid permission issues on the cluster
export UV_CACHE_DIR=./.uv_cache

echo "Starting Ablation Study at $(date)"
echo "Running on node $(hostname)"

# Run the ablation study
# Note: This will run through variations A, B, C, D sequentially within one job.
uv run run_gpt54_endowus_ablations.py

echo "Ablation Study finished at $(date)"

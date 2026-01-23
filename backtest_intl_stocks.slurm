#!/bin/bash
#SBATCH --job-name=stock_backtest_intl_%a
#SBATCH --output=runs/%A_%a/run_%A_%a.out
#SBATCH --error=runs/%A_%a/run_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=h100-96:1
#SBATCH --mem=700G
#SBATCH --array=0-11

# Define international stocks (SGX, HKEX, LSE)
STOCKS=(
    # SGX (4)
    "D05.SI"
    "O39.SI"
    "Z74.SI"
    "U11.SI"
    # HKEX (4)
    "0700.HK"
    "9988.HK"
    "1398.HK"
    "1288.HK"
    # LSE (4)
    "AZN.L"
    "HSBA.L"
    "SHEL.L"
    "RR.L"
)

# Get the stock for this array task
STOCK=${STOCKS[$SLURM_ARRAY_TASK_ID]}

# Date range
START_DATE="2020-01-01"
END_DATE="2024-12-31"

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Stock: $STOCK"
echo "Date Range: $START_DATE to $END_DATE"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Create output directory for this job
mkdir -p runs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

# Sync dependencies
echo "Syncing dependencies..."
uv sync

# Run the backtest for this stock
echo "Starting backtest for $STOCK..."
uv run python run_stock_backtest.py "$STOCK" "$START_DATE" "$END_DATE"

# Check exit status
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Successfully completed backtest for $STOCK"
else
    echo "✗ Failed backtest for $STOCK with exit code $EXIT_CODE"
fi

exit $EXIT_CODE

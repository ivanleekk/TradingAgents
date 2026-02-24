#!/bin/bash
#SBATCH --job-name=stock_backtest_continue_%a
#SBATCH --output=runs/%A_%a/run_%A_%a.out
#SBATCH --error=runs/%A_%a/run_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=h100-96:1
#SBATCH --mem=700G
#SBATCH --array=0-23

# This script continues backtesting from the end dates found in results_bulk
# to 2024-12-31 for stocks that need updating
# Generated based on actual CSV data analysis

# Define stocks and their continuation dates
# Format: "STOCK:START_DATE" where START_DATE is one day after the last end date
declare -a STOCK_CONFIGS=(
    # Stock with 8 days remaining (ended 2024-12-23)
    "AAPL:2024-12-24"
    # Stocks with large gaps (need significant updates)
    "AMZN:2023-08-08"
    "BAC:2023-09-26"
    "COST:2023-10-17"
    "HD:2023-10-10"
    "KO:2023-12-05"
    "PG:2023-10-17"
    "TSLA:2023-08-22"
    "WMT:2023-10-31"
)

# Get the configuration for this array task
CONFIG=${STOCK_CONFIGS[$SLURM_ARRAY_TASK_ID]}

# Split configuration into stock and start date
IFS=':' read -r STOCK START_DATE <<< "$CONFIG"

# End date is always 2024-12-31
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
echo "Starting continuation backtest for $STOCK..."
uv run python run_stock_backtest.py "$STOCK" "$START_DATE" "$END_DATE"

# Check exit status
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Successfully completed continuation backtest for $STOCK"
else
    echo "✗ Failed continuation backtest for $STOCK with exit code $EXIT_CODE"
fi

exit $EXIT_CODE

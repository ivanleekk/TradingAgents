# Parallel Backtest Suite for TradingAgents

This directory contains scripts to run backtests for 32 stocks in parallel on a SLURM cluster.

## Overview

- **Date Range**: January 1, 2020 - December 31, 2024
- **Total Stocks**: 32 (20 US + 12 International)
- **Execution**: Parallel using SLURM job arrays

## Stock List

### US Stocks (20)

#### Technology (4)
- NVDA, AAPL, MSFT, AVGO

#### Healthcare (4)
- LLY, JNJ, ABBV, UNH

#### Financial Services (4)
- BRK-B, JPM, V, BAC

#### Consumer Cyclical (4)
- AMZN, TSLA, HD, MCD

#### Consumer Defensive (4)
- WMT, COST, PG, KO

### International Stocks (12)

#### Singapore Exchange (SGX) (4)
- D05.SI, O39.SI, Z74.SI, U11.SI

#### Hong Kong Exchange (HKEX) (4)
- 0700.HK, 9988.HK, 1398.HK, 1288.HK

#### London Stock Exchange (LSE) (4)
- AZN.L, HSBA.L, SHEL.L, RR.L

## Files

### Python Scripts
- `run_stock_backtest.py` - Main script that runs backtest for a single stock

### SLURM Scripts
- `backtest_us_stocks.slurm` - Job array for 20 US stocks (array=0-19)
- `backtest_intl_stocks.slurm` - Job array for 12 international stocks (array=0-11)
- `backtest_all_stocks.slurm` - Job array for all 32 stocks (array=0-31)

### Helper Scripts
- `submit_backtests.sh` - Easy submission script with options

## Usage

### Quick Start

Submit all 32 stocks in parallel:
```bash
./submit_backtests.sh all
```

Submit only US stocks:
```bash
./submit_backtests.sh us
```

Submit only international stocks:
```bash
./submit_backtests.sh intl
```

### Direct SLURM Submission

Submit all stocks:
```bash
sbatch backtest_all_stocks.slurm
```

Submit only US stocks:
```bash
sbatch backtest_us_stocks.slurm
```

Submit only international stocks:
```bash
sbatch backtest_intl_stocks.slurm
```

### Manual Single Stock Run

Run a single stock manually:
```bash
python run_stock_backtest.py AAPL 2020-01-01 2024-12-31
```

## Monitoring Jobs

Check job status:
```bash
squeue -u $USER
```

Check specific job array:
```bash
squeue -j JOB_ID
```

View detailed job info:
```bash
scontrol show job JOB_ID
```

Cancel a job:
```bash
scancel JOB_ID
```

Cancel specific array task:
```bash
scancel JOB_ID_TASK_ID
```

## Output

### Logs
Job logs are saved to:
- Standard output: `runs/JOB_ID_TASK_ID/run_JOB_ID_TASK_ID.out`
- Standard error: `runs/JOB_ID_TASK_ID/run_JOB_ID_TASK_ID.err`

### Results
Results are saved to:
- `results/{STOCK}_decisions_2020-01-01_2024-12-31.csv`

Each CSV contains:
- `stock`: Stock symbol
- `test_date`: Trading date (ISO format)
- `decision`: Trading decision for that date

### Checkpoints
The script saves checkpoints every 10 trading days:
- `results/{STOCK}_decisions_2020-01-01_2024-12-31_checkpoint.csv`

If a job fails, partial results are saved:
- `results/{STOCK}_decisions_2020-01-01_2024-12-31_partial.csv`

## Resource Allocation

Each job requests:
- **Time**: 24 hours
- **CPUs**: 4 cores
- **GPU**: 1x H100-96GB
- **Memory**: 700GB
- **Partition**: gpu

Adjust these in the `.slurm` files if needed:
```bash
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=h100-96:1
#SBATCH --mem=700G
```

## Troubleshooting

### Check Failed Jobs
```bash
# Find failed jobs
sacct -j JOB_ID --format=JobID,State,ExitCode

# View error log
cat runs/JOB_ID_TASK_ID/run_JOB_ID_TASK_ID.err
```

### Rerun Failed Tasks
If task 5 failed:
```bash
sbatch --array=5 backtest_all_stocks.slurm
```

If tasks 5, 10, 15 failed:
```bash
sbatch --array=5,10,15 backtest_all_stocks.slurm
```

### View Resource Usage
```bash
# View job efficiency
seff JOB_ID

# Detailed accounting
sacct -j JOB_ID --format=JobID,JobName,MaxRSS,Elapsed,State
```

## Expected Runtime

- **Trading days per stock**: ~1,258 weekdays (2020-2024)
- **Estimated time per stock**: Variable (depends on API calls and model inference)
- **Parallel execution**: All stocks run simultaneously
- **Total walltime**: ~24 hours (depending on system load)

## Notes

1. Ensure `.env` file contains necessary API keys before running
2. The script automatically skips weekends
3. Results include checkpoint saves every 10 days
4. Each stock runs independently - failures don't affect others
5. International stocks may require additional data sources to be configured

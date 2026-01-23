#!/bin/bash

# Helper script to submit backtest jobs to SLURM
# Usage: ./submit_backtests.sh [us|intl|all]

set -e

OPTION=${1:-all}

echo "=================================="
echo "TradingAgents Backtest Submission"
echo "=================================="
echo ""

case $OPTION in
    us)
        echo "Submitting US stocks only (20 stocks)..."
        echo "Stocks: NVDA, AAPL, MSFT, AVGO, LLY, JNJ, ABBV, UNH, BRK-B, JPM, V, BAC, AMZN, TSLA, HD, MCD, WMT, COST, PG, KO"
        echo ""
        JOB_ID=$(sbatch --parsable backtest_us_stocks.slurm)
        echo "✓ Submitted US stocks job array: $JOB_ID"
        echo "  Monitor with: squeue -j $JOB_ID"
        ;;
    
    intl)
        echo "Submitting International stocks only (12 stocks)..."
        echo "Stocks: D05.SI, O39.SI, Z74.SI, U11.SI, 0700.HK, 9988.HK, 1398.HK, 1288.HK, AZN.L, HSBA.L, SHEL.L, RR.L"
        echo ""
        JOB_ID=$(sbatch --parsable backtest_intl_stocks.slurm)
        echo "✓ Submitted International stocks job array: $JOB_ID"
        echo "  Monitor with: squeue -j $JOB_ID"
        ;;
    
    all)
        echo "Submitting ALL stocks (32 stocks total)..."
        echo "US Stocks: NVDA, AAPL, MSFT, AVGO, LLY, JNJ, ABBV, UNH, BRK-B, JPM, V, BAC, AMZN, TSLA, HD, MCD, WMT, COST, PG, KO"
        echo "International: D05.SI, O39.SI, Z74.SI, U11.SI, 0700.HK, 9988.HK, 1398.HK, 1288.HK, AZN.L, HSBA.L, SHEL.L, RR.L"
        echo ""
        JOB_ID=$(sbatch --parsable backtest_all_stocks.slurm)
        echo "✓ Submitted all stocks job array: $JOB_ID"
        echo "  Monitor with: squeue -j $JOB_ID"
        ;;
    
    *)
        echo "Error: Invalid option '$OPTION'"
        echo ""
        echo "Usage: ./submit_backtests.sh [us|intl|all]"
        echo ""
        echo "Options:"
        echo "  us    - Submit only US stocks (20 stocks)"
        echo "  intl  - Submit only International stocks (12 stocks)"
        echo "  all   - Submit all stocks (32 stocks) [default]"
        echo ""
        exit 1
        ;;
esac

echo ""
echo "Date Range: 2020-01-01 to 2024-12-31"
echo ""
echo "Monitor all jobs: squeue -u \$USER"
echo "Check specific job: squeue -j JOB_ID"
echo "Cancel job: scancel JOB_ID"
echo "View output: cat runs/JOB_ID_TASK_ID/run_JOB_ID_TASK_ID.out"
echo ""
echo "Results will be saved to: results/"
echo "=================================="

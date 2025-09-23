#!/bin/bash
#SBATCH --job-name=tradingAgents
#SBATCH --output=runs/%j/run_%j.out
#SBATCH --error=runs/%j/run_%j.err
#SBATCH --time=02:00:00

# Run the script
uv sync
uv run main.py

#!/bin/bash
#SBATCH --job-name=API_Trade
#SBATCH --output=runs/%j/run_%j.out
#SBATCH --error=runs/%j/run_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=long
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ivan.lee@u.nus.edu

# Run the script
uv sync
uv run main.py

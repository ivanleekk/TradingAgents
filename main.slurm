#!/bin/bash
#SBATCH --job-name=tradingAgents
#SBATCH --output=tradingAgents%j.out
#SBATCH --error=tradingAgents%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=h100-96:2
#SBATCH --mem=96G

# Run the script
uv sync
uv run main.py

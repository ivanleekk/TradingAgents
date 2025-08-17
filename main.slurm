#!/bin/bash
#SBATCH --job-name=vllm_test
#SBATCH --output=vllm_test%j.out
#SBATCH --error=vllm_test%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=h100:1
#SBATCH --mem=80G

# Run the script
uv sync
uv run main.py

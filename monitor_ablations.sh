#!/bin/bash
# Monitor the latest running slurm job for run_ablations.slurm

# Get the most recent array base job ID matching our job name
# Array jobs in squeue show as 12345_[0-3], so we extract just the base ID
JOB_ID=$(squeue -h -n ablations_run -o "%i" | head -n 1 | cut -d'_' -f1)

if [ -z "$JOB_ID" ]; then
    echo "No ablations_run job is currently running or pending."

    # Try to find the most recently created runs array directory instead
    # The array folders look like runs/ablations_123456_0
    LATEST_BASE_ID=$(ls -td runs/ablations_* 2>/dev/null | head -1 | grep -o 'ablations_[0-9]*' | grep -o '[0-9]*')

    if [ -n "$LATEST_BASE_ID" ]; then
        echo "Found recent array job ID: $LATEST_BASE_ID"
        echo "Tailing logs for all array tasks..."
        tail -f runs/ablations_${LATEST_BASE_ID}_*/run.out runs/ablations_${LATEST_BASE_ID}_*/run.err 2>/dev/null
    else
        echo "No recent runs found in runs/ablations_*"
    fi
    exit 0
fi

echo "Found running array job base ID: $JOB_ID"
echo "Tailing output logs for all 4 parallel variations..."
echo "Press Ctrl+C to stop monitoring."
echo "--------------------------------------------------------"

# Wait a few seconds to ensure the directories and files are created by SLURM
sleep 3

# Tail all standard out and standard error files for the array job
tail -f runs/ablations_${JOB_ID}_*/run.out runs/ablations_${JOB_ID}_*/run.err 2>/dev/null

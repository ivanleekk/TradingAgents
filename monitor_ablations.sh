#!/bin/bash
# Monitor the latest running slurm job for run_ablations.slurm

# Get the most recent job ID matching our job name
JOB_ID=$(squeue -h -n ablations_run -o "%i" | head -n 1)

if [ -z "$JOB_ID" ]; then
    echo "No ablations_run job is currently running or pending."

    # Try to find the most recently created runs directory instead
    LATEST_DIR=$(ls -td runs/ablations_* 2>/dev/null | head -1)
    if [ -n "$LATEST_DIR" ]; then
        echo "Found recent output directory: $LATEST_DIR"
        echo "Tailing the logs..."
        tail -f "$LATEST_DIR/run.out" "$LATEST_DIR/run.err"
    else
        echo "No recent runs found in runs/ablations_*"
    fi
    exit 0
fi

echo "Found running job: $JOB_ID"
echo "Tailing output logs at runs/ablations_${JOB_ID}/run.out ..."
echo "Press Ctrl+C to stop monitoring."
echo "--------------------------------------------------------"

# Wait a few seconds to ensure the directory and files are created by SLURM
sleep 2

# Tail both standard out and standard error
tail -f runs/ablations_${JOB_ID}/run.out runs/ablations_${JOB_ID}/run.err

#!/bin/bash

# Path to the Python script
script="dreamer.py"

# List of argument strings
arguments=(
    "--configs safegym --seed 10 --logdir ./logdir/safegym-seed-10-conditional-actions --conditional_actions true"
    "--configs safegym --seed 20 --logdir ./logdir/safegym-seed-20-conditional-actions --conditional_actions true"
    "--configs safegym --seed 30 --logdir ./logdir/safegym-seed-30-conditional-actions --conditional_actions true"
    "--configs safegym --seed 40 --logdir ./logdir/safegym-seed-40-conditional-actions --conditional_actions true"
    "--configs safegym --seed 50 --logdir ./logdir/safegym-seed-50-conditional-actions --conditional_actions true"
    "--configs safegym --seed 10 --logdir ./logdir/safegym-seed-10 --conditional_actions false"
    "--configs safegym --seed 20 --logdir ./logdir/safegym-seed-20 --conditional_actions false"
    "--configs safegym --seed 30 --logdir ./logdir/safegym-seed-30 --conditional_actions false"
    "--configs safegym --seed 40 --logdir ./logdir/safegym-seed-40 --conditional_actions false"
    "--configs safegym --seed 50 --logdir ./logdir/safegym-seed-50 --conditional_actions false"
)

log_names=(
    "safe-gym-seed-10-conditional-actions"
    "safe-gym-seed-20-conditional-actions"
    "safe-gym-seed-30-conditional-actions"
    "safe-gym-seed-40-conditional-actions"
    "safe-gym-seed-50-conditional-actions"
    "safe-gym-seed-10"
    "safe-gym-seed-20"
    "safe-gym-seed-30"
    "safe-gym-seed-40"
    "safe-gym-seed-50"
)

# Max number of parallel processes
max_parallel=2

# Function to run the script
run_script() {
    local args="$1"
    echo "Starting $script with arguments: $args"
    python $script $args
    if [ $? -ne 0 ]; then
        echo "Error: $script failed with arguments: $args"
        exit 1
    fi
}

# Start scripts with a process pool
running_processes=0
for i in "${!arguments[@]}"; do
    args="${arguments[i]}"
    log_file="$log_dir/${log_names[i]}.log"

    run_script "$args" "$log_file" &  # Run the script in the background with logging
    ((running_processes++))  # Increment the number of running processes

    # Wait if we've hit the max parallel limit
    if [ "$running_processes" -ge "$max_parallel" ]; then
        wait -n  # Wait for any process to finish
        ((running_processes--))  # Decrement the number of running processes
    fi
done

# Wait for all remaining processes to finish
wait

echo "All scripts completed."
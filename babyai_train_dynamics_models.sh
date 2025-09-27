#!/usr/bin/env bash

# Array of seeds
seeds=(100 200 300 400 500)

for seed in "${seeds[@]}"; do
    echo "Starting job with seed $seed..."
    python3.9 somniloquy.py \
              --configs babyai-language-100m-multiple-reward-heads \
              --seed "$seed" \
              --logdir "./logdir/babyai-language-100m-multiple-reward-heads-seed-$seed-bart-simple"
    echo "Job with seed $seed finished."
done

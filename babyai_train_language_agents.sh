#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <use_language_grads> <n_model_training_steps>"
    exit 1
fi

use_language_grads="$1"
n_training_steps="$2"
seeds=(100 200 300 400 500)
language_goals=("red" "green" "blue" "purple")

for seed in "${seeds[@]}"; do
    echo "Training agents for seed $seed..."
    for goal in "${language_goals[@]}"; do
        if [ "$use_language_grads" = "true" ]; then
            model_path="./logdir/babyai-language-100m-multiple-reward-heads-seed-$seed-bart-simple/latest.pt"
            logdir="./logdir/babyai-language-100m-multiple-reward-heads-seed-$seed-bart-simple-train-language-agent-goal-$goal"
        else
            model_path="./logdir/babyai-language-100m-multiple-reward-heads-seed-$seed-bart-simple-no-language-grads/latest.pt"
            logdir="./logdir/babyai-language-100m-multiple-reward-heads-seed-$seed-bart-simple-train-language-agent-goal-$goal-no-language-grads"
        fi

        echo "Training language agent for goal $goal"
        python3.9 train_language_agent_in_model.py \
                  --configs babyai-language-100m-multiple-reward-heads-train-agent \
                  --seed "$seed" \
                  --use_learned_reward "False" \
                  --logdir "$logdir" \
                  --checkpoint "$model_path" \
                  --language_goal "$goal" \
                  --model_steps "$n_training_steps" \
                  --language_grads "$use_language_grads"

        echo "Training learned reward agent for goal $goal"
        python3.9 train_language_agent_in_model.py \
                  --configs babyai-language-100m-multiple-reward-heads-train-agent \
                  --seed "$seed" \
                  --use_learned_reward "True" \
                  --logdir "$logdir-learned-reward" \
                  --checkpoint "$model_path" \
                  --language_goal "$goal" \
                  --model_steps "$n_training_steps" \
                  --language_grads "$use_language_grads"



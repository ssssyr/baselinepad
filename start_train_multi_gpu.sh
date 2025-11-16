#!/bin/bash

# --- Multi-GPU Training Quick Start Script ---

# 1. Set target GPUs
# Restrict training to GPUs 4, 5, 6, and 7
export CUDA_VISIBLE_DEVICES=4,5,6,7

# 2. Distributed Training Parameters
NUM_GPUS=4
# Set a master port for communication. Change if this port is in use.
# Find a random free port
MASTER_PORT=${PORT:-$(shuf -i 29500-49151 -n 1)}

# 3. Training Configuration
TRAIN_SCRIPT="train_robot.py"
# This script assumes you are training the MetaWorld action prediction model.
CONFIG_FILE="metaworld_4d.yaml"

# 4. Batch Size
# Set per-GPU batch size. Total batch size will be (BATCH_SIZE * NUM_GPUS).
# For MetaWorld with limited data, use smaller batch size.
# If you encounter Out-Of-Memory errors, reduce this value.
BATCH_SIZE=8

# 5. WandB Parameters (Optional, but recommended)
WANDB_PROJECT="metaworld_action_prediction_multigpu"
# Generates a unique run name with a timestamp.
WANDB_RUN_NAME="4xGPU-metaworld-action-prediction-bs$(($BATCH_SIZE * $NUM_GPUS))-run_$(date +%Y%m%d_%H%M%S)"

# --- DO NOT EDIT BELOW THIS LINE ---

echo "======================================================"
echo "Starting multi-GPU training with the following settings:"
echo "- GPUs: $CUDA_VISIBLE_DEVICES ($NUM_GPUS cards)"
echo "- Script: $TRAIN_SCRIPT"
echo "- Config: $CONFIG_FILE"
echo "- Per-GPU Batch Size: $BATCH_SIZE"
echo "- Total Batch Size: $(($BATCH_SIZE * $NUM_GPUS))"
echo "- WandB Project: $WANDB_PROJECT"
echo "- WandB Run: $WANDB_RUN_NAME"
echo "======================================================"

# Change to the script directory
cd "$(dirname "$0")" || exit 1

# Launch training using torchrun
torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT $TRAIN_SCRIPT \
    --config $CONFIG_FILE \
    --global-batch-size $(($BATCH_SIZE * $NUM_GPUS)) \
    --use-wandb \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "$WANDB_RUN_NAME"


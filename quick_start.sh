#!/bin/bash

# Quick Start - MetaWorld Action Prediction on 4x A100 (GPUs 4,5,6,7)

export CUDA_VISIBLE_DEVICES=4,5,6,7

torchrun --nproc_per_node=4 --master_port=29500 train_robot.py \
    --config metaworld_4d.yaml \
    --global-batch-size 32 \
    --use-wandb \
    --wandb-project "metaworld-action-prediction" \
    --wandb-run-name "4xA100-quick-start-$(date +%Y%m%d_%H%M%S)"

echo "✅ Training started! Check wandb for progress."
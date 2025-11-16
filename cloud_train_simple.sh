#!/bin/bash

# MetaWorld Training Script - Run on Cloud Server Directly

echo "🚀 Starting MetaWorld Training..."

# Set GPUs
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Configuration
CONFIG_FILE="metaworld_4d.yaml"
TRAIN_SCRIPT="train_robot.py"
BATCH_SIZE=64
EPOCHS=1000
LEARNING_RATE=1e-5

# Results directory
RESULTS_DIR="results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "📊 Configuration:"
echo "  - Config: $CONFIG_FILE"
echo "  - GPUs: $CUDA_VISIBLE_DEVICES"
echo "  - Batch Size: $BATCH_SIZE"
echo "  - Epochs: $EPOCHS"
echo "  - Results: $RESULTS_DIR"

# Check if files exist
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ ERROR: Config file '$CONFIG_FILE' not found in current directory!"
    echo "   Current directory: $(pwd)"
    echo "   Please copy metaworld_4d.yaml to this directory"
    exit 1
fi

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "❌ ERROR: Training script '$TRAIN_SCRIPT' not found in current directory!"
    echo "   Please copy train_robot.py to this directory"
    exit 1
fi

echo "✅ Files found, starting training..."

# Launch training
torchrun --nproc_per_node=4 --master_port=29500 "$TRAIN_SCRIPT" \
    --config "$CONFIG_FILE" \
    --global-batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --learning-rate "$LEARNING_RATE" \
    --results-dir "$RESULTS_DIR" \
    --use-wandb \
    --wandb-project "metaworld-action-prediction" \
    --wandb-run-name "4xA100-$(date +%Y%m%d_%H%M%S)"

echo "✅ Training completed! Results in: $RESULTS_DIR"
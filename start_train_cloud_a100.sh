#!/bin/bash

# =================================================================
#   MetaWorld Action Prediction Training - 4x A100 Cloud Server
#   GPUs: 4,5,6,7 | Config: metaworld_4d.yaml | Data: cloud paths
# =================================================================

echo "🚀 Starting MetaWorld Action Prediction Training on 4x A100..."
echo "📍 Config: metaworld_4d.yaml"
echo "🖥️  GPUs: 4,5,6,7 (A100)"
echo "📁 Data Path: /home/ct_24210860031/812datasets/SYR/feature_complete"
echo "================================================================"

# 1. GPU Configuration
export CUDA_VISIBLE_DEVICES=4,5,6,7
NUM_GPUS=4

# 2. Port Configuration
MASTER_PORT=${PORT:-$(shuf -i 29500-49151 -n 1)}
echo "🔌 Master Port: $MASTER_PORT"

# 3. Training Configuration
TRAIN_SCRIPT="train_robot.py"
CONFIG_FILE="metaworld_4d.yaml"

# 4. Batch Configuration (optimized for 4x A100 with limited data)
PER_GPU_BATCH_SIZE=16
TOTAL_BATCH_SIZE=$(($PER_GPU_BATCH_SIZE * $NUM_GPUS))
echo "📦 Batch Size: $PER_GPU_BATCH_SIZE per GPU → $TOTAL_BATCH_SIZE total"

# 5. Training Parameters (from config)
EPOCHS=1000
LEARNING_RATE=1e-5

# 6. Directories
RESULTS_DIR="results/metaworld_a100_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"
echo "💾 Results Dir: $RESULTS_DIR"

# 7. WandB Configuration
WANDB_PROJECT="metaworld-action-prediction"
WANDB_RUN_NAME="4xA100-metaworld-bs${TOTAL_BATCH_SIZE}-$(date +%Y%m%d_%H%M%S)"
echo "📊 WandB: $WANDB_PROJECT / $WANDB_RUN_NAME"

# 8. System Optimization for A100
export TORCH_CUDNN_V8_API_ENABLED=1
export NCCL_DEBUG=WARN

echo "================================================================"
echo "🎯 Training Configuration Summary:"
echo "   • Script: $TRAIN_SCRIPT"
echo "   • Config: $CONFIG_FILE"
echo "   • GPUs: $CUDA_VISIBLE_DEVICES ($NUM_GPUS cards)"
echo "   • Epochs: $EPOCHS"
echo "   • Learning Rate: $LEARNING_RATE"
echo "   • Global Batch Size: $TOTAL_BATCH_SIZE"
echo "   • Results: $RESULTS_DIR"
echo "   • WandB: $WANDB_RUN_NAME"
echo "================================================================"

# 9. Pre-flight Checks
echo "🔍 Pre-flight checks..."

# Check config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ ERROR: Config file '$CONFIG_FILE' not found!"
    exit 1
fi

# Check training script exists
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "❌ ERROR: Training script '$TRAIN_SCRIPT' not found!"
    exit 1
fi

# Check data directory (using feature_path from config)
FEATURE_PATH="/home/ct_24210860031/812datasets/SYR/feature_complete"
if [ ! -d "$FEATURE_PATH" ]; then
    echo "❌ ERROR: Feature data directory '$FEATURE_PATH' not found!"
    echo "   Please check your data paths in $CONFIG_FILE"
    exit 1
fi

# Check dataset.json exists
if [ ! -f "$FEATURE_PATH/dataset_rgb_s_d.json" ]; then
    echo "❌ ERROR: Dataset metadata '$FEATURE_PATH/dataset_rgb_s_d.json' not found!"
    exit 1
fi

echo "✅ All checks passed!"
echo "================================================================"

# 10. Change to correct directory and set Python path
SCRIPT_DIR="/home/ct_24210860031/812code/SYR/baselinepad"
echo "📁 Changing to script directory: $SCRIPT_DIR"
cd "$SCRIPT_DIR" || {
    echo "❌ ERROR: Cannot change to script directory $SCRIPT_DIR"
    exit 1
}

# Add current directory to Python path for relative imports
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
echo "🐍 Python path set to: $PYTHONPATH"

# 11. Launch Training
echo "🚀 Launching training..."
echo "Command: torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT $TRAIN_SCRIPT --config $CONFIG_FILE"

torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT $TRAIN_SCRIPT \
    --config "$CONFIG_FILE" \
    --global-batch-size "$TOTAL_BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --learning-rate "$LEARNING_RATE" \
    --results-dir "$RESULTS_DIR" \
    --use-wandb \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "$WANDB_RUN_NAME"

# 11. Post-training Summary
EXIT_CODE=$?
echo "================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "🎉 Training completed successfully!"
    echo "📁 Results saved to: $RESULTS_DIR"
    echo "📊 Check WandB for training metrics: https://wandb.ai/$WANDB_PROJECT"
else
    echo "❌ Training failed with exit code: $EXIT_CODE"
    echo "🔍 Check the logs above for error details"
fi
echo "================================================================"

exit $EXIT_CODE
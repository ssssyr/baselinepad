#!/bin/bash

# =================================================================
#   MetaWorld Action (Pose) Prediction Training - 4x A100
#   Project root, config, and script paths use ABSOLUTE paths.
# =================================================================

set -e

# ---- 0) 项目根目录（按你的实际路径） ----
SCRIPT_DIR="/home/ct_24210860031/812code/SYR/baselinepad"
CONFIG_FILE="$SCRIPT_DIR/configs/metaworld_4d.yaml"
TRAIN_SCRIPT="$SCRIPT_DIR/train_robot.py"

# ---- 1) 打印基础信息 ----
echo "🚀 Starting MetaWorld Action Prediction Training on 4x A100..."
echo "📁 Project: $SCRIPT_DIR"
echo "📍 Config:  $CONFIG_FILE"
echo "🖥️  GPUs:    4,5,6,7 (A100)"

# ---- 2) 数据与结果目录 ----
FEATURE_PATH="/home/ct_24210860031/812datasets/SYR/metaworld_features"  # 提取后的特征目录（含 dataset_rgb_s_d.json）
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$SCRIPT_DIR/results/metaworld_a100_${TIMESTAMP}"

echo "📁 Data Path:    $FEATURE_PATH"
echo "💾 Results Dir:  $RESULTS_DIR"
mkdir -p "$RESULTS_DIR"

# ---- 3) GPU/端口/批次设置 ----
export CUDA_VISIBLE_DEVICES=4,5,6,7
NUM_GPUS=4
MASTER_PORT=${PORT:-$(shuf -i 29500-49151 -n 1)}
echo "🔌 Master Port:  $MASTER_PORT"

# 每卡 batch 与全局 batch（传给 --global-batch-size；脚本内会按 GPU 数切分）
PER_GPU_BATCH_SIZE=16
TOTAL_BATCH_SIZE=$((PER_GPU_BATCH_SIZE * NUM_GPUS))
echo "📦 Batch Size:   $PER_GPU_BATCH_SIZE per GPU → $TOTAL_BATCH_SIZE total"

# ---- 4) 训练超参（YAML 文件优先）----
# 下面的参数已注释掉，将使用 config 文件中的设置。
# 如果需要从脚本指定，请取消注释并添加到下面的 torchrun 命令中。
# EPOCHS=1000
# LEARNING_RATE=1e-5

# ---- 4.5) Checkpoint 恢复设置 ----
CHECKPOINT_PATH=""
echo ""
read -p "🔄 Do you want to resume from checkpoint? (y/n): " RESUME_CHOICE
if [[ "$RESUME_CHOICE" =~ ^[Yy]$ ]]; then
    echo "📁 Please enter the full path to your checkpoint file:"
    echo "   Example: /home/ct_24210860031/812code/SYR/baselinepad/results/metaworld_a100_20251119_014323/000-DiT-XL-2-2025-11-19-01-43-46/checkpoints/0020000.pt"
    read -p "🎯 Checkpoint path: " CHECKPOINT_PATH
    
    if [ ! -f "$CHECKPOINT_PATH" ]; then
        echo "❌ ERROR: Checkpoint file '$CHECKPOINT_PATH' not found!"
        exit 1
    fi
    
    echo "✅ Found checkpoint: $CHECKPOINT_PATH"
    echo "🔄 Training will resume from this checkpoint..."
else
    echo "🆕 Starting fresh training (no checkpoint resume)"
fi

# ---- 5) W&B 配置（按需开启） ----
WANDB_PROJECT="metaworld-action-prediction"
WANDB_RUN_NAME="4xA100-metaworld-bs${TOTAL_BATCH_SIZE}-${TIMESTAMP}"
echo "📊 WandB:       $WANDB_PROJECT / $WANDB_RUN_NAME"

# ---- 6) 系统环境优化（可选）----
export TORCH_CUDNN_V8_API_ENABLED=1
export NCCL_DEBUG=WARN

echo "================================================================"
echo "🎯 Training Configuration Summary:"
echo "   • Script: $TRAIN_SCRIPT"
echo "   • Config: $CONFIG_FILE"
echo "   • GPUs:   $CUDA_VISIBLE_DEVICES ($NUM_GPUS cards)"
echo "   • Epochs: (from config)"
echo "   • LR:     (from config)"
echo "   • Global Batch Size: $TOTAL_BATCH_SIZE"
echo "   • Results: $RESULTS_DIR"
echo "   • WandB:  $WANDB_RUN_NAME"
echo "================================================================"

# ---- 7) 预检（使用绝对路径）----
echo "🔍 Pre-flight checks..."

if [ ! -f "$CONFIG_FILE" ]; then
  echo "❌ ERROR: Config file '$CONFIG_FILE' not found!"
  exit 1
fi

if [ ! -f "$TRAIN_SCRIPT" ]; then
  echo "❌ ERROR: Training script '$TRAIN_SCRIPT' not found!"
  exit 1
fi

if [ ! -d "$FEATURE_PATH" ]; then
  echo "❌ ERROR: Feature data directory '$FEATURE_PATH' not found!"
  exit 1
fi

if [ ! -f "$FEATURE_PATH/dataset_rgb_s_d.json" ]; then
  echo "❌ ERROR: '$FEATURE_PATH/dataset_rgb_s_d.json' not found!"
  exit 1
fi

echo "✅ All checks passed!"

# ---- 8) 进入项目目录并设置 PYTHONPATH ----
echo "📁 Changing to script directory: $SCRIPT_DIR"
cd "$SCRIPT_DIR" || { echo "❌ ERROR: Cannot change to $SCRIPT_DIR"; exit 1; }
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
echo "🐍 Python path set to: $PYTHONPATH"

# ---- 9) 启动训练（关键：显式传入 --config 与 --feature-path）----
echo "🚀 Launching training..."
echo "Command:"
echo "torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT $TRAIN_SCRIPT \\"
echo "  --config \"$CONFIG_FILE\" \\"
echo "  --feature-path \"$FEATURE_PATH\" \\"
echo "  --global-batch-size \"$TOTAL_BATCH_SIZE\" \\"

echo "  --results-dir \"$RESULTS_DIR\" \\"
echo "  --use-wandb \\"
echo "  --wandb-project \"$WANDB_PROJECT\" \\"
echo "  --wandb-run-name \"$WANDB_RUN_NAME\" \\"
echo "  --dynamics"

torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT "$TRAIN_SCRIPT" \
  --config "$CONFIG_FILE" \
  --feature-path "$FEATURE_PATH" \
  --global-batch-size "$TOTAL_BATCH_SIZE" \

  --results-dir "$RESULTS_DIR" \
  --use-wandb \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-run-name "$WANDB_RUN_NAME" \
  --dynamics \
  ${CHECKPOINT_PATH:+--resume "$CHECKPOINT_PATH"}

# ---- 10) 结束状态 ----
EXIT_CODE=$?
echo "================================================================"
if [ $EXIT_CODE -eq 0 ]; then
  echo "🎉 Training completed successfully!"
  echo "📁 Results saved to: $RESULTS_DIR"
else
  echo "❌ Training failed with exit code: $EXIT_CODE"
  echo "🔍 Check the logs above for error details"
fi
echo "================================================================"

exit $EXIT_CODE

# 云服务器 MetaWorld 动作预测训练指南

## 📋 已完成的配置

### 1. 配置文件更新 (`configs/metaworld_4d.yaml`)
- ✅ 数据路径已更新为云服务器地址: `/home/ct_24210860031/812datasets/SYR/feature_complete`
- ✅ 动作预测配置已设置 (4维绝对坐标)

### 2. 训练脚本创建
- ✅ `start_train_metaworld_cloud_simple.sh` - 推荐使用的简化版本
- ✅ `start_train_multi_gpu.sh` - 已更新为MetaWorld配置

## 🚀 快速开始

### 推荐训练命令
```bash
cd /home/syr/code/prediction_with_action
./start_train_metaworld_cloud_simple.sh
```

### 手动训练命令
```bash
cd /home/syr/code/prediction_with_action
torchrun --nproc_per_node=4 --master_port=29500 train_robot.py \
    --config metaworld_4d.yaml \
    --global-batch-size 32 \
    --epochs 1000 \
    --learning-rate 1e-4 \
    --results-dir "results/metaworld_$(date +%Y%m%d_%H%M%S)" \
    --ckpt-every 100 \
    --eval-every 50 \
    --use-wandb \
    --wandb-project "metaworld_action_prediction"
```

## ⚙️ 训练参数说明

### 针对小数据集优化 (2,829个样本)
- **Batch Size**: 8 (每GPU) → 32 (总计)
- **Epochs**: 1000 (更多迭代次数)
- **Learning Rate**: 1e-4 (较小学习率防止过拟合)
- **Checkpoint**: 每100轮保存
- **Evaluation**: 每50轮评估

### 云服务器设置
- **GPU**: 4个GPU (GPU 0,1,2,3)
- **Data Path**: `/home/ct_24210860031/812datasets/SYR/feature_complete`
- **Config**: `metaworld_4d.yaml`

## 📊 监控训练

### WandB 监控
训练会自动上传到 Weights & Biases:
- 项目: `metaworld_action_prediction`
- 运行名称: 包含时间戳的唯一名称

### 本地日志
- 检查点保存: `results/metaworld_YYYYMMDD_HHMMSS/`
- 训练日志: 终端输出

## 🛠️ 故障排除

### 常见问题
1. **CUDA内存不足**: 减少 `BATCH_SIZE` 从 8 到 4
2. **端口冲突**: 修改 `MASTER_PORT` (29500-49151)
3. **数据路径错误**: 确认 `/home/ct_24210860031/812datasets/SYR/feature_complete` 存在
4. **权限问题**: 运行 `chmod +x start_train_metaworld_cloud_simple.sh`

### 自定义参数
编辑 `start_train_metaworld_cloud_simple.sh` 中的变量:
- `BATCH_SIZE`: 根据GPU内存调整
- `EPOCHS`: 根据需要调整
- `CUDA_VISIBLE_DEVICES`: 选择可用GPU

## 📈 训练建议

### 由于数据量较小 (仅50个轨迹)
1. **密切监控过拟合**: 观察训练/验证损失差异
2. **早停策略**: 如果验证损失上升，提前停止训练
3. **数据增强**: 考虑在配置中启用更多数据增强
4. **正则化**: 确保启用了适当的正则化技术

### 评估策略
- 保留部分数据用于最终评估 (例如5个轨迹)
- 在训练过程中监控在未见数据上的表现

祝训练顺利！🎯
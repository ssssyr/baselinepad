# 🚀 MetaWorld 云服务器训练启动指南

## ⚡ 快速启动 (推荐)

### 1. 一键启动
```bash
cd /home/syr/code/prediction_with_action
./start_train_cloud_a100.sh
```

### 2. 极速启动 (最简)
```bash
cd /home/syr/code/prediction_with_action
./quick_start.sh
```

## 🎯 训练配置

- **GPU配置**: 4张A100 (GPU 4,5,6,7)
- **配置文件**: `metaworld_4d.yaml`
- **数据路径**: `/home/ct_24210860031/812datasets/SYR/feature_complete`
- **Batch Size**: 8/GPU → 32 总计
- **训练轮数**: 1000 epochs
- **学习率**: 1e-5
- **监控**: WandB自动上传

## 📊 监控训练进度

### 1. 本地监控
- 终端输出显示实时训练进度
- 检查点保存在 `results/metaworld_a100_YYYYMMDD_HHMMSS/`

### 2. WandB监控
- 项目: `metaworld-action-prediction`
- URL: https://wandb.ai/metaworld-action-prediction
- 实时查看损失曲线、指标变化

## 🔧 训练参数调整

### 如需修改参数，编辑 `start_train_cloud_a100.sh`:

```bash
# Batch Size (如果显存不足，减少这个值)
PER_GPU_BATCH_SIZE=8    # 默认8，可改为4或16

# 学习率 (如果训练不稳定，调整这个值)
LEARNING_RATE=1e-5      # 默认1e-5，可改为5e-6或2e-5

# 训练轮数 (根据需要调整)
EPOCHS=1000             # 默认1000
```

### 修改配置文件
```bash
vim metaworld_4d.yaml
```

## ⚠️ 故障排除

### 1. 显存不足
错误: `CUDA out of memory`
解决: 减少 `PER_GPU_BATCH_SIZE` 从8改为4

### 2. 端口冲突
错误: `Address already in use`
解决: 脚本会自动选择新端口，或手动指定
```bash
MASTER_PORT=29600 ./start_train_cloud_a100.sh
```

### 3. 数据路径错误
错误: `No such file or directory`
解决: 检查路径是否正确
```bash
ls -la /home/ct_24210860031/812datasets/SYR/feature_complete/
```

### 4. 权限问题
错误: `Permission denied`
解决: 添加执行权限
```bash
chmod +x start_train_cloud_a100.sh
```

## 📈 预期训练表现

### 对于小数据集 (2,829样本)
- **收敛时间**: ~100-300 epochs 可能收敛
- **过拟合风险**: 高，需监控验证损失
- **训练时长**: 4xA100 约1-3小时 (1000 epochs)

### 监控指标
- **损失下降**: 前50 epochs 快速下降
- **动作预测精度**: 应随训练提高
- **验证损失**: 如果开始上升，考虑早停

## 🎉 训练完成

### 检查输出
- **模型检查点**: `results/metaworld_a100_*/ckpts/`
- **训练日志**: 终端输出和WandB
- **最佳模型**: 通常在验证损失最低的检查点

### 下一步
1. 使用训练好的模型进行推理测试
2. 在测试集上评估性能
3. 考虑收集更多数据重新训练

祝训练顺利！🚀
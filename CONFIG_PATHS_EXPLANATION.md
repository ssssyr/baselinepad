# MetaWorld配置文件路径说明

## 🎯 关键结论

**训练时实际使用的是 `feature_path`**，这个路径必须正确！

## 📁 两个路径的详细说明

### 1. `training.feature_path` - 【重要】实际训练数据路径
```yaml
training:
  feature_path: "/home/ct_24210860031/812datasets/SYR/feature_complete"
```

**包含内容**:
- `dataset_rgb_s_d.json` - 主元数据文件
- `episode0000000/` - 处理好的轨迹数据
  - `color_wrist_1_0000.npy` - 图像特征 (1, 4, 32, 32)
  - `text_clip.npy` - 文本嵌入 (1, 512)
- `episode0000001/` - 其他轨迹...

**训练时的作用**:
- `dataset.py` 从这里加载特征数据
- 模型直接使用这些预处理的特征进行训练

### 2. `metaworld.data_path` - 【辅助】原始数据路径
```yaml
metaworld:
  data_path: "/home/ct_24210860031/812datasets/SYR/metaworld_raw"
```

**原始设计包含**:
- `class_000000/` - 原始图像文件夹
  - `0000.png` - 原始图像
  - `0001.png` - 原始图像
- `dataset_info.json` - 原始动作轨迹数据

**作用**:
- 主要用于数据预处理阶段
- 某些评估脚本可能需要原始数据

## ✅ 你的当前配置状态

```yaml
# 训练时实际使用 ✅
training:
  feature_path: "/home/ct_24210860031/812datasets/SYR/feature_complete"  # ✓ 正确

# 辅助路径 ⚠️
metaworld:
  data_path: "/home/ct_24210860031/812datasets/SYR/metaworld_raw"  # 可能不存在
```

## 🚀 训练前检查清单

### 必须存在:
- ✅ `/home/ct_24210860031/812datasets/SYR/feature_complete/dataset_rgb_s_d.json`
- ✅ `/home/ct_24210860031/812datasets/SYR/feature_complete/episode0000000/`

### 可选存在:
- ⭕ `/home/ct_24210860031/812datasets/SYR/metaworld_raw/` (不影响训练)

## 🔧 如果路径有问题

### 训练失败的常见错误:
```
FileNotFoundError: [Errno 2] No such file or directory: '/home/ct_24210860031/812datasets/SYR/feature_complete'
```

### 解决方案:
1. **确认feature_path存在**:
   ```bash
   ls -la /home/ct_24210860031/812datasets/SYR/feature_complete/
   ```

2. **如果不存在，检查数据是否在其他位置**:
   ```bash
   find /home/ct_24210860031/ -name "feature_complete" 2>/dev/null
   ```

3. **修改配置文件中的feature_path**:
   ```yaml
   training:
     feature_path: "/实际的/feature_complete/路径"
   ```

## 📝 最佳实践

1. **只关注feature_path**: 确保这个路径正确
2. **忽略data_path错误**: 如果这个路径不存在，不影响训练
3. **训练前验证**: 运行 `ls` 命令确认数据存在
4. **相对路径**: 考虑使用相对于项目根目录的路径

**总结**: 你的训练数据路径是 `feature_path`，确保这个正确即可！
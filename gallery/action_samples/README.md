# Action Samples - 动作评估样本

## 📂 目录说明

这个目录包含从Bridge原始数据集提取的**图像+动作序列**样本，用于评估模型的动作预测能力。

## 📊 数据格式

每个样本包含两个文件：

### 1. 图像文件 (`{sample_id}.jpg`)
- 原始Bridge数据集的RGB图像
- 尺寸: 640x480 (原始分辨率)
- 作为模型的输入

### 2. 动作文件 (`{sample_id}_actions.json`)
```json
{
  "sample_id": "open_microwave_01",
  "task": "open_microwave",
  "instruction": "open microwave",
  "frame_idx": 10,
  "actions": [
    [x1, y1, z1, roll1, pitch1, yaw1, gripper1],  // 未来第1步
    [x2, y2, z2, roll2, pitch2, yaw2, gripper2],  // 未来第2步
    [x3, y3, z3, roll3, pitch3, yaw3, gripper3]   // 未来第3步
  ],
  "action_description": "Future 3-step actions: [x, y, z, roll, pitch, yaw, gripper]"
}
```

**动作维度说明**：
- `x, y, z`: 末端执行器的3D位置（单位：米）
- `roll, pitch, yaw`: 末端执行器的姿态（单位：弧度）
- `gripper`: 抓取器状态 (0=打开, 1=关闭)

## 🎯 样本列表

| Sample ID | Task | Instruction | Frame | 说明 |
|-----------|------|-------------|-------|------|
| `open_microwave_01` | open_microwave | open microwave | 10 | 打开微波炉（动作早期） |
| `open_microwave_02` | open_microwave | open microwave | 12 | 打开微波炉（另一轨迹） |
| `close_microwave_01` | close_microwave | close microwave | 8 | 关闭微波炉（动作早期） |
| `close_microwave_02` | close_microwave | close microwave | 10 | 关闭微波炉（另一轨迹） |
| `pick_and_place_01` | pnp_push_sweep | pick and place | 8 | 抓取和放置（动作早期） |
| `stack_blocks_01` | stack_blocks | stack blocks | 12 | 堆叠方块（动作早期） |

**说明**：
- 所有样本选择动作的**早期帧**（帧索引8-12），不是快结束时
- 这样可以评估模型在动作初始阶段的预测能力
- Ground truth动作来自人工示教的trajectory

## 🔧 使用方法

### 评估动作预测

```bash
# 评估单个样本
python -m src.evaluation.evaluate_with_actions \
    --ckpt <checkpoint_path> \
    --sample open_microwave_01 \
    --output output/action_evaluation

# 评估所有样本
python -m src.evaluation.evaluate_with_actions \
    --ckpt <checkpoint_path> \
    --input input/action_samples \
    --output output/action_evaluation
```

### 评估指标

1. **动作MSE**: `np.mean((pred_action - gt_action)**2)`
   - 整体动作精度
   
2. **位置误差**: `np.linalg.norm(pred_xyz - gt_xyz)`
   - 3D位置的欧氏距离
   
3. **旋转误差**: 旋转角度差异
   
4. **抓取器准确率**: 抓取器状态是否正确

## 📝 元数据

所有样本的汇总信息保存在 `metadata.json`。

## 🔗 数据来源

- **数据集**: Bridge V2 Dataset
- **路径**: `/mnt/sda/datasets/bridge_dataset/raw/bridge_data_v2/`
- **提取脚本**: `scripts/extract_action_samples.py`

## ⚠️ 注意事项

1. **动作范围**：
   - 位置 (x,y,z): 通常在 [-0.1, 0.1] 范围内（相对移动）
   - 旋转: 通常在 [-0.1, 0.1] 弧度范围内
   - 抓取器: 0 或 1

2. **坐标系**：
   - 使用机器人基座坐标系
   - x: 前/后, y: 左/右, z: 上/下

3. **时间步**：
   - 每个动作对应约0.5秒的执行时间
   - 3步动作 ≈ 1.5秒的未来轨迹


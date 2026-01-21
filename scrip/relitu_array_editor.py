import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ============================================================
# 最直观的编辑方式 - 直接修改数组
# 每23帧一个关键点，共11个关键点控制整个曲线
# 0, 23, 46, 69, 92, 115, 138, 161, 184, 207, 230
# ============================================================

# 定义关键帧的权重值（直接修改这些数组！）
KEY_FRAMES = [0, 23, 46, 69, 92, 115, 138, 161, 184, 207, 230]

# Vision 权重 - 在下面直接修改数值
VISION_WEIGHTS = np.array([
    0.8,   # 帧 0
    0.9,   # 帧 23
    0.6,   # 帧 46
    0.4,   # 帧 69
    0.55,  # 帧 92 (开始下降)
    0.7,   # 帧 115 (遮挡中最低)
    0.9,   # 帧 138
    0.45,  # 帧 161
    0.65,  # 帧 184 (开始恢复)
    0.7,   # 帧 207
    0.8,   # 帧 230
])

# Depth 权重
DEPTH_WEIGHTS = np.array([
    0.4,   # 帧 0
    0.6,   # 帧 23
    0.4,   # 帧 46
    0.2,   # 帧 69
    0.35,  # 帧 92 (开始上升)
    0.6,   # 帧 115
    0.7,   # 帧 138 (遮挡中最高)
    0.3,   # 帧 161
    0.4,   # 帧 184 (开始下降)
    0.5,   # 帧 207
    0.5,   # 帧 230
])

# Force 权重
FORCE_WEIGHTS = np.array([
    0.1,   # 帧 0
    0.1,   # 帧 23
    0.4,   # 帧 46
    0.8,   # 帧 69
    0.2,   # 帧 92 (预接触)
    0.28,  # 帧 115
    0.68,  # 帧 138 (接触中上升)
    0.78,  # 帧 161
    0.4,   # 帧 184 (接触峰值)
    0.1,   # 帧 207
    0.1,   # 帧 230
])

# State 权重
STATE_WEIGHTS = np.array([
    0.5,   # 帧 0
    0.5,   # 帧 23
    0.7,   # 帧 46
    0.7,   # 帧 69
    0.675, # 帧 92 (略微上升)
    0.73,  # 帧 115
    0.75,  # 帧 138 (遮挡中最高)
    0.75,  # 帧 161
    0.675, # 帧 184 (开始恢复)
    0.5,   # 帧 207
    0.5,   # 帧 230
])

# ============================================================
# 自动插值生成完整数据 - 无需修改以下代码
# ============================================================

T = 230
all_frames = np.arange(T)

# 插值生成每帧的权重
vision_full = np.interp(all_frames, KEY_FRAMES, VISION_WEIGHTS)
depth_full = np.interp(all_frames, KEY_FRAMES, DEPTH_WEIGHTS)
force_full = np.interp(all_frames, KEY_FRAMES, FORCE_WEIGHTS)
state_full = np.interp(all_frames, KEY_FRAMES, STATE_WEIGHTS)

# 构建数据矩阵
data_matrix = np.vstack([vision_full, depth_full, force_full, state_full])

# ============================================================
# 绘图部分
# ============================================================

sns.set_style("whitegrid")
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 15,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
})

fig, ax = plt.subplots(figsize=(10, 2.8))

sns.heatmap(data_matrix,
            ax=ax,
            cmap='coolwarm',
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Normalized Weight', 'shrink': 0.8},
            xticklabels=False,
            yticklabels=['Vision', 'Depth', 'Force', 'State'])

ax.set_xlabel('Time Steps (Frames)', fontsize=16, fontweight='bold')
ax.set_ylabel('')
ax.set_xticks(KEY_FRAMES)
ax.set_xticklabels(KEY_FRAMES, rotation=0)
ax.set_yticklabels(['Vision', 'Depth', 'Force', 'State'], rotation=0, va='center')
ax.tick_params(left=False, bottom=False)

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)
    spine.set_color('#333333')

plt.tight_layout()

# 保存
output_dir = '/home/syr/code/prediction_with_action/docs'
os.makedirs(output_dir, exist_ok=True)

pdf_path = os.path.join(output_dir, 'modality_weights_heatmap.pdf')
png_path = os.path.join(output_dir, 'modality_weights_heatmap.png')

plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
plt.savefig(png_path, dpi=300, bbox_inches='tight', format='png')

print(f"热力图已保存:")
print(f"  PDF: {pdf_path}")
print(f"  PNG: {png_path}")
print(f"\n关键帧权重:")
print(f"帧: {KEY_FRAMES}")
print(f"Vision: {VISION_WEIGHTS}")
print(f"Depth:  {DEPTH_WEIGHTS}")
print(f"Force:  {FORCE_WEIGHTS}")
print(f"State:  {STATE_WEIGHTS}")

plt.show()

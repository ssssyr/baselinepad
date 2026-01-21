import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

# ============================================================
# 配置区域 - 在这里调整热力图
# ============================================================

# 总帧数
T = 230

# 定义每个模态的分段权重模式
# 格式: [(开始帧, 结束帧, 起始权重, 结束权重), ...]
# 使用 None 表示保持上一个权重

MODALITY_PATTERNS = {
    'Vision': [
        (0, 92, 0.8, 0.8),           # 0-92帧: 保持0.8
        (92, 116, 0.8, 0.1),         # 92-116帧: 0.8降到0.1
        (116, 184, 0.1, 0.1),        # 116-184帧: 保持0.1
        (184, 207, 0.1, 0.8),        # 184-207帧: 0.1升到0.8
        (207, T, 0.8, 0.8),          # 207-230帧: 保持0.8
    ],
    'Depth': [
        (0, 92, 0.1, 0.1),           # 0-92帧: 保持0.1
        (92, 116, 0.1, 0.6),         # 92-116帧: 0.1升到0.6
        (116, 138, 0.6, 0.7),        # 116-138帧: 0.6升到0.7
        (138, 184, 0.7, 0.7),        # 138-184帧: 保持0.7
        (184, 207, 0.7, 0.1),        # 184-207帧: 0.7降到0.1
        (207, T, 0.1, 0.1),          # 207-230帧: 保持0.1
    ],
    'Force': [
        (0, 92, 0.1, 0.1),           # 0-92帧: 保持0.1
        (92, 116, 0.1, 0.3),         # 92-116帧: 预接触
        (116, 196, 0.3, 0.8),        # 116-196帧: 接触阶段升到0.8
        (196, 207, 0.8, 0.8),        # 196-207帧: 保持峰值
        (207, T, 0.8, 0.1),          # 207-230帧: 降到0.1
    ],
    'State': [
        (0, 92, 0.6, 0.6),           # 0-92帧: 保持0.6
        (92, 138, 0.6, 0.75),        # 92-138帧: 升到0.75
        (138, 184, 0.75, 0.75),      # 138-184帧: 保持0.75
        (184, 207, 0.75, 0.6),       # 184-207帧: 降到0.6
        (207, T, 0.6, 0.6),          # 207-230帧: 保持0.6
    ],
}

# 遮挡区域标注
OCCLUSION_ZONES = [
    {'start': 92, 'end': 184, 'color': 'gray', 'alpha': 0.3, 'label': 'Visual Occlusion Event'}
]

# 接触区域标注
CONTACT_ZONES = [
    {'start': 116, 'end': 196, 'color': 'green', 'alpha': 0.15,
     'ymin': 0.4, 'ymax': 0.65, 'label': 'Force Contact'}
]

# ============================================================
# 以下代码自动生成权重矩阵，无需修改
# ============================================================

def generate_weights_from_patterns(patterns, T):
    """根据分段模式生成权重数组"""
    weights = np.zeros(T)
    current_frame = 0

    for start, end, weight_start, weight_end in patterns:
        # 填充间隔
        if start > current_frame:
            weights[current_frame:start] = weights[current_frame - 1] if current_frame > 0 else weight_start

        # 线性插值填充当前段
        length = end - start
        if length > 0:
            weights[start:end] = np.linspace(weight_start, weight_end, length)

        current_frame = max(current_frame, end)

    # 填充剩余部分
    if current_frame < T:
        weights[current_frame:] = weights[current_frame - 1] if current_frame > 0 else 0

    return weights

# 生成各模态权重
modality_weights = {}
for modality, pattern in MODALITY_PATTERNS.items():
    modality_weights[modality] = generate_weights_from_patterns(pattern, T)

# 构建数据矩阵
data_matrix = np.vstack([
    modality_weights['Vision'],
    modality_weights['Depth'],
    modality_weights['Force'],
    modality_weights['State']
])

# ============================================================
# 绘图部分
# ============================================================

# 设置样式
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

# 创建图表
fig, ax = plt.subplots(figsize=(10, 2.8))

# 绘制热力图
sns.heatmap(data_matrix,
            ax=ax,
            cmap='coolwarm',
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Normalized Weight', 'shrink': 0.8},
            xticklabels=False,
            yticklabels=['Vision', 'Depth', 'Force', 'State'])

# 添加遮挡区域标注
for zone in OCCLUSION_ZONES:
    ax.axvspan(zone['start'], zone['end'],
               color=zone['color'], alpha=zone['alpha'], ymin=0, ymax=1)
    if 'label' in zone:
        mid = (zone['start'] + zone['end']) / 2
        ax.text(mid, -0.15, zone['label'],
               horizontalalignment='center', verticalalignment='top',
               fontsize=11, fontweight='bold', color='#333333',
               transform=ax.transData)

# 添加接触区域标注
for zone in CONTACT_ZONES:
    ax.axvspan(zone['start'], zone['end'],
               color=zone['color'], alpha=zone['alpha'],
               ymin=zone.get('ymin', 0), ymax=zone.get('ymax', 1))
    if 'label' in zone:
        ax.text(zone['start'] + 2, 2.2, zone['label'],
               horizontalalignment='left', verticalalignment='top',
               fontsize=9, fontweight='bold', color='#2d5a27',
               transform=ax.transData)

# 坐标轴设置
ax.set_xlabel('Time Steps (Frames)', fontsize=12, fontweight='bold')
ax.set_ylabel('')
ax.set_xticks(np.arange(0, T + 1, step=23))
ax.set_xticklabels(np.arange(0, T + 1, step=23), rotation=0)
ax.set_yticklabels(['Vision', 'Depth', 'Force', 'State'], rotation=0, va='center')
ax.tick_params(left=False, bottom=False)

# 边框样式
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)
    spine.set_color('#333333')

# 保存图片
plt.tight_layout()
output_dir = '/home/syr/code/prediction_with_action/docs'
os.makedirs(output_dir, exist_ok=True)

pdf_path = os.path.join(output_dir, 'modality_weights_heatmap.pdf')
png_path = os.path.join(output_dir, 'modality_weights_heatmap.png')

plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
plt.savefig(png_path, dpi=300, bbox_inches='tight', format='png')

print(f"热力图已保存:")
print(f"  PDF: {pdf_path}")
print(f"  PNG: {png_path}")
print(f"\n权重范围:")
for modality in ['Vision', 'Depth', 'Force', 'State']:
    w = modality_weights[modality]
    print(f"  {modality}: [{w.min():.2f}, {w.max():.2f}]")

plt.show()

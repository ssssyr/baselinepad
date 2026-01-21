import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

# ============================================================
# CSV文件编辑模式 - 在Excel中编辑heatmap_data.csv
# ============================================================

# CSV文件路径
CSV_FILE = '/home/syr/code/prediction_with_action/scrip/heatmap_data.csv'

# 如果CSV不存在，从模板创建
template_data = {
    'Frame': list(range(0, 231, 23)),  # 0到230，每23帧一个关键点
    'Vision': [0.8, 0.8, 0.8, 0.8, 0.45, 0.1, 0.1, 0.1, 0.1, 0.45, 0.8],
    'Depth':  [0.1, 0.1, 0.1, 0.1, 0.35, 0.6, 0.7, 0.7, 0.4, 0.1, 0.1],
    'Force':  [0.1, 0.1, 0.1, 0.1, 0.2, 0.28, 0.48, 0.68, 0.8, 0.8, 0.1],
    'State':  [0.6, 0.6, 0.6, 0.6, 0.675, 0.73, 0.75, 0.75, 0.675, 0.6, 0.6],
}

if not os.path.exists(CSV_FILE):
    pd.DataFrame(template_data).to_csv(CSV_FILE, index=False)
    print(f"已创建CSV模板文件: {CSV_FILE}")
    print(f"请在Excel或其他工具中编辑该文件，然后重新运行此脚本")

# 读取CSV数据
df = pd.read_csv(CSV_FILE)
df = df.sort_values('Frame')

# 插值到完整帧数（230帧）
T = 230
all_frames = np.arange(T)
vision_full = np.interp(all_frames, df['Frame'], df['Vision'])
depth_full = np.interp(all_frames, df['Frame'], df['Depth'])
force_full = np.interp(all_frames, df['Frame'], df['Force'])
state_full = np.interp(all_frames, df['Frame'], df['State'])

# 构建数据矩阵
data_matrix = np.vstack([vision_full, depth_full, force_full, state_full])

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
ax.axvspan(92, 184, color='gray', alpha=0.3, ymin=0, ymax=1)
ax.text(138, -0.15, 'Visual Occlusion Event',
       horizontalalignment='center', verticalalignment='top',
       fontsize=11, fontweight='bold', color='#333333',
       transform=ax.transData)

# 添加接触区域标注
ax.axvspan(116, 196, color='green', alpha=0.15, ymin=0.4, ymax=0.65)
ax.text(118, 2.2, 'Force Contact',
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
print(f"\n从CSV文件读取: {CSV_FILE}")
print(f"权重范围:")
print(f"  Vision: [{vision_full.min():.2f}, {vision_full.max():.2f}]")
print(f"  Depth:  [{depth_full.min():.2f}, {depth_full.max():.2f}]")
print(f"  Force:  [{force_full.min():.2f}, {force_full.max():.2f}]")
print(f"  State:  [{state_full.min():.2f}, {state_full.max():.2f}]")

plt.show()

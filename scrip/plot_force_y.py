import matplotlib.pyplot as plt
import numpy as np
import os

# 读取数据
data_path = '/mnt/sda/datasets/real_data/Squeeze hand sanitizer foam from bottle./episode_0000.npz'
data = np.load(data_path)

# 提取force_torque数据 (53帧, 6维)
force_torque = data['force_torque']
# y轴力是索引1 (fx, fy, fz, tx, ty, tz)
fy = force_torque[:, 1]

# 生成时间轴 (假设每帧约0.1秒)
time = np.arange(len(fy)) * 0.1

# 绘图
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(time, fy, color='#1f77b4', linewidth=2)
ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
ax.set_ylabel('Force Y (N)', fontsize=12, fontweight='bold')
ax.set_title('Y-Axis Force over Time - Episode 0000', fontsize=14, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, linestyle='--')

# 添加零线
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

# 标注最大值和最小值
max_idx = np.argmax(fy)
min_idx = np.argmin(fy)
ax.plot(time[max_idx], fy[max_idx], 'ro', label=f'Max: {fy[max_idx]:.2f}N')
ax.plot(time[min_idx], fy[min_idx], 'go', label=f'Min: {fy[min_idx]:.2f}N')
ax.legend()

plt.tight_layout()

# 保存到docs目录
output_dir = '/home/syr/code/prediction_with_action/docs'
os.makedirs(output_dir, exist_ok=True)

pdf_path = os.path.join(output_dir, 'force_y_timeseries.pdf')
png_path = os.path.join(output_dir, 'force_y_timeseries.png')

plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
plt.savefig(png_path, dpi=300, bbox_inches='tight', format='png')

print(f"图表已保存:")
print(f"  PDF: {pdf_path}")
print(f"  PNG: {png_path}")
print(f"\n数据统计:")
print(f"  总帧数: {len(fy)}")
print(f"  Y轴力范围: [{fy.min():.2f}, {fy.max():.2f}] N")
print(f"  平均值: {fy.mean():.2f} N")
print(f"  标准差: {fy.std():.2f} N")

plt.show()

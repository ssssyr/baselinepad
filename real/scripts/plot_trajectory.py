#!/usr/bin/env python3
"""
绘制机械臂末端执行器的三维运动轨迹
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib as mpl


def setup_style():
    """设置科技感样式"""
    mpl.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 11,
        'axes.linewidth': 1.2,
        'axes.labelsize': 12,
        'axes.labelcolor': '#333333',
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '#cccccc',
        'grid.alpha': 0.3,
    })


def plot_trajectory(npz_path, output_path=None):
    """从npz文件读取并绘制xyz三维轨迹"""

    # 加载数据
    data = np.load(npz_path)
    poses = data['robot_pose']  # shape: (N, 6) -> [x, y, z, rx, ry, rz]

    # 提取xyz坐标
    x = poses[:, 0]
    y = poses[:, 1]
    z = poses[:, 2]

    # 创建3D图形 - 白色背景
    fig = plt.figure(figsize=(14, 11), facecolor='#ffffff')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#ffffff')

    # 科技感配色方案（渐变色）
    phases = [
        (0, 70, '#00d4ff', 'Phase I (0-70)'),      # 青蓝色
        (71, 128, '#00ff88', 'Phase II (71-128)'),  # 荧光绿
        (129, 175, '#ffcc00', 'Phase III (129-175)'), # 金黄色
        (175, len(x) - 1, '#ff6b6b', 'Phase IV (175-end)'),  # 珊瑚红
    ]

    # 绘制每个阶段（使用散点）
    for start, end, color, label in phases:
        if start < len(x):
            end = min(end, len(x) - 1)
            ax.scatter(x[start:end+1], y[start:end+1], z[start:end+1],
                      c=color, s=15, alpha=0.7, label=label, edgecolors='none')

    # 标记起点和终点
    ax.scatter([x[0]], [y[0]], [z[0]], c='#00cc00', s=200, marker='o',
               edgecolors='black', linewidths=2, label='Start', zorder=10,
               alpha=1.0)
    ax.scatter([x[-1]], [y[-1]], [z[-1]], c='#cc0000', s=250, marker='*',
               edgecolors='black', linewidths=2, label='End', zorder=10,
               alpha=1.0)

    # 标记阶段分界点
    phase_boundaries = [70, 128, 175]
    for idx in phase_boundaries:
        if idx < len(x):
            ax.scatter([x[idx]], [y[idx]], [z[idx]], c='#666666', s=120,
                      marker='D', edgecolors='black', linewidths=2,
                      alpha=1.0, zorder=8)

    # 设置标签和标题
    ax.set_xlabel('X (m)', fontsize=14, fontweight='bold', color='#333333')
    ax.set_ylabel('Y (m)', fontsize=14, fontweight='bold', color='#333333')
    ax.set_zlabel('Z (m)', fontsize=14, fontweight='bold', color='#333333')
    ax.set_title(f'End-Effector 3D Trajectory\n{Path(npz_path).name}',
                 fontsize=16, fontweight='bold', color='#333333', pad=20)

    # 设置相等的比例
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # 设置刻度标签颜色
    ax.tick_params(axis='x', colors='#333333', labelsize=10)
    ax.tick_params(axis='y', colors='#333333', labelsize=10)
    ax.tick_params(axis='z', colors='#333333', labelsize=10)

    # 设置网格样式
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, alpha=0.3, color='#cccccc', linestyle='--', linewidth=0.8)

    # 设置轴线颜色
    ax.xaxis.line.set_color('#333333')
    ax.yaxis.line.set_color('#333333')
    ax.zaxis.line.set_color('#333333')

    # 图例样式
    legend = ax.legend(loc='upper left', fontsize=10, framealpha=0.9,
                       facecolor='#ffffff', edgecolor='#333333')
    for text in legend.get_texts():
        text.set_color('#333333')
    legend.get_frame().set_linewidth(1.2)

    # 打印轨迹信息
    print(f"轨迹点数: {len(x)}")
    print(f"X范围: [{x.min():.3f}, {x.max():.3f}] m")
    print(f"Y范围: [{y.min():.3f}, {y.max():.3f}] m")
    print(f"Z范围: [{z.min():.3f}, {z.max():.3f}] m")
    print(f"总行程: {np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2)):.3f} m")

    # 调整视角
    ax.view_init(elev=20, azim=45)

    # 保存或显示
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#ffffff')
        print(f"\n图像已保存到: {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="绘制机械臂末端xyz三维轨迹")
    parser.add_argument('npz_path', type=str, help='npz文件路径')
    parser.add_argument('-o', '--output', type=str, default=None, help='输出图像路径（可选）')
    args = parser.parse_args()

    setup_style()
    plot_trajectory(args.npz_path, args.output)

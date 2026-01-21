import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# --- 1. 设置学术图表风格 ---
# 使用 seaborn 的 paper 风格，字体稍微调大以便阅读
sns.set_theme(style="ticks", context="paper", font_scale=1.35)  # 适度增大字体适应双栏单侧
# 设置西文字体为 Times New Roman 或类似衬线体，中文字体需另外设置(如果需要)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.linewidth'] = 1.2 # 加粗坐标轴线
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2


# --- 2. 生成模拟数据 (Synthetic Data Generation) ---
# 在实际应用中，请用你的实验数据替换这部分
def generate_mock_data():
    t = np.linspace(0, 0.8, 500) # 时间轴：0 到 0.8 秒
    t_contact = 0.15 # 接触发生在 0.15秒 (150ms)
    force_threshold = 15.0 # 设定的过载阈值 (N)

    # 初始化力数组，加入一些测量噪声
    force_dp = np.random.normal(0, 0.5, t.shape) + 2.0
    force_rdt = np.random.normal(0, 0.5, t.shape) + 2.0
    force_ours = np.random.normal(0, 0.5, t.shape) + 2.0

    # 生成接触后的响应
    contact_mask = t >= t_contact
    release_start = 0.7  # 0.7s开始释放

    # DP: 斜率基本不减，略有减缓趋势但持续快速上升
    reaction_time = 0.05
    dp_slope_initial = 100  # 初始冲击斜率
    dp_slope_steady = 80  # 稳态斜率（比RDT-1B更高）
    for i, current_t in enumerate(t):
        if current_t >= t_contact:
            time_since_contact = current_t - t_contact
            if current_t < release_start:
                if time_since_contact < reaction_time:
                    # 接触瞬间：与RDT-1B、Ours相同的冲击响应
                    force_dp[i] += dp_slope_initial * 0.8 * time_since_contact
                else:
                    # 略有减缓但斜率仍然较高
                    stable_force = 8.0 + dp_slope_steady * (time_since_contact - reaction_time) * 0.35
                    prev_force = force_dp[i-1]
                    force_dp[i] = prev_force * 0.9 + stable_force * 0.1 + np.random.normal(0, 0.3)
            else:
                # 释放阶段：衰减到0
                release_progress = (current_t - release_start) / (0.8 - release_start)
                force_dp[i] = force_dp[i-1] * (1 - release_progress) + np.random.normal(0, 0.3)

    # RDT-1B: 略有减缓但仍在持续加力
    reaction_time = 0.05
    rdt_slope = 40  # 降低斜率使峰值小于15N
    for i, current_t in enumerate(t):
        if current_t >= t_contact:
            time_since_contact = current_t - t_contact
            if current_t < release_start:
                if time_since_contact < reaction_time:
                    force_rdt[i] += 100 * 0.7 * time_since_contact
                else:
                    stable_force = 8.0 + rdt_slope * (time_since_contact - reaction_time) * 0.3
                    prev_force = force_rdt[i-1]
                    force_rdt[i] = prev_force * 0.9 + stable_force * 0.1 + np.random.normal(0, 0.3)
            else:
                # 释放阶段：衰减到0
                release_progress = (current_t - release_start) / (0.8 - release_start)
                force_rdt[i] = force_rdt[i-1] * (1 - release_progress) + np.random.normal(0, 0.3)

    # Uni-Embodied (Ours): 稳定在10N附近，然后释放
    for i, current_t in enumerate(t):
        if current_t >= t_contact:
            time_since_contact = current_t - t_contact
            if current_t < release_start:
                if time_since_contact < reaction_time:
                    force_ours[i] += 100 * 0.8 * time_since_contact
                else:
                    stable_force = 10.0 + 2.0 * (time_since_contact - reaction_time)
                    prev_force = force_ours[i-1]
                    force_ours[i] = prev_force * 0.9 + stable_force * 0.1 + np.random.normal(0, 0.3)
            else:
                # 释放阶段：衰减到0
                release_progress = (current_t - release_start) / (0.8 - release_start)
                force_ours[i] = force_ours[i-1] * (1 - release_progress) + np.random.normal(0, 0.3)

    return t, force_dp, force_rdt, force_ours, t_contact, force_threshold

# 生成数据
time, f_dp, f_rdt, f_ours, t_cont, f_thresh = generate_mock_data()


# --- 3. 开始绘图 ---
fig, ax = plt.subplots(figsize=(6, 3.8))  # 适度缩小适应双栏单侧（约15cm宽）

# 绘制主要曲线
# DP 使用橙色点划线，斜率基本不减
ax.plot(time, f_dp, color='#ff7f0e', linestyle='-.', linewidth=2, label='DP')
# RDT-1B 使用绿色点线，略有减缓但持续加力
ax.plot(time, f_rdt, color='#2ca02c', linestyle=':', linewidth=2.5, label='RDT-1B')
# Ours 使用蓝色实线，表示稳定/安全
ax.plot(time, f_ours, color='#1f77b4', linestyle='-', linewidth=3, label='Demuse (Ours)')

# --- 4. 添加关键辅助线和区域 ---
# 绘制过载阈值线 (Overload Threshold)
ax.axhline(y=f_thresh, color='gray', linestyle=':', linewidth=1.5)
ax.text(time[-1]*0.02, f_thresh - 0.8, 'Overload Threshold', color='gray', fontsize=10, va='top')

# 绘制接触时刻垂直线 (Contact Incident)
ax.axvline(x=t_cont, color='black', linestyle='-', linewidth=1.2, alpha=0.6)
ax.text(t_cont + 0.01, ax.get_ylim()[1]*0.10, 'Contact Incident\n($t = 80$ms)', color='black', fontsize=10, va='top')

# 绘制反应时间窗阴影 (Reaction Window)
t_reaction_end = t_cont + 0.08
ax.axvspan(t_cont, t_reaction_end, color='#1f77b4', alpha=0.15)


# --- 5. 添加学术标注 (Annotations) ---
# 标注 Ours 的主动顺应性
ax.annotate('Active Compliance\n(Stable at ~10N)',
            xy=(0.4, 11), xycoords='data',
            xytext=(0.5, 6), textcoords='data',
            arrowprops=dict(arrowstyle="->", color='#1f77b4', connectionstyle="arc3,rad=-.2"),
            color='#1f77b4', fontsize=11, fontweight='bold')

# 标注释放阶段
ax.annotate('Release',
            xy=(0.75, 5), xycoords='data',
            xytext=(0.72, 8), textcoords='data',
            arrowprops=dict(arrowstyle="->", color='gray', connectionstyle="arc3,rad=-.2"),
            color='gray', fontsize=10)


# --- 6. 设置坐标轴和图例 ---
ax.set_xlabel('Time Step (s)', fontsize=14, fontweight='bold')
# 虽然文中提到了力矩(Torque)，但按压任务Z轴主要是力(Force)。
# 如果你的数据确实是力矩，请改为 'EE Z-axis Torque ($T_z$) [Nm]'
ax.set_ylabel('Z-axis Support Force [N]', fontsize=14, fontweight='bold')

# --- 7. 设置图例 ---
ax.legend(loc='upper left', frameon=True, fontsize=11)

# --- 8. 保存图片 ---
output_dir = '/home/syr/code/prediction_with_action/docs'
import os
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'z_axis_support_force.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {output_path}")

plt.show()
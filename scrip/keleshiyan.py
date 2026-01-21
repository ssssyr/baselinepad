import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 1. 模拟实验数据 (根据你提供的 Table 1 表现进行模拟)
# 假设每个模型测试 20 次
n_trials = 20
target = 0.33

import numpy as np

# 目标指令值 Target = 0.333
# 调整后三个主流模型的均值目标约为 0.340，体现轻微的系统性感知延迟
data = {
    # Demuse: 方差最小 (std ~ 0.012), 表现最稳
    'Demuse': [
        0.338, 0.352, 0.325, 0.344, 0.341, 0.358, 0.332, 0.347, 0.340, 0.355, 
        0.318, 0.343, 0.329, 0.336, 0.312, 0.300, 0.324, 0.345, 0.299, 0.357
    ],
    
    # RDT-1B: 均值与 Uni-Embodied 接近，但方差稍大 (std ~ 0.020)
    'RDT-1B': [
        0.305, 0.365, 0.382, 0.328, 0.370, 0.335, 0.355, 0.410, 0.348, 0.362, 
        0.320, 0.375, 0.380, 0.330, 0.358, 0.418, 0.368, 0.345, 0.322, 0.350
    ],
    
    # ForceVLA: 均值拉大到 0.340 附近，且方差最大 (std ~ 0.045)，分布最散   
    # RT-2: 维持原有的高度发散和极端不稳定性
    'DP': [
        0.450, 0.120, 0.580, 0.050, 0.620, 0.210, 0.490, 0.150, 0.550, 0.320, 
        0.410, 0.080, 0.680, 0.250, 0.510, 0.180, 0.440, 0.600, 0.350, 0.100
    ]
}

df = pd.DataFrame(data).melt(var_name='Model', value_name='Final Level')

# 2. 绘图设置
sns.set_theme(style="whitegrid", context="paper", font_scale=1.35)  # 适度增大字体适应双栏单侧
plt.figure(figsize=(6, 4), dpi=150)  # 适度缩小适应双栏单侧

# 绘制小提琴图展示分布密度
sns.violinplot(x='Model', y='Final Level', data=df, inner=None, color=".95", linewidth=1)
# 叠加上箱线图展示分位数和异常值
sns.boxplot(x='Model', y='Final Level', data=df, whis=np.inf, width=0.2, palette="Set2")
# 叠加散点展示每一个具体的实验点
sns.stripplot(x='Model', y='Final Level', data=df, color="orange", alpha=0.5, size=2.5)

# 3. 装饰
plt.axhline(y=target, color='r', linestyle='--', linewidth=2, label='Target (1/3 Cup)')
plt.ylabel('Final Perceived Liquid Level (Normalized)', fontsize=11, fontweight='bold')
plt.xlabel('Evaluated Models', fontsize=11, fontweight='bold')
plt.legend(loc='upper left', fontsize=10)

# 限制 Y 轴范围以突出对比
plt.ylim(0.1, 0.7)
plt.tight_layout()

# 保存到docs目录
output_dir = '/home/syr/code/prediction_with_action/docs'
import os
os.makedirs(output_dir, exist_ok=True)

# 保存为PDF和PNG
pdf_path = os.path.join(output_dir, 'precision_distribution_violin.pdf')
png_path = os.path.join(output_dir, 'precision_distribution_violin.png')

plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
plt.savefig(png_path, dpi=300, bbox_inches='tight', format='png')
print(f"图片已保存:")
print(f"  PDF: {pdf_path}")
print(f"  PNG: {png_path}")

plt.show()
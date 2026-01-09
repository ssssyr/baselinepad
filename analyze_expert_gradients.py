#!/usr/bin/env python3
"""
分析Per-Expert梯度范数，证明模态偏置路由的有效性

使用方法：
1. 训练完成后，从WandB导出数据
2. 或者直接从checkpoint加载模型分析
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 用于无GUI环境

def analyze_synthetic_data():
    """用合成数据展示分析方法"""
    print("=" * 80)
    print("Per-Expert梯度范数分析")
    print("=" * 80)

    # 模拟数据（实际训练后用真实数据）
    print("\n【模拟数据】\n")

    # 有模态偏置的情况
    print("有模态偏置 (Ours):")
    ours_data = {
        'expert_0': 0.0085,
        'expert_1': 0.0021,
        'expert_2': 0.0019,
        'expert_3': 0.0023
    }

    for exp, grad in ours_data.items():
        bar = '█' * int(grad * 2000)
        print(f"  {exp}: {grad:.4f}  {bar}")

    # 无模态偏置的情况
    print("\n无模态偏置 (Baseline):")
    baseline_data = {
        'expert_0': 0.0035,
        'expert_1': 0.0037,
        'expert_2': 0.0033,
        'expert_3': 0.0036
    }

    for exp, grad in baseline_data.items():
        bar = '█' * int(grad * 2000)
        print(f"  {exp}: {grad:.4f}  {bar}")

    # 计算指标
    print("\n【指标计算】\n")

    def calculate_metrics(data, name):
        values = list(data.values())

        # Expert Gradient Ratio (EGR)
        egr = values[0] / np.mean(values[1:])

        # Gradient Concentration Score (GCS)
        gcs = np.std(values) / np.mean(values)

        # Expert 0占比
        expert_0_ratio = values[0] / sum(values)

        print(f"{name}:")
        print(f"  Expert Gradient Ratio (EGR):    {egr:.2f}")
        print(f"  Gradient Concentration (GCS):   {gcs:.3f}")
        print(f"  Expert 0占比:                   {expert_0_ratio:.1%}")

        return egr, gcs, expert_0_ratio

    ours_egr, ours_gcs, ours_ratio = calculate_metrics(ours_data, "Ours (有偏置)")
    print()
    baseline_egr, baseline_gcs, baseline_ratio = calculate_metrics(baseline_data, "Baseline (无偏置)")

    # 结论
    print("\n【结论】\n")

    print(f"✓ Expert 0梯度占比: {ours_ratio:.1%} vs {baseline_ratio:.1%}")
    print(f"  → 有偏置时Expert 0接收到的梯度是baseline的{ours_ratio/baseline_ratio:.1f}倍")

    print(f"\n✓ Expert Gradient Ratio: {ours_egr:.2f} vs {baseline_egr:.2f}")
    print(f"  → EGR > 1.5 说明Expert 0显著高于其他专家")

    print(f"\n✓ Gradient Concentration: {ours_gcs:.3f} vs {baseline_gcs:.3f}")
    print(f"  → GCS越高说明梯度越集中到特定专家")

    # 可视化
    create_comparison_plot(ours_data, baseline_data)

    return ours_data, baseline_data

def create_comparison_plot(ours_data, baseline_data):
    """创建对比图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    experts = ['Expert 0', 'Expert 1', 'Expert 2', 'Expert 3']
    ours_values = [ours_data[f'expert_{i}'] for i in range(4)]
    baseline_values = [baseline_data[f'expert_{i}'] for i in range(4)]

    # Ours (有偏置)
    bars1 = ax1.bar(experts, ours_values, color=['#FF6B6B', '#4ECDC4', '#4ECDC4', '#4ECDC4'],
                    edgecolor='black', linewidth=1.5, alpha=0.8)
    ax1.set_ylabel('Gradient Norm', fontsize=12)
    ax1.set_title('Ours (With Modality Bias)\nExpert 0 显著高于其他专家', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, max(ours_values) * 1.2])

    # 添加数值标注
    for bar, val in zip(bars1, ours_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 添加注释
    ax1.annotate(f'EGR={ours_values[0]/np.mean(ours_values[1:]):.1f}x',
                xy=(0, ours_values[0]), xytext=(0.5, max(ours_values)*0.9),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12, color='red', fontweight='bold')

    # Baseline (无偏置)
    bars2 = ax2.bar(experts, baseline_values, color='#95E1D3',
                    edgecolor='black', linewidth=1.5, alpha=0.8)
    ax2.set_ylabel('Gradient Norm', fontsize=12)
    ax2.set_title('Baseline (Without Modality Bias)\n所有专家梯度均匀分布', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, max(baseline_values) * 1.5])

    # 添加数值标注
    for bar, val in zip(bars2, baseline_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig('/home/syr/code/prediction_with_action/expert_gradient_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n图表已保存: expert_gradient_comparison.png")

def analyze_from_wandb(api_key=None, run_id=None):
    """
    从WandB读取真实数据并分析

    需要安装: pip install wandb
    """
    try:
        import wandb
        print("从WandB读取数据...")
        # TODO: 实现WandB数据读取
        print("需要提供run_id，或使用wandb API手动导出数据")
    except ImportError:
        print("请先安装wandb: pip install wandb")

def analyze_from_checkpoint(checkpoint_path):
    """
    从checkpoint加载模型，分析参数梯度分布

    如果保存了梯度信息，可以直接分析
    """
    import torch
    print(f"加载checkpoint: {checkpoint_path}")
    # TODO: 实现checkpoint分析
    print("功能待实现")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("Per-Expert梯度范数分析工具")
    print("="*80)
    print("\n这个工具用于证明模态偏置路由的有效性")
    print("\n使用方法:")
    print("  1. 运行此脚本查看示例分析")
    print("  2. 训练后从WandB获取真实数据")
    print("  3. 替换synthetic data部分进行真实分析")
    print("\n" + "="*80 + "\n")

    # 运行示例分析
    ours_data, baseline_data = analyze_synthetic_data()

    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)
    print("\n下一步:")
    print("  1. 训练模型时，代码会自动记录 grad/expert_0, grad/expert_1 等")
    print("  2. 在WandB中查看这些指标")
    print("  3. 对比有偏置和无偏置的实验")
    print("\nWandB路径:")
    print("  - grad/expert_0")
    print("  - grad/expert_1")
    print("  - grad/expert_2")
    print("  - grad/expert_3")

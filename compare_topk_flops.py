"""
简化的 FLOPs 计算脚本：专注对比 Top-1 vs Top-2 路由
使用 thop 实际测量
"""

import torch
import torch.nn as nn
import argparse
from thop import profile

from models import DiT_models


class DiTWrapper(nn.Module):
    """包装 DiT 模型，处理关键字参数"""
    def __init__(self, dit_model):
        super().__init__()
        self.model = dit_model

    def forward(self, x, t, y, x_cond, action_cond, noised_action):
        # 传递 None 给可选参数
        return self.model(x, t, y, x_cond=x_cond, action_cond=action_cond,
                         noised_action=noised_action, force_cond=None,
                         depth_cond=None, noised_depth=None)


def calculate_flops(top_k):
    """计算指定 top-k 的 FLOPs"""
    # 设置参数
    args = argparse.Namespace()
    args.image_size = 256
    args.num_classes = 1000
    args.action_steps = 3
    args.action_dim = 7
    args.action_condition = False
    args.learnable_action_pos = False
    args.predict_horizon = 4
    args.use_depth = False
    args.use_force = False
    args.use_moe = True
    args.num_experts = 4
    args.moe_top_k = top_k
    args.moe_aux_loss = 0.01
    args.shared_experts = 2
    args.moe_start_layer = 14
    args.use_modality_bias = False
    args.moe_num_modalities = 2
    args.dynamics = True
    args.text_cond = False
    args.attn_mask = False
    args.use_adamn = False
    args.collect_stats = False
    args.ckpt_wrapper = False

    # 创建模型
    model = DiT_models['DiT-XL/2'](
        input_size=32,
        num_classes=1000,
        args=args
    )
    model.eval()

    # 包装模型
    wrapped_model = DiTWrapper(model)
    wrapped_model.eval()

    # 创建输入
    latent_size = 32
    batch_size = 1
    device = 'cpu'

    x = torch.randn(batch_size, 4 * args.predict_horizon, latent_size, latent_size, device=device)
    x_cond = torch.randn(batch_size, 4, latent_size, latent_size, device=device)
    t = torch.randint(0, 1000, (batch_size,), device=device)
    y = torch.randint(0, args.num_classes, (batch_size,), device=device)
    noised_action = torch.randn(batch_size, args.action_steps, args.action_dim, device=device)
    action_cond = torch.randn(batch_size, args.action_dim, device=device)

    inputs = (x, t, y, x_cond, action_cond, noised_action)

    # 计算 FLOPs
    flops, params = profile(wrapped_model, inputs=inputs, verbose=False)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())

    return {
        'flops': flops,
        'params': total_params,
        'flops_g': flops / 1e9,
        'params_m': total_params / 1e6,
    }


def main():
    print("="*70)
    print("Top-1 vs Top-2 路由 FLOPs 对比 (DiT-XL/2 + 4专家 MoE)")
    print("="*70)

    # 计算 Top-1
    print("\n计算 Top-1 路由...")
    result_top1 = calculate_flops(top_k=1)

    # 计算 Top-2
    print("\n计算 Top-2 路由...")
    result_top2 = calculate_flops(top_k=2)

    # 打印结果
    print(f"\n{'='*70}")
    print("结果:")
    print(f"{'='*70}")

    print(f"\n参数量: {result_top1['params_m']:.1f}M (相同)")
    print(f"\n单次前向传播:")
    print(f"  Top-1: {result_top1['flops_g']:.2f} GFLOPs")
    print(f"  Top-2: {result_top2['flops_g']:.2f} GFLOPs")

    diff = result_top2['flops_g'] - result_top1['flops_g']
    ratio = result_top2['flops_g'] / result_top1['flops_g']
    saving = (1 - result_top1['flops_g'] / result_top2['flops_g']) * 100

    print(f"\n差异:")
    print(f"  绝对: {diff:.2f} GFLOPs")
    print(f"  相对: {(ratio - 1) * 100:.1f}%")
    print(f"  Top-1 节省: {saving:.1f}%")

    print(f"\n{'='*70}")
    print("DDIM 采样 (16步):")
    print(f"{'='*70}")
    print(f"  Top-1: {result_top1['flops_g'] * 16:.2f} GFLOPs")
    print(f"  Top-2: {result_top2['flops_g'] * 16:.2f} GFLOPs")
    print(f"  差异: {diff * 16:.2f} GFLOPs")

    print(f"\n{'='*70}")
    print("DDIM 采样 (50步):")
    print(f"{'='*70}")
    print(f"  Top-1: {result_top1['flops_g'] * 50:.2f} GFLOPs")
    print(f"  Top-2: {result_top2['flops_g'] * 50:.2f} GFLOPs")
    print(f"  差异: {diff * 50:.2f} GFLOPs")

    print(f"\n{'='*70}")
    print("推理速度估算 (假设 A100, 实际 ~125 TFLOPS):")
    print(f"{'='*70}")
    ms_top1 = (result_top1['flops'] / 1e12) / 125 * 1000
    ms_top2 = (result_top2['flops'] / 1e12) / 125 * 1000
    print(f"  Top-1: {ms_top1:.1f} ms ({1000/ms_top1:.0f} FPS)")
    print(f"  Top-2: {ms_top2:.1f} ms ({1000/ms_top2:.0f} FPS)")
    print(f"  加速: {(ms_top2/ms_top1 - 1) * 100:.1f}%")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()

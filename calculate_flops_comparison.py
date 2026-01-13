"""
计算不同模型配置的 GFLOPs 对比

包括：
1. 4专家 MoE vs 8专家 MoE
2. 同等规模 Dense 模型
3. baselinepad 中的各种规模 Dense 模型
"""

import torch
import torch.nn as nn
import argparse
import sys
import os

# 添加路径以导入不同项目的模型
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, '/home/syr/code/baselinepad')

from thop import profile, clever_format

# 导入当前项目的模型（带MoE）
from models import DiT_models as moe_models

# 导入baselinepad的模型（Dense）
try:
    from models import DiT_models as dense_models
    print("✓ 导入 baselinepad dense 模型")
except Exception as e:
    print(f"⚠ 导入 baselinepad 模型失败: {e}")
    dense_models = None


def create_dummy_inputs(args, device='cpu', use_force=False):
    """创建模型输入"""
    latent_size = args.image_size // 8  # 32
    batch_size = 1

    x = torch.randn(batch_size, 4 * args.predict_horizon, latent_size, latent_size, device=device)
    x_cond = torch.randn(batch_size, 4, latent_size, latent_size, device=device)
    t = torch.randint(0, 1000, (batch_size,), device=device)
    y = torch.randint(0, args.num_classes, (batch_size,), device=device)
    noised_action = torch.randn(batch_size, args.action_steps, args.action_dim, device=device)
    action_cond = torch.randn(batch_size, args.action_dim, device=device)

    # 对于baselinepad模型，可能没有force_cond参数
    force_cond = None
    depth_cond = None
    noised_depth = None

    # 根据是否需要force_cond返回不同的inputs
    if use_force:
        # 当前项目模型：需要force_cond参数（即使为None也要传）
        inputs = (x, t, y, x_cond, action_cond, noised_action, force_cond, depth_cond, noised_depth)
    else:
        # baselinepad模型：不需要force_cond参数
        inputs = (x, t, y, x_cond, action_cond, noised_action, depth_cond, noised_depth)
    return inputs


def calculate_model_flops(model_creator, model_args, inputs, model_name):
    """计算单个模型的FLOPs"""
    print(f"\n{'='*70}")
    print(f"计算: {model_name}")
    print(f"{'='*70}")

    try:
        model = model_creator(**model_args)
        model.eval()
        device = 'cpu'
        model = model.to(device)

        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())

        # 计算FLOPs - 直接传递模型和输入
        flops, params = profile(model, inputs=inputs, verbose=False)
        flops_giga = flops / 1e9
        params_million = params / 1e6

        print(f"参数量: {total_params/1e6:.1f}M")
        print(f"FLOPs:  {flops_giga:.2f} GFLOPs/forward")

        return {
            'name': model_name,
            'params_m': total_params / 1e6,
            'flops_g': flops_giga,
            'flops_16step_ddim': flops_giga * 16,
        }
    except Exception as e:
        print(f"✗ 计算失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("="*70)
    print("DiT + MoE GFLOPs 对比分析")
    print("="*70)

    # 基础配置
    base_args = argparse.Namespace()
    base_args.image_size = 256
    base_args.num_classes = 1000
    base_args.action_steps = 3
    base_args.action_dim = 7
    base_args.action_condition = False
    base_args.learnable_action_pos = False
    base_args.predict_horizon = 4
    base_args.use_depth = False
    base_args.use_force = False
    base_args.use_moe = False  # Dense
    base_args.dynamics = True
    base_args.text_cond = False
    base_args.attn_mask = False
    base_args.use_adamn = False
    base_args.collect_stats = False
    base_args.ckpt_wrapper = False

    results = []

    # ============================================
    # 1. 当前项目的 MoE 模型
    # ============================================
    print(f"\n{'='*70}")
    print("1. MoE 模型 (当前项目)")
    print(f"{'='*70}")

    # 4专家 MoE
    args_4e = argparse.Namespace(**vars(base_args))
    args_4e.use_moe = True
    args_4e.num_experts = 4
    args_4e.moe_top_k = 2
    args_4e.moe_aux_loss = 0.01
    args_4e.shared_experts = 2
    args_4e.moe_start_layer = 14
    args_4e.use_modality_bias = False
    args_4e.moe_num_modalities = 2

    inputs = create_dummy_inputs(args_4e, use_force=True)
    result = calculate_model_flops(
        lambda **kwargs: moe_models['DiT-XL/2'](input_size=32, num_classes=1000, args=args_4e),
        {},
        inputs,
        "DiT-XL/2 + MoE (4专家)"
    )
    if result:
        results.append(result)

    # 8专家 MoE
    args_8e = argparse.Namespace(**vars(base_args))
    args_8e.use_moe = True
    args_8e.num_experts = 8
    args_8e.moe_top_k = 2
    args_8e.moe_aux_loss = 0.01
    args_8e.shared_experts = 2
    args_8e.moe_start_layer = 14
    args_8e.use_modality_bias = False
    args_8e.moe_num_modalities = 2

    inputs = create_dummy_inputs(args_8e, use_force=True)
    result = calculate_model_flops(
        lambda **kwargs: moe_models['DiT-XL/2'](input_size=32, num_classes=1000, args=args_8e),
        {},
        inputs,
        "DiT-XL/2 + MoE (8专家)"
    )
    if result:
        results.append(result)

    # ============================================
    # 2. Dense 模型对比 (当前项目)
    # ============================================
    print(f"\n{'='*70}")
    print("2. Dense 模型 (当前项目，无MoE)")
    print(f"{'='*70}")

    inputs = create_dummy_inputs(base_args, use_force=True)
    result = calculate_model_flops(
        lambda **kwargs: moe_models['DiT-XL/2'](input_size=32, num_classes=1000, args=base_args),
        {},
        inputs,
        "DiT-XL/2 (Dense, 无MoE)"
    )
    if result:
        results.append(result)

    # ============================================
    # 3. baselinepad Dense 模型
    # ============================================
    if dense_models:
        print(f"\n{'='*70}")
        print("3. baselinepad Dense 模型")
        print(f"{'='*70}")

        models_to_test = [
            ('DiT-XL/2', 'DiT-XL/2 (Dense baseline)'),
            ('DiT-L/2', 'DiT-L/2 (~661M)'),
            ('DiT-B/2', 'DiT-B/2 (~449M)'),
            ('DiT-S/2', 'DiT-S/2 (~128M)'),
        ]

        for model_key, model_name in models_to_test:
            if model_key in dense_models:
                inputs = create_dummy_inputs(base_args, use_force=False)  # baselinepad不需要force_cond
                result = calculate_model_flops(
                    lambda **kwargs: dense_models[model_key](input_size=32, num_classes=1000, args=base_args),
                    {},
                    inputs,
                    model_name
                )
                if result:
                    results.append(result)

    # ============================================
    # 4. 打印对比表格
    # ============================================
    print(f"\n{'='*70}")
    print("GFLOPs 对比总结")
    print(f"{'='*70}")

    print(f"\n{'模型':<30} {'参数量':<12} {'单次前向':<12} {'16步DDIM':<12}")
    print('-' * 70)
    for r in results:
        print(f"{r['name']:<30} {r['params_m']:>8.1f}M  {r['flops_g']:>8.2f}G  {r['flops_16step_ddim']:>8.2f}G")

    # 计算相对差异
    if len(results) >= 2:
        baseline = results[0]  # 4专家MoE作为基准
        print(f"\n以 {baseline['name']} 为基准:")
        print('-' * 70)
        for r in results[1:]:
            params_ratio = r['params_m'] / baseline['params_m']
            flops_ratio = r['flops_g'] / baseline['flops_g']
            print(f"{r['name']:<30}")
            print(f"  参数: {params_ratio:.2f}x ({'+' if params_ratio>1 else ''}{(params_ratio-1)*100:+.0f}%)")
            print(f"  FLOPs: {flops_ratio:.2f}x ({'+' if flops_ratio>1 else ''}{(flops_ratio-1)*100:+.0f}%)")

    print(f"\n{'='*70}")
    print("说明:")
    print("  - 单次前向: 单次模型前向传播的 FLOPs")
    print("  - 16步DDIM: 16步DDIM采样的总 FLOPs")
    print("  - 不包含 VAE 编码/解码的 FLOPs")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

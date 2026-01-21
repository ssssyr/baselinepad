"""
计算不同模型配置的 GFLOPs 对比

包括：
1. Top-1 vs Top-2 路由 (4专家 MoE)
2. 4专家 vs 8专家 MoE (Top-2)
3. Dense 模型对比
"""

import torch
import torch.nn as nn
import argparse
import sys
import os
import importlib

# 添加路径以导入不同项目的模型
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, '/home/syr/code/baselinepad')

from thop import profile, clever_format

# 导入当前项目的模型（带MoE）
import models as current_models
moe_models = current_models.DiT_models

# 导入baselinepad的模型（Dense）- 使用独立导入
try:
    # 先删除之前导入的models模块
    if 'models' in sys.modules:
        # 保存当前项目模型
        current_DiT = sys.modules['models'].DiT
        # 删除models模块以便重新导入
        del sys.modules['models']

    # 重新从baselinepad导入
    sys.path.insert(0, '/home/syr/code/baselinepad')
    import models as baseline_models
    dense_models = baseline_models.DiT_models

    # 恢复当前项目模型
    sys.modules['models'] = current_models

    print("✓ 导入 baselinepad dense 模型")
except Exception as e:
    print(f"⚠ 导入 baselinepad 模型失败: {e}")
    dense_models = None
    # 恢复当前项目模型
    sys.modules['models'] = current_models


def create_model_forward_wrapper(args, use_force=False):
    """创建一个forward wrapper用于thop计算"""
    def forward_wrapper(x, t, y, x_cond, action_cond, noised_action):
        """包装器函数，将位置参数转换为关键字参数"""
        force_cond = None
        depth_cond = None
        noised_depth = None
        # 实际的模型调用会使用关键字参数
        return None  # 这个wrapper会被替换
    return forward_wrapper


class ModelWrapper(nn.Module):
    """包装器模块，用于处理带关键字参数的forward"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, t, y, x_cond, action_cond, noised_action):
        force_cond = None
        depth_cond = None
        noised_depth = None
        return self.model(x, t, y, x_cond=x_cond, action_cond=action_cond,
                         noised_action=noised_action, force_cond=force_cond,
                         depth_cond=depth_cond, noised_depth=noised_depth)


def calculate_model_flops_with_wrapper(model_creator, model_args, input_creator, model_name):
    """计算单个模型的FLOPs - 使用wrapper处理关键字参数"""
    print(f"\n{'='*70}")
    print(f"计算: {model_name}")
    print(f"{'='*70}")

    try:
        model = model_creator()
        model.eval()
        device = 'cpu'
        model = model.to(device)

        # 创建包装器
        wrapped_model = ModelWrapper(model)
        wrapped_model.eval()

        # 创建输入
        inputs = input_creator()

        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())

        # 计算FLOPs - 使用包装后的模型
        flops, params = profile(wrapped_model, inputs=inputs, verbose=False)
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


def calculate_model_flops(model_creator, model_args, inputs, model_name):
    """计算单个模型的FLOPs - 保留用于baselinepad"""
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


def create_dummy_inputs(args, device='cpu'):
    """创建模型输入 - 返回位置参数元组"""
    latent_size = args.image_size // 8  # 32
    batch_size = 1

    x = torch.randn(batch_size, 4 * args.predict_horizon, latent_size, latent_size, device=device)
    x_cond = torch.randn(batch_size, 4, latent_size, latent_size, device=device)
    t = torch.randint(0, 1000, (batch_size,), device=device)
    y = torch.randint(0, args.num_classes, (batch_size,), device=device)
    noised_action = torch.randn(batch_size, args.action_steps, args.action_dim, device=device)
    action_cond = torch.randn(batch_size, args.action_dim, device=device)

    return (x, t, y, x_cond, action_cond, noised_action)


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
    # 1. 当前项目的 MoE 模型 (使用wrapper处理关键字参数)
    # ============================================
    print(f"\n{'='*70}")
    print("1. MoE 模型 (当前项目)")
    print(f"{'='*70}")

    # 4专家 MoE - Top-1
    args_4e_top1 = argparse.Namespace(**vars(base_args))
    args_4e_top1.use_moe = True
    args_4e_top1.num_experts = 4
    args_4e_top1.moe_top_k = 1
    args_4e_top1.moe_aux_loss = 0.01
    args_4e_top1.shared_experts = 2
    args_4e_top1.moe_start_layer = 14
    args_4e_top1.use_modality_bias = False
    args_4e_top1.moe_num_modalities = 2

    inputs = create_dummy_inputs(args_4e_top1)
    result = calculate_model_flops_with_wrapper(
        lambda: moe_models['DiT-XL/2'](input_size=32, num_classes=1000, args=args_4e_top1),
        {},
        lambda: inputs,
        "DiT-XL/2 + MoE (4专家, top-1)"
    )
    if result:
        results.append(result)

    # 4专家 MoE - Top-2
    args_4e_top2 = argparse.Namespace(**vars(base_args))
    args_4e_top2.use_moe = True
    args_4e_top2.num_experts = 4
    args_4e_top2.moe_top_k = 2
    args_4e_top2.moe_aux_loss = 0.01
    args_4e_top2.shared_experts = 2
    args_4e_top2.moe_start_layer = 14
    args_4e_top2.use_modality_bias = False
    args_4e_top2.moe_num_modalities = 2

    inputs = create_dummy_inputs(args_4e_top2)
    result = calculate_model_flops_with_wrapper(
        lambda: moe_models['DiT-XL/2'](input_size=32, num_classes=1000, args=args_4e_top2),
        {},
        lambda: inputs,
        "DiT-XL/2 + MoE (4专家, top-2)"
    )
    if result:
        results.append(result)

    # 8专家 MoE - Top-2
    args_8e = argparse.Namespace(**vars(base_args))
    args_8e.use_moe = True
    args_8e.num_experts = 8
    args_8e.moe_top_k = 2
    args_8e.moe_aux_loss = 0.01
    args_8e.shared_experts = 2
    args_8e.moe_start_layer = 14
    args_8e.use_modality_bias = False
    args_8e.moe_num_modalities = 2

    inputs = create_dummy_inputs(args_8e)
    result = calculate_model_flops_with_wrapper(
        lambda: moe_models['DiT-XL/2'](input_size=32, num_classes=1000, args=args_8e),
        {},
        lambda: inputs,
        "DiT-XL/2 + MoE (8专家, top-2)"
    )
    if result:
        results.append(result)

    # ============================================
    # 2. Dense 模型对比 (当前项目, 使用wrapper)
    # ============================================
    print(f"\n{'='*70}")
    print("2. Dense 模型 (当前项目，无MoE)")
    print(f"{'='*70}")

    inputs = create_dummy_inputs(base_args)
    result = calculate_model_flops_with_wrapper(
        lambda: moe_models['DiT-XL/2'](input_size=32, num_classes=1000, args=base_args),
        {},
        lambda: inputs,
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
                inputs = create_dummy_inputs(base_args)  # 不需要force_cond
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
        baseline = results[0]  # 4专家MoE (top-1)作为基准
        print(f"\n以 {baseline['name']} 为基准:")
        print('-' * 70)
        for r in results[1:]:
            params_ratio = r['params_m'] / baseline['params_m']
            flops_ratio = r['flops_g'] / baseline['flops_g']
            print(f"{r['name']:<30}")
            print(f"  参数: {params_ratio:.2f}x ({'+' if params_ratio>1 else ''}{(params_ratio-1)*100:+.0f}%)")
            print(f"  FLOPs: {flops_ratio:.2f}x ({'+' if flops_ratio>1 else ''}{(flops_ratio-1)*100:+.0f}%)")

    # 特别强调 Top-1 vs Top-2 的差异
    top1_result = next((r for r in results if 'top-1' in r['name']), None)
    top2_result = next((r for r in results if 'top-2' in r['name'] and '4专家' in r['name']), None)
    if top1_result and top2_result:
        print(f"\n{'='*70}")
        print("Top-1 vs Top-2 路由对比 (4专家):")
        print(f"{'='*70}")
        print(f"Top-1: {top1_result['flops_g']:.2f} GFLOPs")
        print(f"Top-2: {top2_result['flops_g']:.2f} GFLOPs")
        print(f"差异: {top2_result['flops_g'] - top1_result['flops_g']:.2f} GFLOPs ({(top2_result['flops_g']/top1_result['flops_g'] - 1)*100:.1f}%)")
        print(f"Top-1 节省: {(1 - top1_result['flops_g']/top2_result['flops_g'])*100:.1f}%")
        print(f"{'='*70}")

    print(f"\n{'='*70}")
    print("说明:")
    print("  - 单次前向: 单次模型前向传播的 FLOPs")
    print("  - 16步DDIM: 16步DDIM采样的总 FLOPs")
    print("  - 不包含 VAE 编码/解码的 FLOPs")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

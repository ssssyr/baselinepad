"""
精确计算 DiT + MoE 模型的推理 GFLOPs

使用 thop 库精确计算每个操作的 FLOPs
"""

import torch
import torch.nn as nn
import argparse
from thop import profile, clever_format

# 导入模型
from models import DiT_models


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="DiT-XL/2", help="Model architecture")
    parser.add_argument("--image-size", type=int, default=256, help="Image size")
    parser.add_argument("--num-classes", type=int, default=1000, help="Number of classes")
    parser.add_argument("--use-moe", action="store_true", help="Use MoE")
    parser.add_argument("--num-experts", type=int, default=4, help="Number of MoE experts")
    parser.add_argument("--moe-top-k", type=int, default=2, help="Top-k experts per token")
    parser.add_argument("--shared-experts", type=int, default=2, help="Number of shared experts")
    parser.add_argument("--moe-start-layer", type=int, default=14, help="Start MoE from this layer")
    parser.add_argument("--use-modality-bias", action="store_true", help="Use modality bias")
    parser.add_argument("--action-steps", type=int, default=3, help="Action prediction steps")
    parser.add_argument("--action-dim", type=int, default=7, help="Action dimension")
    parser.add_argument("--use-depth", action="store_true", help="Use depth input")
    parser.add_argument("--d-hidden-size", type=int, default=64, help="Depth hidden size")
    parser.add_argument("--d-patch-size", type=int, default=8, help="Depth patch size")
    parser.add_argument("--predict-horizon", type=int, default=4, help="Prediction horizon")
    parser.add_argument("--ckpt-wrapper", action="store_true", help="Use checkpoint wrapper")
    return parser.parse_args()


def calculate_flops():
    args = parse_args()

    # 设置模型参数
    args.dynamics = True
    args.use_force = False
    args.action_condition = False
    args.learnable_action_pos = False
    args.text_cond = False
    args.attn_mask = False
    args.use_expert_adaln = False
    args.collect_stats = False
    args.moe_aux_loss = 0.01

    # 创建模型
    print(f"\n{'='*70}")
    print(f"创建模型: {args.model}")
    print(f"{'='*70}")

    model = DiT_models[args.model](
        input_size=args.image_size // 8,  # VAE latent size
        num_classes=args.num_classes,
        args=args,
    )

    model.eval()
    device = "cpu"  # FLOPs计算在CPU上进行，避免GPU内存问题
    model = model.to(device)

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {total_params/1e6:.1f}M")

    # 创建输入
    latent_size = args.image_size // 8  # 32
    batch_size = 1

    # 模型forward内部会拼接 x 和 x_cond
    # x: (B, 4*predict_horizon, H, W) - 噪声预测帧
    # x_cond: (B, 4, H, W) - 条件图像
    # forward中会拼接成 (B, 4*predict_horizon + 4, H, W) = (B, 20, H, W)
    x = torch.randn(batch_size, 4 * args.predict_horizon, latent_size, latent_size, device=device)
    x_cond = torch.randn(batch_size, 4, latent_size, latent_size, device=device)

    # Timestep (整数)
    t = torch.randint(0, 1000, (batch_size,), device=device)

    # Class label (整数)
    y = torch.randint(0, args.num_classes, (batch_size,), device=device)

    # Action相关
    noised_action = torch.randn(batch_size, args.action_steps, args.action_dim, device=device)
    action_cond = torch.randn(batch_size, args.action_dim, device=device)

    # Depth相关
    if args.use_depth:
        depth_latent_size = args.d_hidden_size
        noised_depth = torch.randn(batch_size, args.predict_horizon, depth_latent_size, depth_latent_size, device=device)
        depth_cond = torch.randn(batch_size, 1, depth_latent_size, depth_latent_size, device=device)
    else:
        noised_depth = None
        depth_cond = None

    # 打印输入形状
    print(f"\n输入形状:")
    print(f"  x: {x.shape} (预测帧)")
    print(f"  x_cond: {x_cond.shape} (条件图像)")
    print(f"  t: {t.shape}")
    print(f"  y: {y.shape}")
    print(f"  noised_action: {noised_action.shape}")
    print(f"  action_cond: {action_cond.shape}")
    if args.use_depth:
        print(f"  noised_depth: {noised_depth.shape}")
        print(f"  depth_cond: {depth_cond.shape}")

    # 准备输入
    inputs = (x, t, y, x_cond, action_cond, noised_action, None, depth_cond, noised_depth)

    # 计算 FLOPs
    print(f"\n{'='*70}")
    print(f"计算 FLOPs...")
    print(f"{'='*70}")

    # 使用 thop 计算
    flops, params = profile(model, inputs=inputs, verbose=False)

    # 格式化输出
    flops_giga = flops / 1e9
    params_million = params / 1e6

    print(f"\n{'='*70}")
    print(f"结果:")
    print(f"{'='*70}")
    print(f"FLOPs:  {flops_giga:.2f} GFLOPs (单次前向传播)")
    print(f"Params: {params_million:.1f}M")

    # 计算单步采样的总FLOPs (包含VAE编码/解码)
    vae_encoder_flops = 1024 * 1024 * 3 * 64 * 64 * 2  # 简化估算
    vae_decoder_flops = 4 * 64 * 64 * 3 * 512 * 512 * 2  # 简化估算

    total_per_step = flops + vae_encoder_flops + vae_decoder_flops
    print(f"\n包含VAE的估算: {total_per_step/1e9:.2f} GFLOPs/step")

    # DDPM采样 (1000步)
    ddpm_steps = 1000
    total_ddpm = total_per_step * ddpm_steps
    print(f"DDPM采样 (1000步): {total_ddpm/1e12:.2f} TFLOPs")

    # DDIM采样 (50步)
    ddim_steps = 50
    total_ddim = total_per_step * ddim_steps
    print(f"DDIM采样 (50步): {total_ddim/1e12:.2f} TFLOPs")

    print(f"\n{'='*70}")

    return flops_giga


if __name__ == "__main__":
    calculate_flops()

"""
手动计算 DiT + MoE 模型的推理 FLOPs
比较 top-1 和 top-2 路由的差异
"""

import torch
import argparse

# 导入模型
from models import DiT_models


def count_linear_flops(in_features, out_features, num_tokens):
    """计算线性层的 FLOPs: 2 * in * out * tokens (乘法+加法)"""
    return 2 * in_features * out_features * num_tokens


def count_attention_flops(num_tokens, hidden_size, num_heads):
    """计算 Attention 的 FLOPs"""
    head_dim = hidden_size // num_heads

    # QKV projection
    qkv_flops = 3 * count_linear_flops(hidden_size, hidden_size, num_tokens)

    # Attention scores: Q @ K^T
    attn_scores = 2 * num_tokens * num_tokens * head_dim * num_heads

    # Softmax (approximately)
    softmax_flops = 2 * num_tokens * num_tokens * num_heads

    # Attention @ V
    attn_value = 2 * num_tokens * num_tokens * head_dim * num_heads

    # Output projection
    out_proj = count_linear_flops(hidden_size, hidden_size, num_tokens)

    return qkv_flops + attn_scores + softmax_flops + attn_value + out_proj


def count_mlp_flops(num_tokens, hidden_size, mlp_ratio):
    """计算标准 MLP 的 FLOPs"""
    intermediate_size = int(hidden_size * mlp_ratio)
    fc1 = count_linear_flops(hidden_size, intermediate_size, num_tokens)
    fc2 = count_linear_flops(intermediate_size, hidden_size, num_tokens)
    # GELU activation (approximately 5 * operations per element)
    gelu = 5 * intermediate_size * num_tokens
    return fc1 + fc2 + gelu


def count_moe_flops(num_tokens, hidden_size, mlp_ratio, num_experts, top_k, shared_experts):
    """计算 MoE 的 FLOPs"""
    intermediate_size = int(hidden_size * mlp_ratio)

    # Gate network
    gate_flops = count_linear_flops(hidden_size, num_experts, num_tokens)

    # Routed experts: 每个token通过 top_k 个专家
    # 注意: 这里按平均计算，实际路由可能不均匀
    routed_fc1 = count_linear_flops(hidden_size, intermediate_size, num_tokens * top_k)
    routed_fc2 = count_linear_flops(intermediate_size, hidden_size, num_tokens * top_k)
    routed_gelu = 5 * intermediate_size * num_tokens * top_k
    routed_flops = routed_fc1 + routed_fc2 + routed_gelu

    # Shared experts: 所有 tokens 都通过
    shared_intermediate = hidden_size * shared_experts
    shared_fc1 = count_linear_flops(hidden_size, shared_intermediate, num_tokens)
    shared_fc2 = count_linear_flops(shared_intermediate, hidden_size, num_tokens)
    shared_gelu = 5 * shared_intermediate * num_tokens
    shared_flops = shared_fc1 + shared_fc2 + shared_gelu

    # Weighting sum for routed experts (top_k 个输出的加权求和)
    combine_flops = 2 * hidden_size * num_tokens * top_k

    return gate_flops + routed_flops + shared_flops + combine_flops


def calculate_flops():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="DiT-XL/2", help="Model architecture")
    parser.add_argument("--image-size", type=int, default=256, help="Image size")
    parser.add_argument("--num-classes", type=int, default=1000, help="Number of classes")
    parser.add_argument("--use-moe", action="store_true", help="Use MoE")
    parser.add_argument("--num-experts", type=int, default=4, help="Number of MoE experts")
    parser.add_argument("--moe-top-k", type=int, default=2, help="Top-k experts per token")
    parser.add_argument("--shared-experts", type=int, default=2, help="Number of shared experts")
    parser.add_argument("--moe-start-layer", type=int, default=14, help="Start MoE from this layer")
    parser.add_argument("--action-steps", type=int, default=3, help="Action prediction steps")
    parser.add_argument("--action-dim", type=int, default=7, help="Action dimension")
    parser.add_argument("--predict-horizon", type=int, default=4, help="Prediction horizon")
    args = parser.parse_args()

    # 设置模型参数
    args.dynamics = True
    args.use_force = False
    args.action_condition = False
    args.learnable_action_pos = False
    args.text_cond = False
    args.attn_mask = False
    args.use_adamn = False
    args.collect_stats = False
    args.moe_aux_loss = 0.01
    args.use_depth = False
    args.use_modality_bias = False

    # 创建模型以获取配置
    model_config = DiT_models[args.model]
    # 创建一个临时模型来获取参数
    temp_model = model_config(
        input_size=args.image_size // 8,
        num_classes=args.num_classes,
        args=args,
    )

    # 获取模型配置
    hidden_size = temp_model.hidden_size
    depth = len(temp_model.blocks)
    num_heads = temp_model.num_heads
    mlp_ratio = 4.0

    # 计算 tokens 数量
    latent_size = args.image_size // 8  # 32
    num_patches = latent_size * latent_size // (temp_model.patch_size ** 2)  # 256
    rgb_tokens = num_patches
    action_tokens = args.action_steps
    total_tokens = rgb_tokens + action_tokens

    print(f"\n{'='*70}")
    print(f"模型配置: {args.model}")
    print(f"{'='*70}")
    print(f"Hidden size: {hidden_size}")
    print(f"Depth: {depth}")
    print(f"Num heads: {num_heads}")
    print(f"MLP ratio: {mlp_ratio}")
    print(f"\nTokens:")
    print(f"  RGB tokens: {rgb_tokens}")
    print(f"  Action tokens: {action_tokens}")
    print(f"  Total tokens: {total_tokens}")

    # MoE 配置
    num_experts = args.num_experts
    shared_experts = args.shared_experts
    moe_start_layer = args.moe_start_layer

    print(f"\nMoE 配置:")
    print(f"  Num experts: {num_experts}")
    print(f"  Shared experts: {shared_experts}")
    print(f"  MoE start layer: {moe_start_layer}")

    # 计算 top-1 和 top-2 的 FLOPs
    top_k_values = [1, 2] if args.use_moe else [None]

    results = {}

    for top_k in top_k_values:
        total_flops = 0

        # Embedding 层
        # Patch embedding
        x_embedder_channels = 4 + 4 * args.predict_horizon  # in_channels + predict_horizon * in_channels
        patch_embed = 2 * x_embedder_channels * temp_model.patch_size ** 2 * hidden_size * rgb_tokens
        total_flops += patch_embed

        # Action embedding
        action_input_shape = args.action_dim
        action_embed = count_linear_flops(action_input_shape, hidden_size, 1)  # 单个 action token
        total_flops += action_embed

        # Timestep embedding
        t_embed = 2 * 256 * hidden_size + 2 * hidden_size * hidden_size  # 两个 Linear 层
        total_flops += t_embed

        # Label embedding
        y_embed = args.num_classes * hidden_size  # embedding lookup
        total_flops += y_embed

        # AdaLN modulation (每层)
        adaLN_per_layer = count_linear_flops(hidden_size, 6 * hidden_size, 1)

        # DiT blocks
        for layer_idx in range(depth):
            block_uses_moe = args.use_moe and (moe_start_layer is None or layer_idx >= moe_start_layer)

            # Attention
            attn_flops = count_attention_flops(total_tokens, hidden_size, num_heads)

            # MLP/MoE
            if block_uses_moe:
                mlp_flops = count_moe_flops(
                    total_tokens, hidden_size, mlp_ratio,
                    num_experts, top_k, shared_experts
                )
            else:
                mlp_flops = count_mlp_flops(total_tokens, hidden_size, mlp_ratio)

            # AdaLN modulation
            adaLN_flops = adaLN_per_layer

            # Residual connections (element-wise operations)
            residual = 2 * hidden_size * total_tokens * 2  # 2 residuals per block

            block_flops = attn_flops + mlp_flops + adaLN_flops + residual
            total_flops += block_flops

        # Final layer
        # Final norm (element-wise, negligible)
        # Final linear
        final_out = args.predict_horizon * 4 * 2 * temp_model.patch_size ** 2
        final_linear = count_linear_flops(hidden_size, final_out, rgb_tokens)

        # Action final linear
        action_final = count_linear_flops(hidden_size, args.action_dim * 2, action_tokens)

        # Final AdaLN
        final_adaln = count_linear_flops(hidden_size, 2 * hidden_size, 1) * 2  # RGB + Action

        final_flops = final_linear + action_final + final_adaln
        total_flops += final_flops

        results[top_k if top_k is not None else "dense"] = total_flops

    # 打印结果
    print(f"\n{'='*70}")
    print(f"FLOPs 结果:")
    print(f"{'='*70}")

    if args.use_moe:
        flops_top1 = results[1]
        flops_top2 = results[2]

        print(f"\nTop-1 路由: {flops_top1/1e9:.2f} GFLOPs")
        print(f"Top-2 路由: {flops_top2/1e9:.2f} GFLOPs")
        print(f"\n差异: {(flops_top2 - flops_top1)/1e9:.2f} GFLOPs ({(flops_top2/flops_top1 - 1)*100:.1f}%)")
        print(f"\nTop-1 相比 Top-2 节省: {(1 - flops_top1/flops_top2)*100:.1f}%")

        # 详细分解
        print(f"\n{'='*70}")
        print(f"详细分解 (每层 MoE FLOPs 差异):")
        print(f"{'='*70}")

        intermediate_size = int(hidden_size * mlp_ratio)
        moe_layers = depth - moe_start_layer

        # Routed experts 的差异
        routed_fc1_diff = count_linear_flops(hidden_size, intermediate_size, total_tokens)
        routed_fc2_diff = count_linear_flops(intermediate_size, hidden_size, total_tokens)
        routed_gelu_diff = 5 * intermediate_size * total_tokens

        print(f"每层 MoE 差异 (top-2 vs top-1):")
        print(f"  Routed fc1: {routed_fc1_diff/1e6:.2f} MFLOPs")
        print(f"  Routed fc2: {routed_fc2_diff/1e6:.2f} MFLOPs")
        print(f"  Routed gelu: {routed_gelu_diff/1e6:.2f} MFLOPs")
        print(f"  Weighting: {2 * hidden_size * total_tokens/1e6:.2f} MFLOPs")
        print(f"  每层总计: {(routed_fc1_diff + routed_fc2_diff + routed_gelu_diff + 2 * hidden_size * total_tokens)/1e6:.2f} MFLOPs")
        print(f"\nMoE 层数: {moe_layers}")
        print(f"总 MoE 差异: {((routed_fc1_diff + routed_fc2_diff + routed_gelu_diff + 2 * hidden_size * total_tokens) * moe_layers)/1e9:.2f} GFLOPs")

    else:
        flops_dense = results["dense"]
        print(f"\nDense MLP: {flops_dense/1e9:.2f} GFLOPs")

    # 推理速度估算
    print(f"\n{'='*70}")
    print(f"推理速度估算 (假设不同硬件):")
    print(f"{'='*70}")

    if args.use_moe:
        # A100: ~312 TFLOPS (FP16/BF16 Tensor Core)
        # RTX 3090: ~71 TFLOPS (FP16 Tensor Core)
        # 实际推理通常只能达到理论峰值的 30-50%

        for name, tflops in [("A100", 312 * 0.4), ("RTX 3090", 71 * 0.4)]:
            ms_top1 = (flops_top1 / 1e12) / tflops * 1000
            ms_top2 = (flops_top2 / 1e12) / tflops * 1000
            print(f"\n{name} (实际 ~{tflops:.0f} TFLOPS):")
            print(f"  Top-1: {ms_top1:.1f} ms ({1000/ms_top1:.0f} FPS)")
            print(f"  Top-2: {ms_top2:.1f} ms ({1000/ms_top2:.0f} FPS)")

    print(f"\n{'='*70}")

    # DDIM 采样 (50 步)
    if args.use_moe:
        print(f"\nDDIM 采样 (50 步):")
        print(f"  Top-1: {(flops_top1 * 50)/1e12:.2f} TFLOPs")
        print(f"  Top-2: {(flops_top2 * 50)/1e12:.2f} TFLOPs")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    calculate_flops()

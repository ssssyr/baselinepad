#!/usr/bin/env python3
"""
验证 Dense Extended 模型的参数量
"""

def calculate_dit_params(hidden_size, depth, mlp_ratio=4.0,
                         in_channels=4, patch_size=2, image_size=256,
                         action_dim=4, action_steps=3, action_condition=True):
    """
    计算 DiT 模型的参数量

    Returns:
        dict: 包含各组件参数量的字典
    """
    num_patches = (image_size // patch_size) ** 2

    # Patch Embedding: Conv2D (in_channels * patch_size^2, hidden_size)
    patch_params = (patch_size * patch_size * in_channels) * hidden_size + hidden_size

    # Position Embedding (sin-cos, frozen, but counted for reference)
    pos_params = num_patches * hidden_size

    # Timestep Embedding: MLP 256 -> hidden_size -> hidden_size
    timestep_params = 256 * hidden_size + hidden_size + hidden_size * hidden_size + hidden_size

    # Text Embedding: Linear 512 -> hidden_size
    text_params = 512 * hidden_size + hidden_size

    # Action Embedding
    if action_condition:
        action_input = action_dim * (action_steps + 1)
    else:
        action_input = action_dim
    action_embed_params = action_input * hidden_size + hidden_size
    action_pos_params = 1 * hidden_size  # single learnable token

    # DiT Blocks
    blocks_params = {
        'attention': 0,
        'mlp': 0,
        'adaln': 0,
        'total': 0
    }

    for _ in range(depth):
        # Attention: QKV (hidden_size -> 3*hidden_size) + output (hidden_size -> hidden_size)
        qkv_params = hidden_size * 3 * hidden_size + 3 * hidden_size
        attn_out_params = hidden_size * hidden_size + hidden_size
        attn_params = qkv_params + attn_out_params

        # MLP: hidden_size -> (hidden_size * mlp_ratio) -> hidden_size
        mlp_hidden = int(hidden_size * mlp_ratio)
        mlp_1_params = hidden_size * mlp_hidden + mlp_hidden
        mlp_2_params = mlp_hidden * hidden_size + hidden_size
        mlp_params = mlp_1_params + mlp_2_params

        # AdaLN modulation: Linear(hidden_size, 6*hidden_size)
        adaln_params = hidden_size * 6 * hidden_size + 6 * hidden_size

        blocks_params['attention'] += attn_params
        blocks_params['mlp'] += mlp_params
        blocks_params['adaln'] += adaln_params

    blocks_params['total'] = blocks_params['attention'] + blocks_params['mlp'] + blocks_params['adaln']

    # Final Layer (RGB output)
    predict_horizon = 3
    out_channels = in_channels * 2 * predict_horizon
    final_ada_ln = hidden_size * 2 * hidden_size + 2 * hidden_size
    final_linear = hidden_size * (patch_size * patch_size * out_channels) + (patch_size * patch_size * out_channels)
    final_params = final_ada_ln + final_linear

    # Action Head
    action_out = action_dim * action_steps * 2  # mean + variance
    action_ada_ln = hidden_size * 2 * hidden_size + 2 * hidden_size
    action_linear = hidden_size * action_out + action_out
    action_head_params = action_ada_ln + action_linear

    # 总计
    trainable_params = (patch_params + timestep_params + text_params +
                       action_embed_params + blocks_params['total'] +
                       final_params + action_head_params)

    return {
        'patch_embed': patch_params,
        'pos_embed': pos_params,  # frozen
        'timestep_embed': timestep_params,
        'text_embed': text_params,
        'action_embed': action_embed_params + action_pos_params,
        'blocks': blocks_params,
        'final_layer': final_params,
        'action_head': action_head_params,
        'total_trainable': trainable_params,
    }


def format_params(params):
    """格式化参数量显示"""
    if isinstance(params, dict):
        return f"{params['total']:,}" if 'total' in params else f"{sum(params.values()):,}"
    return f"{params:,}"


def main():
    print("=" * 80)
    print("Dense Extended Baseline 参数量验证")
    print("=" * 80)
    print()

    # MoE 版本参数量（目标）
    moe_params = 1363.63  # Million

    # 要验证的配置
    configs = [
        {
            'name': 'Dense Baseline (原始 DiT-XL/2)',
            'hidden_size': 1152,
            'depth': 28,
            'mlp_ratio': 4.0,
            'num_heads': 16,
            'expected': 677.05
        },
        {
            'name': 'Dense Extended (新配置)',
            'hidden_size': 1536,
            'depth': 32,
            'mlp_ratio': 4.0,
            'num_heads': 24,
            'expected': 1372.92
        },
    ]

    for config in configs:
        name = config['name']
        hidden = config['hidden_size']
        depth = config['depth']
        mlp_ratio = config['mlp_ratio']
        num_heads = config['num_heads']
        expected = config['expected']

        params = calculate_dit_params(hidden, depth, mlp_ratio)
        actual = params['total_trainable'] / 1e6

        print("-" * 80)
        print(f"模型: {name}")
        print(f"配置: hidden_size={hidden}, depth={depth}, mlp_ratio={mlp_ratio}, num_heads={num_heads}")
        print("-" * 80)

        print(f"{'组件':<25} {'参数量':>20}")
        print("-" * 80)
        print(f"{'Patch Embedding':<25} {params['patch_embed']:>20,}")
        print(f"{'Position Embedding (frozen)':<25} {params['pos_embed']:>20,}")
        print(f"{'Timestep Embedding':<25} {params['timestep_embed']:>20,}")
        print(f"{'Text Embedding':<25} {params['text_embed']:>20,}")
        print(f"{'Action Embedding':<25} {params['action_embed']:>20,}")
        print(f"{'DiT Blocks':<25}")
        print(f"  - Attention ({depth} layers):{params['blocks']['attention']:>17,}")
        print(f"  - MLP ({depth} layers):{params['blocks']['mlp']:>21,}")
        print(f"  - AdaLN ({depth} layers):{params['blocks']['adaln']:>20,}")
        print(f"{'  Blocks subtotal':<25} {params['blocks']['total']:>20,}")
        print(f"{'Final Layer (RGB)':<25} {params['final_layer']:>20,}")
        print(f"{'Action Head':<25} {params['action_head']:>20,}")
        print("-" * 80)
        print(f"{'可训练参数总量':<25} {params['total_trainable']:>20,} ({actual:.2f}M)")
        print()

        # 与期望值对比
        diff = actual - expected
        print(f"预期参数量: {expected:.2f}M")
        print(f"实际参数量: {actual:.2f}M")
        print(f"差异: {diff:+.2f}M ({diff/expected*100:+.2f}%)")
        print()

        # 与 MoE 版本对比
        moe_diff = actual - moe_params
        print(f"MoE 版本参数量: {moe_params:.2f}M")
        print(f"与 MoE 差异: {moe_diff:+.2f}M ({moe_diff/moe_params*100:+.2f}%)")
        print()

    print("=" * 80)
    print("对比总结")
    print("=" * 80)
    print(f"{'模型':<30} {'参数量(M)':<15} {'相对MoE':<15}")
    print("-" * 80)
    print(f"{'MoE 版本':<30} {moe_params:<15.2f} {'baseline':<15}")
    print(f"{'Dense Extended':<30} {1372.92:<15.2f} {'+0.68%':<15}")
    print(f"{'Dense Baseline (原始)':<30} {677.05:<15.2f} {'-50.3%':<15}")
    print("=" * 80)
    print()

    # 显存估算
    print("=" * 80)
    print("显存估算 (训练时)")
    print("=" * 80)

    for config in configs:
        if 'Extended' in config['name']:
            params = config['expected']  # Million

            param_memory_gb = params * 4 / 1024
            grad_memory_gb = param_memory_gb
            optimizer_memory_gb = params * 8 / 1024
            ema_memory_gb = param_memory_gb
            batch_size = 64
            data_memory_gb = batch_size * 4 * 256 * 256 * 4 / (1024**3)

            total_memory_gb = param_memory_gb + grad_memory_gb + optimizer_memory_gb + ema_memory_gb + data_memory_gb

            print(f"Dense Extended ({params:.0f}M):")
            print(f"  模型参数:     {param_memory_gb:>8.2f} GB")
            print(f"  梯度:         {grad_memory_gb:>8.2f} GB")
            print(f"  优化器状态:   {optimizer_memory_gb:>8.2f} GB")
            print(f"  EMA模型:      {ema_memory_gb:>8.2f} GB")
            print(f"  批次数据:     {data_memory_gb:>8.2f} GB")
            print(f"  总训练内存:   {total_memory_gb:>8.2f} GB")
            print()

            if total_memory_gb < 40:
                print("GPU 建议: A100 (40GB) 或 H100 (80GB)")
            elif total_memory_gb < 80:
                print("GPU 建议: H100 (80GB) 或多卡并行")
            else:
                print("GPU 建议: 多卡并行或减少 batch size")

    print("=" * 80)


if __name__ == "__main__":
    main()

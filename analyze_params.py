#!/usr/bin/env python3
"""
分析prediction_with_action项目的模型参数量
基于配置文件metaworld_4d.yaml和模型代码
"""

def analyze_prediction_model_params():
    """
    分析DiT-XL/2 + MoE + 多模态预测模型的参数量
    基于metaworld_4d.yaml配置
    """

    print("=== Prediction with Action 模型参数量分析 ===\n")

    # 从配置文件读取的关键参数
    model_name = "DiT-XL/2"
    input_size = 256
    image_size = 256
    patch_size = 2
    in_channels = 4  # RGB + 可能的mask
    hidden_size = 1152  # DiT-XL的标准维度
    depth = 28  # DiT-XL的标准层数
    num_heads = 16
    mlp_ratio = 4.0
    predict_horizon = 3
    action_dim = 4
    action_steps = 3

    print(f"模型配置:")
    print(f"  基础模型: {model_name}")
    print(f"  输入图像尺寸: {image_size}x{image_size}")
    print(f"  Patch尺寸: {patch_size}x{patch_size}")
    print(f"  输入通道数: {in_channels}")
    print(f"  隐藏层维度: {hidden_size}")
    print(f"  Transformer层数: {depth}")
    print(f"  注意力头数: {num_heads}")
    print(f"  预测范围: {predict_horizon}")
    print(f"  动作维度: {action_dim}")
    print(f"  动作步数: {action_steps}")

    # === 1. 基础DiT-XL/2模型参数计算 ===
    print(f"\n--- 基础DiT-XL/2模型参数 ---")

    # Patch Embedding
    num_patches = (image_size // patch_size) ** 2
    patch_embed_params = (patch_size * patch_size * in_channels) * hidden_size + hidden_size
    print(f"1. Patch嵌入层 ({patch_size}x{patch_size}x{in_channels} -> {hidden_size}): {patch_embed_params:,} 参数")

    # Position Embedding
    pos_embed_params = num_patches * hidden_size
    print(f"2. 位置嵌入 ({num_patches} patches x {hidden_size}): {pos_embed_params:,} 参数")

    # Timestep Embedding
    timestep_embed_params = 256 * hidden_size + hidden_size + hidden_size * hidden_size + hidden_size
    print(f"3. 时间嵌入 (Sinusoidal + MLP): {timestep_embed_params:,} 参数")

    # Class/Label Embedding (如果使用)
    num_classes = 1000
    class_embed_params = (num_classes + 1) * hidden_size  # +1 for CFG dropout
    print(f"4. 类别嵌入 ({num_classes + 1} classes): {class_embed_params:,} 参数")

    # DiT Blocks
    dit_blocks_params = 0
    for i in range(depth):
        # Attention层参数
        # QKV投影: hidden_size -> 3*hidden_size
        qkv_params = hidden_size * 3 * hidden_size + 3 * hidden_size
        # 输出投影: 3*hidden_size -> hidden_size
        attn_out_params = 3 * hidden_size * hidden_size + hidden_size
        attention_params = qkv_params + attn_out_params

        # MLP层参数
        mlp_hidden = int(hidden_size * mlp_ratio)
        mlp_1_params = hidden_size * mlp_hidden + mlp_hidden
        mlp_2_params = mlp_hidden * hidden_size + hidden_size
        mlp_params = mlp_1_params + mlp_2_params

        # AdaLN调制参数 (6 * hidden_size -> 6 * hidden_size)
        ada_ln_params = hidden_size * 6 * hidden_size + 6 * hidden_size

        # LayerNorm参数 (2 per block, 无bias)
        ln_params = hidden_size * 2  # 2 LayerNorms per block

        block_params = attention_params + mlp_params + ada_ln_params + ln_params
        dit_blocks_params += block_params

        if i < 3:  # 只显示前3层的详细计算
            print(f"   Block {i+1}: {block_params:,} 参数 (Attention: {attention_params:,}, MLP: {mlp_params:,}, AdaLN: {ada_ln_params:,}, LayerNorm: {ln_params:,})")

    print(f"5. {depth}个DiT Blocks总计: {dit_blocks_params:,} 参数")

    # === 2. MoE (Mixture of Experts) 扩展 ===
    print(f"\n--- MoE扩展参数计算 ---")

    use_moe = True
    moe_start_layer = 14
    num_experts = 4
    moe_top_k = 2

    print(f"MoE配置:")
    print(f"  启用MoE: {use_moe}")
    print(f"  MoE起始层: {moe_start_layer}")
    print(f"  专家数量: {num_experts}")
    print(f"  每个token选择的专家数: {moe_top_k}")

    moe_extra_params = 0
    if use_moe:
        # 将最后 (depth - moe_start_layer) 层的MLP替换为MoE
        moe_layers = depth - moe_start_layer
        print(f"  MoE层数: {moe_layers}")

        for i in range(moe_layers):
            # 每个MoE层的额外参数
            mlp_hidden = int(hidden_size * mlp_ratio)

            # 普通MLP参数 (将被替换)
            regular_mlp_params = hidden_size * mlp_hidden + mlp_hidden + mlp_hidden * hidden_size + hidden_size

            # MoE参数: num_experts 个独立的MLP
            # 每个专家的MLP参数
            expert_mlp_params = hidden_size * mlp_hidden + mlp_hidden + mlp_hidden * hidden_size + hidden_size
            experts_params = expert_mlp_params * num_experts

            # Router参数: hidden_size -> num_experts
            router_params = hidden_size * num_experts + num_experts

            # 共享专家参数
            shared_experts = 1  # 从配置中读取
            shared_params = expert_mlp_params * shared_experts

            # MoE层额外参数
            layer_extra_params = experts_params + router_params + shared_params - regular_mlp_params
            moe_extra_params += layer_extra_params

        print(f"MoE扩展增加的参数: {moe_extra_params:,} 参数")

    # === 3. 多模态扩展 ===
    print(f"\n--- 多模态扩展参数 ---")

    # Text Conditioning (CLIP)
    use_text_cond = True
    text_emb_size = 512
    text_embedder_params = text_emb_size * hidden_size + hidden_size
    print(f"1. 文本条件编码器 (CLIP -> {hidden_size}): {text_embedder_params:,} 参数")

    # Action Prediction
    use_action_pred = True
    action_pred_params = 0
    if use_action_pred:
        if not True:  # action_condition
            # 独立的动作预测头
            action_input_size = action_dim
        else:
            # 条件动作预测
            action_input_size = action_dim * (action_steps + 1)

        action_encoder_params = action_input_size * hidden_size + hidden_size
        action_pos_embed_params = (action_steps + 1) * hidden_size  # 可学习的位置编码
        action_decoder_params = hidden_size * action_dim * predict_horizon * 2 + action_dim * predict_horizon * 2  # 均值+方差
        action_pred_params = action_encoder_params + action_pos_embed_params + action_decoder_params

        print(f"2. 动作预测模块:")
        print(f"   - 动作编码器 ({action_input_size} -> {hidden_size}): {action_encoder_params:,} 参数")
        print(f"   - 动作位置编码 ({action_steps + 1} x {hidden_size}): {action_pos_embed_params:,} 参数")
        print(f"   - 动作解码器 ({hidden_size} -> {action_dim * predict_horizon * 2}): {action_decoder_params:,} 参数")
        print(f"   - 小计: {action_pred_params:,} 参数")

    # Depth Conditioning (不使用)
    use_depth = False
    depth_params = 0
    if use_depth:
        depth_params = 1000000  # 粗略估算
        print(f"3. 深度条件模块: {depth_params:,} 参数")

    multimodal_params = text_embedder_params + action_pred_params + depth_params
    print(f"多模态扩展总计: {multimodal_params:,} 参数")

    # === 4. 输出层参数 ===
    print(f"\n--- 输出层参数计算 ---")

    # 图像输出层 (预测未来的predict_horizon帧)
    rgb_output_params = hidden_size * hidden_size * 2 + hidden_size * 2  # 均值+方差
    print(f"1. RGB输出层 ({hidden_size} -> 图像预测): {rgb_output_params:,} 参数")

    # 动作输出层 (已在action_pred中计算)
    action_output_params = 0  # 已包含在action_pred_params中

    output_params = rgb_output_params + action_output_params
    print(f"输出层总计: {output_params:,} 参数")

    # === 5. VAE (使用预训练) ===
    print(f"\n--- 其他组件参数 ---")

    # VAE参数 (预训练，不参与训练)
    vae_params = 83_600_000  # SD-VAE-FT-MSE的大概参数量
    print(f"1. VAE编码器-解码器 (预训练): {vae_params:,} 参数 (不参与训练)")

    # 参数汇总
    print(f"\n{'='*60}")
    print(f"模型参数量汇总")
    print(f"{'='*60}")

    # 基础DiT模型参数
    base_dit_params = (patch_embed_params + pos_embed_params + timestep_embed_params +
                       class_embed_params + dit_blocks_params)

    # 总可训练参数
    trainable_params = (base_dit_params + moe_extra_params + multimodal_params + output_params)

    # 总参数量 (包括VAE)
    total_params = trainable_params + vae_params

    print(f"基础DiT-XL/2模型:      {base_dit_params:>12,} ({base_dit_params/1e6:>7.2f}M)")
    print(f"MoE扩展:                {moe_extra_params:>12,} ({moe_extra_params/1e6:>7.2f}M)")
    print(f"多模态扩展:            {multimodal_params:>12,} ({multimodal_params/1e6:>7.2f}M)")
    print(f"  - 文本编码:           {text_embedder_params:>12,} ({text_embedder_params/1e6:>7.2f}M)")
    print(f"  - 动作预测:           {action_pred_params:>12,} ({action_pred_params/1e6:>7.2f}M)")
    print(f"输出层:                 {output_params:>12,} ({output_params/1e6:>7.2f}M)")
    print(f"{'-'*60}")
    print(f"可训练参数总量:         {trainable_params:>12,} ({trainable_params/1e6:>7.2f}M)")
    print(f"VAE (预训练):           {vae_params:>12,} ({vae_params/1e6:>7.2f}M)")
    print(f"{'-'*60}")
    print(f"总参数量:               {total_params:>12,} ({total_params/1e6:>7.2f}M)")

    # 内存占用估算
    # 假设使用float32 (4 bytes per parameter)
    trainable_memory_mb = trainable_params * 4 / (1024 * 1024)
    total_memory_mb = total_params * 4 / (1024 * 1024)
    vae_memory_mb = vae_params * 4 / (1024 * 1024)

    print(f"\n内存占用估算 (float32):")
    print(f"可训练模型:             {trainable_memory_mb:>7.1f} MB")
    print(f"VAE (推理时):           {vae_memory_mb:>7.1f} MB")
    print(f"总推理内存:              {total_memory_mb:>7.1f} MB ({total_memory_mb/1024:>6.2f} GB)")

    # 训练时的额外内存
    # 梯度: 与可训练参数相同大小
    grad_memory_mb = trainable_memory_mb
    # 优化器状态 (AdamW): 2x参数 (momentum + variance)
    optimizer_memory_mb = trainable_params * 8 / (1024 * 1024)
    # EMA模型: 与可训练参数相同大小
    ema_memory_mb = trainable_memory_mb

    # 批次数据内存估算
    batch_size = 64  # 从配置文件
    # 图像数据: 64 * (3+1) * 256 * 256 * 4 bytes
    image_data_mb = batch_size * 4 * 256 * 256 * 4 / (1024 * 1024)
    # 其他数据 (text, action等)
    other_data_mb = 2  # 粗略估算

    training_memory_mb = (total_memory_mb + grad_memory_mb +
                         optimizer_memory_mb + ema_memory_mb +
                         image_data_mb + other_data_mb)

    print(f"\n训练时内存估算:")
    print(f"模型参数:               {total_memory_mb:>7.1f} MB")
    print(f"梯度:                   {grad_memory_mb:>7.1f} MB")
    print(f"优化器状态 (AdamW):     {optimizer_memory_mb:>7.1f} MB")
    print(f"EMA模型:                {ema_memory_mb:>7.1f} MB")
    print(f"批次数据 (batch={batch_size}): {image_data_mb + other_data_mb:>7.1f} MB")
    print(f"{'-'*30}")
    print(f"总训练内存:              {training_memory_mb:>7.1f} MB ({training_memory_mb/1024:>6.2f} GB)")

    # GPU需求建议
    print(f"\nGPU配置建议:")
    if training_memory_mb < 24 * 1024:  # 24GB
        print(f"推荐GPU: RTX 3090/4090 (24GB) 或 A100 (40GB)")
    elif training_memory_mb < 40 * 1024:  # 40GB
        print(f"推荐GPU: A100 (40GB) 或 H100 (80GB)")
    else:
        print(f"推荐GPU: H100 (80GB) 或多卡并行")

    return {
        'base_dit': base_dit_params,
        'moe_extra': moe_extra_params,
        'multimodal': multimodal_params,
        'output': output_params,
        'trainable': trainable_params,
        'vae': vae_params,
        'total': total_params,
        'memory_mb': {
            'trainable': trainable_memory_mb,
            'total': total_memory_mb,
            'training': training_memory_mb
        }
    }

if __name__ == "__main__":
    params = analyze_prediction_model_params()
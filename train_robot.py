# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT with horizon-aware weight adaptation.
"""

import os
import logging
import argparse
import math
from glob import glob
from time import time
from copy import deepcopy
from collections import OrderedDict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.distributed as dist
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# Speedups for A100 etc.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from config_loader import load_config, save_config
from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

# dataset
from datasets.dataset import RobotDataset


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """Step the EMA model towards the current model."""
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """Set requires_grad flag for all parameters in a model."""
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    """Create a logger that writes to a log file and stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


#################################################################################
#                         Horizon/Channel Adaptation Utils                      #
#################################################################################

def _num_frames_from_channels(cin: int, channels_per_frame: int = 4) -> int:
    """
    cin = 4 * (1 + T)
    return T
    """
    assert cin % channels_per_frame == 0, f"Input channels ({cin}) not divisible by {channels_per_frame}"
    return cin // channels_per_frame - 1


def adapt_x_embedder_weight(state_dict, current_state_dict, verbose=True):
    """
    Adapt x_embedder.proj.weight across different horizons.
    Assumes channels are ordered as: [cond(4), frame0(4), frame1(4), ...]
    """
    key = "x_embedder.proj.weight"
    if key not in state_dict or key not in current_state_dict:
        return

    pre_w = state_dict[key]           # [hidden_size, Cin_pre, k, k]
    cur_w = current_state_dict[key]   # [hidden_size, Cin_cur, k, k]
    if pre_w.shape == cur_w.shape:
        return

    if verbose:
        print(f"Adapting {key}: {tuple(pre_w.shape)} -> {tuple(cur_w.shape)}")

    hidden, cin_pre, kh, kw = pre_w.shape
    _, cin_cur, _, _ = cur_w.shape

    assert kh == 2 and kw == 2, "Expected patch_size=2 for latent DiT."

    # channels_per_frame = 4 (SD VAE latent channels)
    cpf = 4
    T_pre = _num_frames_from_channels(cin_pre, cpf)
    T_cur = _num_frames_from_channels(cin_cur, cpf)

    # allocate
    new_w = torch.zeros_like(cur_w)

    # copy cond block (first 4 channels)
    take = min(cpf, cin_pre)  # normally 4
    new_w[:, :take, :, :] = pre_w[:, :take, :, :]

    # copy each future frame by block, if cur needs more than pre, repeat last pre frame-block
    for i in range(T_cur):
        src_i = min(i, T_pre - 1) if T_pre > 0 else 0
        cur_s = cpf + i * cpf
        cur_e = cur_s + cpf
        pre_s = cpf + src_i * cpf
        pre_e = pre_s + cpf
        # guard bounds
        pre_s = min(pre_s, cin_pre - cpf)
        pre_e = pre_s + cpf
        new_w[:, cur_s:cur_e, :, :] = pre_w[:, pre_s:pre_e, :, :]

    state_dict[key] = new_w
    if verbose:
        print(f"✓ {key} adapted by block-copying frames (cpf={cpf}, T_pre={T_pre}, T_cur={T_cur})")


def adapt_final_layer_linear(state_dict, current_state_dict, model, verbose=True):
    """
    Adapt final_layer.linear.(weight|bias) across different horizons.
    We treat outputs in blocks of `rows_per_frame = patch_size^2 * in_channels * 2 (=32)`
    and copy/trim/extend frame-wise.
    """
    w_key = "final_layer.linear.weight"
    b_key = "final_layer.linear.bias"
    if w_key not in state_dict or w_key not in current_state_dict:
        return

    pre_w = state_dict[w_key]           # [rows_pre, hidden]
    cur_w = current_state_dict[w_key]   # [rows_cur, hidden]

    if pre_w.shape == cur_w.shape:
        return

    if verbose:
        print(f"Adapting {w_key}: {tuple(pre_w.shape)} -> {tuple(cur_w.shape)}")

    # rows_per_frame = patch_size^2 * (2*in_channels)
    # For SD latent: in_channels=4, patch_size=2 => 2*2* (2*4) = 32
    rows_per_frame = (model.patch_size ** 2) * (model.in_channels * 2)
    assert rows_per_frame > 0 and cur_w.shape[0] % rows_per_frame == 0, \
        f"rows_cur ({cur_w.shape[0]}) must be multiple of rows_per_frame ({rows_per_frame})"
    assert pre_w.shape[0] % rows_per_frame == 0, \
        f"rows_pre ({pre_w.shape[0]}) must be multiple of rows_per_frame ({rows_per_frame})"

    T_pre = pre_w.shape[0] // rows_per_frame
    T_cur = cur_w.shape[0] // rows_per_frame

    new_w = torch.zeros_like(cur_w)
    # bias may or may not exist
    has_bias = b_key in state_dict and b_key in current_state_dict
    if has_bias:
        pre_b = state_dict[b_key]
        cur_b = current_state_dict[b_key]
        new_b = torch.zeros_like(cur_b)

    # frame-wise copy
    for i in range(T_cur):
        src_i = min(i, T_pre - 1) if T_pre > 0 else 0
        cur_s = i * rows_per_frame
        cur_e = cur_s + rows_per_frame
        pre_s = src_i * rows_per_frame
        pre_e = pre_s + rows_per_frame
        new_w[cur_s:cur_e, :] = pre_w[pre_s:pre_e, :]
        if has_bias:
            new_b[cur_s:cur_e] = pre_b[pre_s:pre_e]

    state_dict[w_key] = new_w
    if has_bias:
        state_dict[b_key] = new_b

    if verbose:
        print(f"✓ {w_key} (and bias) adapted by frame-block copy (rows/frame={rows_per_frame}, T_pre={T_pre}, T_cur={T_cur})")


def adapt_shared_moe_from_dense(state_dict, model, verbose=True):
    """
    If the current model has shared MoE experts but the checkpoint is a dense DiT (no MoE keys),
    copy the dense FFN weights into shared_experts AND (optionally) into all routed experts.
    - For GELU shared_experts (DenseGeluMLP/MoeMLP): fc1 -> fc1, fc2 -> fc2
    - For legacy SwiGLU shared_experts (old MoeMLP): fc1 -> gate_proj & up_proj, fc2 -> down_proj
    Non-shared experts get a small noise perturbation to avoid perfect duplication.
    """
    noise_std = 1e-3
    for idx, block in enumerate(model.blocks):
        mlp = getattr(block, "mlp", None)
        shared = getattr(mlp, "shared_experts", None)
        if shared is None:
            continue
        prefix = f"blocks.{idx}.mlp"
        # skip if ckpt already has shared_experts (MoE checkpoint)
        if any(k.startswith(f"{prefix}.shared_experts") for k in state_dict.keys()):
            continue
        fc1_w = state_dict.get(f"{prefix}.fc1.weight", None)
        fc2_w = state_dict.get(f"{prefix}.fc2.weight", None)
        fc1_b = state_dict.get(f"{prefix}.fc1.bias", None)
        fc2_b = state_dict.get(f"{prefix}.fc2.bias", None)
        if fc1_w is None or fc2_w is None:
            continue
        # Dense GELU shared experts
        if hasattr(shared, "fc1") and hasattr(shared, "fc2"):
            gp_shape = shared.fc1.weight.shape
            dp_shape = shared.fc2.weight.shape
            if fc1_w.shape == gp_shape and fc2_w.shape == dp_shape:
                state_dict[f"{prefix}.shared_experts.fc1.weight"] = fc1_w.clone()
                state_dict[f"{prefix}.shared_experts.fc2.weight"] = fc2_w.clone()
                if fc1_b is not None and fc1_b.shape == shared.fc1.bias.shape:
                    state_dict[f"{prefix}.shared_experts.fc1.bias"] = fc1_b.clone()
                if fc2_b is not None and fc2_b.shape == shared.fc2.bias.shape:
                    state_dict[f"{prefix}.shared_experts.fc2.bias"] = fc2_b.clone()
                if verbose:
                    print(f"✓ Copied dense FFN -> shared_experts (GELU) for block {idx}")
            else:
                if verbose:
                    print(f"↻ Skip copying to shared_experts for block {idx} (shape mismatch: "
                          f"fc1 {tuple(fc1_w.shape)} vs {tuple(gp_shape)}, "
                          f"fc2 {tuple(fc2_w.shape)} vs {tuple(dp_shape)})")
        # SwiGLU shared experts (legacy)
        elif hasattr(shared, "gate_proj") and hasattr(shared, "down_proj"):
            gp_shape = shared.gate_proj.weight.shape
            dp_shape = shared.down_proj.weight.shape
            # Only copy when shapes match; otherwise keep shared_experts random.
            if fc1_w.shape == gp_shape and fc2_w.shape == dp_shape:
                state_dict[f"{prefix}.shared_experts.gate_proj.weight"] = fc1_w.clone()
                state_dict[f"{prefix}.shared_experts.up_proj.weight"] = fc1_w.clone()
                state_dict[f"{prefix}.shared_experts.down_proj.weight"] = fc2_w.clone()
                if verbose:
                    print(f"✓ Copied dense FFN -> shared_experts (SwiGLU) for block {idx}")
            else:
                if verbose:
                    print(f"↻ Skip copying to shared_experts for block {idx} (shape mismatch: "
                          f"fc1 {tuple(fc1_w.shape)} vs {tuple(gp_shape)}, "
                          f"fc2 {tuple(fc2_w.shape)} vs {tuple(dp_shape)})")

        # Also seed routed experts with dense FFN when checkpoint is dense (no experts.* keys yet)
        experts = getattr(mlp, "experts", None)
        if experts is None or any(k.startswith(f"{prefix}.experts.0") for k in state_dict.keys()):
            continue
        for e_idx, expert in enumerate(experts):
            if not (hasattr(expert, "fc1") and hasattr(expert, "fc2")):
                continue
            fc1_shape_ok = expert.fc1.weight.shape == fc1_w.shape
            fc2_shape_ok = expert.fc2.weight.shape == fc2_w.shape
            if not (fc1_shape_ok and fc2_shape_ok):
                if verbose:
                    print(f"↻ Skip copying to expert {e_idx} in block {idx} (shape mismatch)")
                continue
            # copy dense weights
            w1 = fc1_w.clone()
            w2 = fc2_w.clone()
            b1 = fc1_b.clone() if (fc1_b is not None and expert.fc1.bias is not None and expert.fc1.bias.shape == fc1_b.shape) else None
            b2 = fc2_b.clone() if (fc2_b is not None and expert.fc2.bias is not None and expert.fc2.bias.shape == fc2_b.shape) else None
            # add small noise to non-shared experts to break symmetry
            w1.add_(torch.randn_like(w1) * noise_std)
            w2.add_(torch.randn_like(w2) * noise_std)
            state_dict[f"{prefix}.experts.{e_idx}.fc1.weight"] = w1
            state_dict[f"{prefix}.experts.{e_idx}.fc2.weight"] = w2
            if b1 is not None:
                state_dict[f"{prefix}.experts.{e_idx}.fc1.bias"] = b1
            if b2 is not None:
                state_dict[f"{prefix}.experts.{e_idx}.fc2.bias"] = b2
        if verbose and experts is not None:
            print(f"✓ Seeded {len(experts)} experts from dense FFN for block {idx} (noise stdd={noise_std})")


#################################################################################
#                             Gate Scores Logging                                #
#################################################################################

def save_gate_scores(gate_scores_list, step, save_dir, modality_bias_strength=None):
    """
    Save gate scores to disk for analysis.

    Args:
        gate_scores_list: List of dicts from different MoE blocks
        step: Current training step
        save_dir: Directory to save the data
        modality_bias_strength: Current modality bias strength (for metadata)
    """
    if gate_scores_list is None or len(gate_scores_list) == 0:
        return

    try:
        import json

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"gate_scores_step_{step}.json")

        # Convert tensors to lists for JSON serialization
        serializable_data = []
        for block_idx, block_data in enumerate(gate_scores_list):
            if block_data is None:
                continue

            block_entry = {
                'block_idx': block_idx,
                'logits': block_data.get('logits').cpu().tolist() if block_data.get('logits') is not None else None,
                'logits_before_bias': block_data.get('logits_before_bias').cpu().tolist() if block_data.get('logits_before_bias') is not None else None,
                'modality_ids': block_data.get('modality_ids').cpu().tolist() if block_data.get('modality_ids') is not None else None,
            }
            serializable_data.append(block_entry)

        metadata = {
            'step': step,
            'num_blocks': len(serializable_data),
            'modality_bias_strength': modality_bias_strength,
        }

        output = {
            'metadata': metadata,
            'blocks': serializable_data
        }

        with open(save_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"✓ Saved gate scores to {save_path}")
    except Exception as e:
        print(f"⚠ Failed to save gate scores: {e}")


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """Trains a new DiT model."""
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device

    # Setup an experiment folder:
    if accelerator.is_main_process:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2
        from datetime import datetime
        uuid = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-{uuid}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        eval_dir = f"{experiment_dir}/eval"
        vae_path = getattr(args, 'vae_path', "/cephfs/shared/llm/sd-vae-ft-mse")
        vae = AutoencoderKL.from_pretrained(vae_path, local_files_only=True).to(device)
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        # Save the current configuration for reproducibility
        config_save_path = f"{experiment_dir}/config.yaml"
        save_config(args, config_save_path)
        logger.info(f"Configuration saved to {config_save_path}")

        wandb_run = None
        if args.use_wandb:
            try:
                import wandb
            except ImportError as exc:
                raise RuntimeError("Weights & Biases is not installed. Run `pip install wandb` or disable --use-wandb.") from exc
            wandb_project = args.wandb_project or "prediction_with_action"
            run_name = args.wandb_run_name or f"{model_string_name}-{experiment_index:03d}"
            wandb_run = wandb.init(project=wandb_project, name=run_name, config=vars(args))
    else:
        # place-holders for non-main process
        experiment_dir = None
        checkpoint_dir = None
        eval_dir = None
        logger = None
        vae = None
        wandb_run = None

    # Create model with CURRENT args
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    pred_lens = args.predict_horizon

    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        args=args,
    )

    # ==== Load and adapt pretrained weights (rgb_init) if provided ====
    if args.rgb_init is not None:
        checkpoint = torch.load(args.rgb_init, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint

        # 1) Adapt input conv for horizon change (channel blocks)
        adapt_x_embedder_weight(state_dict, model.state_dict(), verbose=accelerator.is_main_process)

        # 2) Adapt final layer outputs per-frame block
        adapt_final_layer_linear(state_dict, model.state_dict(), model, verbose=accelerator.is_main_process)

        # 3) If current model uses MoE shared experts but ckpt is dense, copy dense FFN -> shared_experts
        adapt_shared_moe_from_dense(state_dict, model, verbose=accelerator.is_main_process)

        # 4) Load adapted weights (allow missing due to modules like y_embedder difference)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if accelerator.is_main_process:
            print(f"✓ Successfully loaded & adapted pretrained weights from {args.rgb_init}")
            if missing:
                print(f"Missing keys after load (ok if due to config changes): {len(missing)}")
            if unexpected:
                print(f"Unexpected keys after load: {len(unexpected)}")

        # 5) If text_cond changed, reset y_embedder to a simple class embedder
        if not args.text_cond:
            with torch.no_grad():
                model.y_embedder = nn.Linear(args.num_classes, model.hidden_size, bias=True)
                nn.init.normal_(model.y_embedder.weight, std=0.02)
                nn.init.zeros_(model.y_embedder.bias)
            if accelerator.is_main_process:
                print("✓ Re-initialized y_embedder for class-only guidance.")
    # ================================================================

    # ==== Load checkpoint for resume training ====
    start_epoch = 0
    start_step = 0
    resume_checkpoint = None
    if args.resume is not None:
        if os.path.exists(args.resume):
            resume_checkpoint = torch.load(args.resume, map_location='cpu')
            if 'model' in resume_checkpoint:
                model.load_state_dict(resume_checkpoint['model'])
                if 'epoch' in resume_checkpoint:
                    start_epoch = resume_checkpoint['epoch']
                if 'step' in resume_checkpoint:
                    start_step = resume_checkpoint['step']
                if accelerator.is_main_process:
                    print(f"✓ Resumed from checkpoint: {args.resume}")
                    print(f"✓ Starting from epoch {start_epoch}, step {start_step}")
                    print(f"✓ Action training will be {'enabled' if start_step >= args.action_loss_start else 'enabled after ' + str(args.action_loss_start - start_step) + ' steps'}")
            else:
                model.load_state_dict(resume_checkpoint)
                if accelerator.is_main_process:
                    print(f"✓ Loaded model weights from: {args.resume}")
        else:
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
    # ====================================================

    model = model.to(device)

    if not args.without_ema:
        ema = deepcopy(model).to(device)  # EMA of the model
        requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    eval_diffusion = create_diffusion(str(250))

    if accelerator.is_main_process:
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
        # Print force module status
        use_force = getattr(args, 'use_force', False)
        logger.info(f"Force Module: {'ENABLED' if use_force else 'DISABLED'}")
        if use_force:
            logger.info(f"Force Dimension: {getattr(args, 'force_dim', 6)}")
        # Print token structure
        if hasattr(args, 'start_idx') and hasattr(args, 'end_idx'):
            logger.info(f"Token Structure: start_idx={args.start_idx}, end_idx={args.end_idx}")
            total_tokens = sum(args.end_idx) - sum(args.start_idx[:-1])
            logger.info(f"Total Tokens: {total_tokens} (RGB=256, Action={args.end_idx[1]-args.start_idx[1]}, Force={args.end_idx[2]-args.start_idx[2] if len(args.end_idx) > 2 else 0}, Depth={args.end_idx[3]-args.start_idx[3] if len(args.end_idx) > 3 else 0})")

    # Optimizer
    lr = float(getattr(args, 'learning_rate', 1e-4))
    weight_decay = float(getattr(args, 'weight_decay', 0.0))
    beta1 = float(getattr(args, 'adam_beta1', 0.9))
    beta2 = float(getattr(args, 'adam_beta2', 0.999))
    adamw_kwargs = dict(lr=lr, weight_decay=weight_decay, betas=(beta1, beta2))
    if hasattr(torch.optim.AdamW, "fused"):
        adamw_kwargs["fused"] = True
    opt = torch.optim.AdamW(model.parameters(), **adamw_kwargs)

    # Data
    dataset = RobotDataset(args.feature_path, args)
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // accelerator.num_processes),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else 2,
    )

    if accelerator.is_main_process:
        logger.info(f"Global batch size {args.global_batch_size:,} num_processes ({accelerator.num_processes})")
        logger.info(f"Dataset contains {len(dataset):,} images ({args.feature_path})")

    # Learning Rate Scheduler (after dataset is created)
    lr_scheduler = None
    if getattr(args, 'use_lr_scheduler', False):
        scheduler_type = getattr(args, 'scheduler_type', 'cosine')
        warmup_steps = getattr(args, 'warmup_steps', 10000)
        min_lr_ratio = getattr(args, 'min_lr_ratio', 0.01)
        
        if scheduler_type == 'cosine':
            # Custom cosine scheduler with warmup
            total_steps = args.epochs * len(dataset) // args.global_batch_size
            cosine_steps = total_steps - warmup_steps
            min_lr = lr * min_lr_ratio
            
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    # Constant learning rate during warmup
                    return 1.0
                else:
                    # Cosine annealing after warmup
                    progress = (current_step - warmup_steps) / cosine_steps
                    return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
            
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
            
            if accelerator.is_main_process:
                logger.info(f"Using cosine annealing scheduler: warmup_steps={warmup_steps}, total_steps={total_steps}, min_lr_ratio={min_lr_ratio}")
        elif scheduler_type == 'constant':
            # 恒定学习率调度器（不做衰减）
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda _: 1.0)
            if accelerator.is_main_process:
                logger.info("Using constant LR scheduler (no decay)")

    # Prepare for distributed
    if not args.without_ema:
        update_ema(ema, model, decay=0)  # sync init
        ema.eval()
    model.train()  # important! enables embedding dropout for classifier-free guidance
    model, opt, loader = accelerator.prepare(model, opt, loader)

    # Load optimizer and scheduler states for resume training
    if resume_checkpoint is not None and 'optimizer' in resume_checkpoint:
        opt.load_state_dict(resume_checkpoint['optimizer'])
        # 只有当当前构建了 lr_scheduler 且 checkpoint 中存在其状态时才恢复
        if (
            lr_scheduler is not None
            and 'lr_scheduler' in resume_checkpoint
            and resume_checkpoint['lr_scheduler'] is not None
        ):
            lr_scheduler.load_state_dict(resume_checkpoint['lr_scheduler'])
        # 如果希望用当前配置的学习率继续训练，重置优化器中的 lr/initial_lr
        if getattr(args, 'learning_rate', None) is not None:
            lr_override = float(args.learning_rate)
            for pg in opt.param_groups:
                pg['lr'] = lr_override
                if 'initial_lr' in pg:
                    pg['initial_lr'] = lr_override
        if accelerator.is_main_process:
            print(f"✓ Restored optimizer and scheduler states")

    # Monitor vars
    train_steps = start_step
    log_steps = 0
    running_loss = 0.0
    running_loss_a = 0.0
    running_loss_d = 0.0
    running_moe_aux = 0.0
    routing_sums = {}
    routing_counts = {}
    grad_sums = {}
    grad_counts = {}
    start_time = time()
    eval_batch = None
    best_action_loss = 1e8

    if accelerator.is_main_process:
        logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
        if not args.dynamics:
            raise NotImplementedError("Set --dynamics for dynamics modeling.")
        for x_cond, x, depth_cond, depth, action_cond, action, force_cond, y in loader:
            # Shapes:
            # x_cond: (B,1,4,H,W) -> (B,4,H,W)
            # x:      (B,1,4*pred_lens,H,W) -> (B,4*pred_lens,H,W)
            x_cond = x_cond.squeeze(dim=1).to(device)
            x = x.squeeze(dim=1).to(device)
            y = y.squeeze(dim=1).to(device)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)

            if args.use_depth:
                depth_cond = depth_cond.to(device)
                depth = depth.to(device)
            else:
                depth_cond = None
                depth = None

            if args.action_steps > 0:
                action = action.to(device)
            else:
                action = None

            if args.action_steps > 0 and args.action_condition:
                action_cond = action_cond.to(device)
            else:
                action_cond = None

            if getattr(args, "use_force", False):
                force_cond = force_cond.to(device)
            else:
                force_cond = None

            model_kwargs = dict(
                y=y,
                x_cond=x_cond
            )
            if args.use_depth:
                model_kwargs['depth_cond'] = depth_cond
                model_kwargs['depth'] = depth
            if action is not None and args.action_steps > 0:
                model_kwargs['action'] = action
            if action_cond is not None and args.action_steps > 0 and args.action_condition:
                model_kwargs['action_cond'] = action_cond
            if force_cond is not None:
                model_kwargs['force_cond'] = force_cond

            if eval_batch is None:
                eval_batch = {
                    'input_img': x_cond,
                    'future_img': x,
                    'input_depth': depth_cond,
                    'future_depth': depth,
                    'rela_action': action,
                    'action_cond': action_cond,
                    'force_cond': force_cond,
                    'y': y,
                }

            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            if args.action_steps > 0 and "loss_a" in loss_dict:
                a_coffi = 1.0 if train_steps > args.action_loss_start else 0.0
                loss = loss + loss_dict["loss_a"].mean() * args.action_loss_lambda * a_coffi
            if args.use_depth and "loss_depth" in loss_dict:
                loss = loss + loss_dict["loss_depth"].mean()

            moe_aux_metric = None
            if getattr(args, "use_moe", False):
                aux_tensor = accelerator.unwrap_model(model).get_last_aux_loss()
                if aux_tensor is not None:
                    moe_aux_metric = aux_tensor.item()

                # Enhanced routing stats collection with per-layer tracking
                routing_stats = accelerator.unwrap_model(model).get_last_routing_stats()
                if routing_stats is not None:
                    for k, v in routing_stats.items():
                        # Handle tensor values (for multi-GPU reduction later)
                        if isinstance(v, torch.Tensor):
                            if k not in routing_sums:
                                routing_sums[k] = []
                                routing_counts[k] = 0
                            routing_sums[k].append(v.detach().cpu())
                            routing_counts[k] += 1
                        else:
                            # Handle scalar values
                            routing_sums[k] = routing_sums.get(k, 0.0) + float(v)
                            routing_counts[k] = routing_counts.get(k, 0) + 1

                # Save gate scores periodically (every 1000 steps) for analysis
                if accelerator.is_main_process and train_steps > 0 and train_steps % 1000 == 0:
                    gate_scores = accelerator.unwrap_model(model).get_last_gate_scores()
                    if gate_scores is not None:
                        gate_scores_dir = os.path.join(experiment_dir, "gate_scores_analysis")
                        modality_bias = getattr(args, "modality_bias_strength_action", None)
                        save_gate_scores(gate_scores, train_steps, gate_scores_dir, modality_bias)

            opt.zero_grad()
            accelerator.backward(loss)

            # === Enhanced gradient norm monitoring ===
            grad_stats = {}
            model_for_grad = accelerator.unwrap_model(model)
            collect_detailed_grad = getattr(args, "collect_stats", False)

            for name, param in model_for_grad.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()

                    # A. Action head vs RGB head gradient norms
                    if "final_layer.a_linear" in name or "final_layer.a_head" in name:
                        grad_stats["grad_norm/action_head"] = grad_norm
                    elif "final_layer.linear" in name and "a_linear" not in name:
                        # RGB head (final_layer.linear)
                        grad_stats["grad_norm/rgb_head"] = grad_norm

                    # B. Router (MoE gate) gradient norms
                    elif "gate.weight" in name or "gate.modality_bias" in name:
                        grad_stats["grad_norm/router"] = grad_stats.get("grad_norm/router", 0.0) + grad_norm

                    # C. Expert AdaLN gradients (per modality)
                    if "norm1_experts" in name:
                        modality_id = name.split(".")[2]
                        modality_name = {0: "rgb", 1: "action", 2: "depth", 3: "force"}.get(int(modality_id), f"mod_{modality_id}")
                        grad_stats[f"grad_norm/adaln_norm1_{modality_name}"] = grad_norm
                    elif "norm2_experts" in name:
                        modality_id = name.split(".")[2]
                        modality_name = {0: "rgb", 1: "action", 2: "depth", 3: "force"}.get(int(modality_id), f"mod_{modality_id}")
                        grad_stats[f"grad_norm/adaln_norm2_{modality_name}"] = grad_norm

                    # D. Shared expert FFN gradients (for comparison)
                    elif "shared_experts" in name and "fc" in name:
                        grad_stats["grad_norm/shared_experts"] = grad_stats.get("grad_norm/shared_experts", 0.0) + grad_norm

            # Calculate grad ratio if both heads present
            if "grad_norm/action_head" in grad_stats and "grad_norm/rgb_head" in grad_stats:
                eps = 1e-8
                grad_stats["grad_norm/ratio_head"] = grad_stats["grad_norm/action_head"] / (grad_stats["grad_norm/rgb_head"] + eps)

            # Accumulate grad stats
            for key in grad_stats:
                grad_sums[key] = grad_sums.get(key, 0.0) + grad_stats[key]
                grad_counts[key] = grad_counts.get(key, 0) + 1
            # =============================================

            opt.step()
            
            # Step learning rate scheduler
            if lr_scheduler is not None:
                lr_scheduler.step()
            
            if not args.without_ema:
                update_ema(ema, model)

            # logging stats
            running_loss += loss_dict["loss"].mean().item()
            if args.action_steps > 0 and "loss_a" in loss_dict:
                running_loss_a += loss_dict["loss_a"].mean().item() * args.action_loss_lambda * (1.0 if train_steps > args.action_loss_start else 0.0)
            if args.use_depth and "loss_depth" in loss_dict:
                running_loss_d += loss_dict["loss_depth"].mean().item()
            if moe_aux_metric is not None:
                running_moe_aux += moe_aux_metric

            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = (running_loss / log_steps)
                avg_loss_a = (running_loss_a / log_steps) if log_steps > 0 else 0.0
                avg_loss_d = (running_loss_d / log_steps) if log_steps > 0 else 0.0
                avg_moe_aux = (running_moe_aux / log_steps) if (log_steps > 0 and getattr(args, "use_moe", False)) else 0.0

                # Enhanced routing stats aggregation with multi-GPU support
                routing_avg = {}
                for k in routing_sums:
                    cnt = routing_counts.get(k, 0)
                    if cnt <= 0:
                        continue

                    # Handle list of tensors (histograms) - need multi-GPU reduction
                    if isinstance(routing_sums[k], list):
                        # Stack tensors and sum across GPUs
                        tensor_list = routing_sums[k]
                        if tensor_list:
                            stacked = torch.stack(tensor_list).sum(dim=0)  # (num_experts,)
                            # Note: For proper multi-GPU, we'd need all_reduce here
                            # For now, normalize by local count
                            routing_avg[k] = stacked.tolist()
                        else:
                            continue
                    else:
                        # Handle scalar values
                        routing_avg[k] = routing_sums[k] / cnt

                # Normalize histograms by token count
                for k in list(routing_avg.keys()):
                    if "/top1_hist" in k or "/topk_hist" in k:
                        modality = k.split("/")[0]  # e.g., "action"
                        token_count_key = f"{modality}/token_count"
                        if token_count_key in routing_avg and isinstance(routing_avg[k], list):
                            # Normalize histogram by total tokens
                            total_tokens = routing_avg[token_count_key]
                            if total_tokens > 0:
                                hist = routing_avg[k]
                                if "/top1_hist" in k:
                                    # Divide by token count
                                    routing_avg[k] = [h / total_tokens for h in hist]
                                elif "/topk_hist" in k:
                                    # Divide by token_count * top_k
                                    top_k = getattr(args, "moe_top_k", 2)
                                    routing_avg[k] = [h / (total_tokens * top_k) for h in hist]

                grad_norm_avg = {}
                for k in grad_sums:
                    cnt = grad_counts.get(k, 0)
                    if cnt > 0:
                        grad_norm_avg[k] = grad_sums[k] / cnt

                if accelerator.is_main_process:
                    # Get current learning rate
                    current_lr = opt.param_groups[0]['lr']
                    log_msg = (f"(step={train_steps:07d}) Train Loss image: {avg_loss:.6f}, "
                               f"Train Loss action:{avg_loss_a:.6f}, Train Loss depth:{avg_loss_d:.6f}, ")
                    if getattr(args, "use_moe", False):
                        log_msg += f"MoE aux loss:{avg_moe_aux:.6f}, "
                        if routing_avg:
                            if "action_hit_rate" in routing_avg:
                                log_msg += f"ActionHit:{routing_avg['action_hit_rate']:.3f}, "
                            if "action_coverage" in routing_avg:
                                log_msg += f"ActionCov:{routing_avg['action_coverage']:.3f}, "
                            if "rgb_coverage" in routing_avg:
                                log_msg += f"RGBCov:{routing_avg['rgb_coverage']:.3f}, "
                    log_msg += f"Train Steps/Sec: {steps_per_sec:.2f}, LR: {current_lr:.2e}"
                    # 添加梯度范数信息
                    if grad_norm_avg:
                        log_msg += " | GradNorm: "
                        grad_parts = []
                        for k in sorted(grad_norm_avg.keys()):
                            modality = k.split("_")[-1]  # rgb/action/depth/force
                            grad_parts.append(f"{modality}={grad_norm_avg[k]:.4f}")
                        log_msg += ", ".join(grad_parts)
                    logger.info(log_msg)
                    if args.use_wandb:
                        import wandb
                        log_payload = {
                            "train/loss_image": avg_loss,
                            "train/steps_per_sec": steps_per_sec,
                            "train/learning_rate": current_lr,
                        }
                        if args.action_steps > 0:
                            log_payload["train/loss_action"] = avg_loss_a
                        if args.use_depth:
                            log_payload["train/loss_depth"] = avg_loss_d
                        if getattr(args, "use_moe", False):
                            log_payload["train/moe_aux_loss"] = avg_moe_aux

                            # Enhanced routing stats with better naming
                            for k, v in routing_avg.items():
                                # Skip histograms (handle separately)
                                if "/top1_hist" in k or "/topk_hist" in k:
                                    continue
                                # Use clearer naming convention
                                if "/" in k:
                                    parts = k.split("/")
                                    if len(parts) == 2:
                                        modality, metric = parts
                                        log_payload[f"routing/{modality}/{metric}"] = v
                                else:
                                    log_payload[f"moe/{k}"] = v

                            # Log histograms separately for plotting
                            for k, v in routing_avg.items():
                                if "/top1_hist" in k:
                                    modality = k.split("/")[0]
                                    for expert_idx, prob in enumerate(v):
                                        log_payload[f"routing/{modality}/top1_hist/e{expert_idx}"] = prob
                                elif "/topk_hist" in k:
                                    modality = k.split("/")[0]
                                    for expert_idx, prob in enumerate(v):
                                        log_payload[f"routing/{modality}/topk_hist/e{expert_idx}"] = prob

                        # Enhanced gradient norm logging with better naming
                        for k, v in grad_norm_avg.items():
                            log_payload[f"grad/{k}"] = v

                        wandb.log(log_payload, step=train_steps)

                running_loss = 0.0
                running_loss_a = 0.0
                running_loss_d = 0.0
                running_moe_aux = 0.0
                routing_sums = {}
                routing_counts = {}
                grad_sums = {}
                grad_counts = {}
                log_steps = 0
                start_time = time()

            # evaluate
            if train_steps > 0 and train_steps % args.eval_every == 0:
                if accelerator.is_main_process:
                    logger.info("start evaluating model")
                    model.eval()
                    with torch.no_grad():
                        input_img = eval_batch['input_img']
                        target_img = eval_batch['future_img']
                        input_depth = eval_batch['input_depth']
                        target_depth = eval_batch['future_depth']
                        rela_action = eval_batch['rela_action']
                        action_cond_b = eval_batch['action_cond']
                        force_cond_b = eval_batch.get('force_cond')
                        y_b = eval_batch['y']

                        z = torch.randn(size=target_img.shape, device=device)
                        noise_depth = torch.randn(size=target_depth.shape, device=device) if args.use_depth else None
                        noise_action = torch.randn(size=rela_action.shape, device=device) if args.action_steps > 0 else None

                        eval_model_kwargs = dict(
                            y=y_b,
                            x_cond=input_img
                        )
                        if args.use_depth:
                            eval_model_kwargs['depth_cond'] = input_depth
                        if noise_action is not None:
                            eval_model_kwargs['noised_action'] = noise_action
                        if noise_depth is not None:
                            eval_model_kwargs['noised_depth'] = noise_depth
                        if action_cond_b is not None:
                            eval_model_kwargs['action_cond'] = action_cond_b
                        # 只在 force_cond 非 None 时传入
                        if force_cond_b is not None and torch.is_tensor(force_cond_b):
                            eval_model_kwargs['force_cond'] = force_cond_b
                        elif getattr(args, "use_force", False):
                            # use_force=True 但数据为 None，传零张量
                            eval_model_kwargs['force_cond'] = torch.zeros(input_img.shape[0], 1, args.force_dim, device=device)
                        samples = eval_diffusion.p_sample_loop(
                            model, z.shape, z, clip_denoised=False, model_kwargs=eval_model_kwargs, progress=True,
                            device=device
                        )
                        if args.use_depth or args.action_steps > 0:
                            img_samples, action_samples, depth_samples = samples
                        else:
                            img_samples = samples
                            action_samples = None
                            depth_samples = None

                        img_mse_error = torch.nn.functional.mse_loss(target_img, img_samples)
                        img_mse_value = img_mse_error.detach().item()
                        logger.info(f"(step={train_steps:07d}) Train img mse: {img_mse_value:.6f}")

                        if args.use_depth and depth_samples is not None:
                            depth_mse_error = torch.nn.functional.mse_loss(target_depth, depth_samples)
                            depth_mse_value = depth_mse_error.detach().item()
                            logger.info(f"(step={train_steps:07d}) Train depth mse: {depth_mse_value:.6f}")
                        else:
                            depth_mse_value = None

                        if args.action_steps > 0 and action_samples is not None:
                            action_mse_error = torch.nn.functional.mse_loss(rela_action, action_samples)
                            action_mse_value = action_mse_error.detach().item()
                            logger.info(f"(step={train_steps:07d}) Train action mse: {action_mse_value:.6f}")
                            if action_mse_value < best_action_loss:
                                best_action_loss = action_mse_value
                                checkpoint_path = f"{checkpoint_dir}/best_action_loss.pt"
                                torch.save({
                                    "model": model.module.state_dict() if accelerator.num_processes > 1 else model.state_dict(),
                                    "args": args
                                }, checkpoint_path)
                                logger.info(f"Saved checkpoint to {checkpoint_path}")
                        else:
                            action_mse_value = None

                    if args.use_wandb:
                        import wandb
                        eval_log = {"eval/loss_image": img_mse_value}
                        if depth_mse_value is not None:
                            eval_log["eval/loss_depth"] = depth_mse_value
                        if action_mse_value is not None:
                            eval_log["eval/loss_action"] = action_mse_value
                            eval_log["eval/best_action_loss"] = best_action_loss
                        wandb.log(eval_log, step=train_steps)

                    # save qualitative imgs
                    img_save_path = os.path.join(eval_dir, 'step_' + str(train_steps))
                    os.makedirs(img_save_path, exist_ok=True)
                    if args.use_depth and depth_samples is not None:
                        depth_samples_np = depth_samples.cpu().detach().numpy()
                        input_depth_np = input_depth.cpu().detach().numpy()
                        target_depth_np = target_depth.cpu().detach().numpy()
                    for i in range(img_samples.shape[0]):
                        input_img_save = vae.decode(input_img[i:i + 1] / 0.18215).sample
                        save_image(input_img_save, os.path.join(img_save_path, str(i) + "_input.png"),
                                   nrow=4, normalize=True, value_range=(-1, 1))
                        if args.use_depth and depth_samples is not None:
                            image = Image.fromarray((input_depth_np[i] * 100)[0].astype(np.uint8))
                            image.save(os.path.join(img_save_path, str(i) + "_input_depth.png"))
                        for j in range(pred_lens):
                            target_img_save = vae.decode(target_img[i:i+1, 4*j:4*(j+1)] / 0.18215).sample
                            samples_img_save = vae.decode(img_samples[i:i+1, 4*j:4*(j+1)] / 0.18215).sample
                            save_image(target_img_save, os.path.join(img_save_path, f"{i}_{j}_target.png"),
                                       nrow=4, normalize=True, value_range=(-1, 1))
                            save_image(samples_img_save, os.path.join(img_save_path, f"{i}_{j}_pred.png"),
                                       nrow=4, normalize=True, value_range=(-1, 1))
                            if args.use_depth and depth_samples is not None:
                                image = Image.fromarray((depth_samples_np[i, j:j+1] * 100)[0].astype(np.uint8))
                                image.save(os.path.join(img_save_path, f"{i}_{j}_pred_depth.png"))
                                image = Image.fromarray((target_depth_np[i, j:j+1] * 100)[0].astype(np.uint8))
                                image.save(os.path.join(img_save_path, f"{i}_{j}_target_depth.png"))

                    model.train()

            # Save checkpoint
            if train_steps % args.ckpt_every == 0:
                if accelerator.is_main_process:
                    # Optionally skip optimizer/lr state to shrink checkpoint size.
                    if getattr(args, "save_model_only", False):
                        checkpoint = {
                            "model": model.module.state_dict() if accelerator.num_processes > 1 else model.state_dict(),
                            "epoch": epoch,
                            "step": train_steps,
                            "args": args,
                        }
                    else:
                        checkpoint = {
                            "model": model.module.state_dict() if accelerator.num_processes > 1 else model.state_dict(),
                            "optimizer": opt.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
                            "epoch": epoch,
                            "step": train_steps,
                            "args": args,
                        }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

    model.eval()  # disable randomized embedding dropout

    if accelerator.is_main_process:
        logger.info("Done!")
        if args.use_wandb:
            import wandb
            wandb.finish()


if __name__ == "__main__":
    # Create argument parser with config file support
    parser = argparse.ArgumentParser(description="Train DiT model with config file support")

    # Config file arg
    parser.add_argument("--config", type=str, default="default.yaml",
                        help="Path to YAML config file (default: configs/default.yaml)")

    # Main args (can be overridden by config)
    parser.add_argument("--feature-path", type=str)
    parser.add_argument("--results-dir", type=str)
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()))
    parser.add_argument("--image-size", type=int, choices=[256, 512])
    parser.add_argument("--num-classes", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--global-batch-size", type=int)
    parser.add_argument("--global-seed", type=int)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"])
    parser.add_argument("--vae-path", type=str)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--adam-beta1", type=float)
    parser.add_argument("--adam-beta2", type=float)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--ckpt-every", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--ckpt-wrapper", action="store_true")
    parser.add_argument("--save-model-only", action="store_true",
                        help="If set, checkpoints only contain model weights (no optimizer/lr scheduler).")
    parser.add_argument("--without-ema", action="store_true")

    # Checkpoint resume
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")

    # Init
    parser.add_argument("--dit-init", type=str)
    parser.add_argument("--rgb-init", type=str)

    # Model components
    parser.add_argument("--attn-mask", action="store_true")
    parser.add_argument("--predict-horizon", type=int)
    parser.add_argument("--skip-step", type=int)

    # Text conditioning
    parser.add_argument("--dynamics", action="store_true")
    parser.add_argument("--text-cond", action="store_true")
    parser.add_argument("--clip-path", type=str)
    parser.add_argument("--text-emb-size", type=int)

    # Depth
    parser.add_argument("--use-depth", action="store_true")
    parser.add_argument("--d-hidden-size", type=int)
    parser.add_argument("--d-patch-size", type=int)
    parser.add_argument("--depth-filter", action="store_true")

    # Force
    parser.add_argument("--use-force", action="store_true")
    parser.add_argument("--force-dim", type=int)
    parser.add_argument("--force-stats-path", type=str)
    parser.add_argument("--force-mean", type=float, nargs="+")
    parser.add_argument("--force-std", type=float, nargs="+")

    # Action
    parser.add_argument("--learnable-action-pos", action="store_true")
    parser.add_argument("--action-steps", type=int)
    parser.add_argument("--action-dim", type=int)
    parser.add_argument("--action-scale", type=float)
    parser.add_argument("--absolute-action", action="store_true")
    parser.add_argument("--action-condition", action="store_true")

    # Loss
    parser.add_argument("--action-loss-lambda", type=float)
    parser.add_argument("--action-loss-start", type=int)

    # MoE
    parser.add_argument("--use-moe", action="store_true")
    parser.add_argument("--num-experts", type=int)
    parser.add_argument("--moe-top-k", type=int)
    parser.add_argument("--aux-loss-weight", type=float)
    parser.add_argument("--router-z-loss-weight", type=float)
    parser.add_argument("--moe-start-layer", type=int)
    parser.add_argument("--moe-shared-experts", type=int)

    # Expert AdaLN
    parser.add_argument("--use-expert-adaln", action="store_true",
                        help="Use per-modality expert LayerNorms in DiT blocks (CogVideoX style)")

    # Wandb
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str)
    parser.add_argument("--wandb-run-name", type=str)

    # Parse CLI
    cli_args = parser.parse_args()

    # Load YAML config and merge with CLI
    try:
        args = load_config(cli_args.config, "configs", cli_args)
        print(f"✓ Loaded configuration from: configs/{cli_args.config}")
    except FileNotFoundError:
        print(f"⚠ Config file not found: configs/{cli_args.config}")
        print("Using default configuration...")
        args = load_config("default.yaml", "configs")
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        print("Falling back to command line arguments only...")
        args = cli_args

    # Convert dashes to underscores for compatibility
    for attr_name in dir(args):
        if '-' in attr_name and not attr_name.startswith('_'):
            new_name = attr_name.replace('-', '_')
            if not hasattr(args, new_name):
                setattr(args, new_name, getattr(args, attr_name))

    main(args)

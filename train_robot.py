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
from accelerate import Accelerator
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


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """Trains a new DiT model."""
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    accelerator = Accelerator()
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

        # 3) Load adapted weights (allow missing due to modules like y_embedder difference)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if accelerator.is_main_process:
            print(f"✓ Successfully loaded & adapted pretrained weights from {args.rgb_init}")
            if missing:
                print(f"Missing keys after load (ok if due to config changes): {len(missing)}")
            if unexpected:
                print(f"Unexpected keys after load: {len(unexpected)}")

        # 4) If text_cond changed, reset y_embedder to a simple class embedder
        if not args.text_cond:
            with torch.no_grad():
                model.y_embedder = nn.Linear(args.num_classes, model.hidden_size, bias=True)
                nn.init.normal_(model.y_embedder.weight, std=0.02)
                nn.init.zeros_(model.y_embedder.bias)
            if accelerator.is_main_process:
                print("✓ Re-initialized y_embedder for class-only guidance.")
    # ================================================================

    model = model.to(device)

    if not args.without_ema:
        ema = deepcopy(model).to(device)  # EMA of the model
        requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    eval_diffusion = create_diffusion(str(250))

    if accelerator.is_main_process:
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    lr = float(getattr(args, 'learning_rate', 1e-4))
    weight_decay = float(getattr(args, 'weight_decay', 0.0))
    beta1 = float(getattr(args, 'adam_beta1', 0.9))
    beta2 = float(getattr(args, 'adam_beta2', 0.999))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(beta1, beta2))

    # Learning Rate Scheduler
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

    # Data
    dataset = RobotDataset(args.feature_path, args)
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // accelerator.num_processes),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    if accelerator.is_main_process:
        logger.info(f"Global batch size {args.global_batch_size:,} num_processes ({accelerator.num_processes})")
        logger.info(f"Dataset contains {len(dataset):,} images ({args.feature_path})")

    # Prepare for distributed
    if not args.without_ema:
        update_ema(ema, model, decay=0)  # sync init
        ema.eval()
    model.train()  # important! enables embedding dropout for classifier-free guidance
    model, opt, loader = accelerator.prepare(model, opt, loader)

    # Monitor vars
    train_steps = 0
    log_steps = 0
    running_loss = 0.0
    running_loss_a = 0.0
    running_loss_d = 0.0
    start_time = time()
    eval_batch = None
    best_action_loss = 1e8

    if accelerator.is_main_process:
        logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
        if not args.dynamics:
            raise NotImplementedError("Set --dynamics for dynamics modeling.")
        for x_cond, x, depth_cond, depth, action_cond, action, y in loader:
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

            if eval_batch is None:
                eval_batch = {
                    'input_img': x_cond,
                    'future_img': x,
                    'input_depth': depth_cond,
                    'future_depth': depth,
                    'rela_action': action,
                    'action_cond': action_cond,
                    'y': y,
                }

            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            if args.action_steps > 0 and "loss_a" in loss_dict:
                a_coffi = 1.0 if train_steps > args.action_loss_start else 0.0
                loss = loss + loss_dict["loss_a"].mean() * args.action_loss_lambda * a_coffi
            if args.use_depth and "loss_depth" in loss_dict:
                loss = loss + loss_dict["loss_depth"].mean()

            opt.zero_grad()
            accelerator.backward(loss)
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

            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = (running_loss / log_steps)
                avg_loss_a = (running_loss_a / log_steps) if log_steps > 0 else 0.0
                avg_loss_d = (running_loss_d / log_steps) if log_steps > 0 else 0.0

                if accelerator.is_main_process:
                    # Get current learning rate
                    current_lr = opt.param_groups[0]['lr']
                    logger.info(f"(step={train_steps:07d}) Train Loss image: {avg_loss:.6f}, "
                                f"Train Loss action:{avg_loss_a:.6f}, Train Loss depth:{avg_loss_d:.6f}, "
                                f"Train Steps/Sec: {steps_per_sec:.2f}, LR: {current_lr:.2e}")
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
                        wandb.log(log_payload, step=train_steps)

                running_loss = 0.0
                running_loss_a = 0.0
                running_loss_d = 0.0
                log_steps = 0
                start_time = time()

            # evaluate
            if train_steps > 0 and train_steps % args.eval_every == 0:
                if accelerator.is_main_process:
                    logger.info("start evaluating model")
                    model.eval()
                    input_img = eval_batch['input_img']
                    target_img = eval_batch['future_img']
                    input_depth = eval_batch['input_depth']
                    target_depth = eval_batch['future_depth']
                    rela_action = eval_batch['rela_action']
                    action_cond_b = eval_batch['action_cond']
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
                    checkpoint = {
                        "model": model.module.state_dict() if accelerator.num_processes > 1 else model.state_dict(),
                        "args": args
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
    parser.add_argument("--without-ema", action="store_true")

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

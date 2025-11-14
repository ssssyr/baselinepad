# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from accelerate import Accelerator

from config_loader import load_config, save_config
from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
import torch.nn as nn
import cv2
from torchvision.utils import save_image

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


# Import dataset classes from the new dataset module
from dataset import RobotDataset

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup accelerator:
    accelerator = Accelerator()
    device = accelerator.device

    # Setup an experiment folder:
    if accelerator.is_main_process:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        from datetime import datetime
        uuid = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-{uuid}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        eval_dir = f"{experiment_dir}/eval"
        vae_path = getattr(args, 'vae_path', "/cephfs/shared/llm/sd-vae-ft-mse")
        vae = AutoencoderKL.from_pretrained(vae_path, local_files_only=True, trust_remote_code=True).to(device)
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
            wandb_run = wandb.init(
                project=wandb_project,
                name=run_name,
                config=vars(args),
            )

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    pred_lens = args.predict_horizon

    if args.rgb_init is not None:
        model = DiT_models[args.model](
            input_size=latent_size,
            num_classes=args.num_classes,
            args=args,
        )
        # load model from args.rgb_init
        checkpoint = torch.load(args.rgb_init, map_location='cpu')
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                pretrained_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                pretrained_dict = checkpoint["state_dict"]
            else:
                pretrained_dict = checkpoint
        else:
            pretrained_dict = checkpoint
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        print("load",len(pretrained_dict.keys()))
        print("model", len(model_dict.keys()))
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        model.load_state_dict(model_dict)
        model = model.to('cpu')
    else:
        # train from scratch
        model = DiT_models[args.model](
            input_size=latent_size,
            num_classes=args.num_classes,
            args=args,
        )
    
    # Note that parameter initialization is done within the DiT constructor
    model = model.to(device)

    if not args.without_ema:
        ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    eval_diffusion = create_diffusion(str(250))
    # vae = AutoencoderKL.from_pretrained("/home/gyj/llm/sd-vae-ft-mse").to(device)
    if accelerator.is_main_process:
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    lr = getattr(args, 'learning_rate', 1e-4)
    weight_decay = getattr(args, 'weight_decay', 0.0)
    beta1 = getattr(args, 'adam_beta1', 0.9)
    beta2 = getattr(args, 'adam_beta2', 0.999)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(beta1, beta2))

    # Setup data:
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

    # Prepare models for training:
    if not args.without_ema:
        update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
        ema.eval()  # EMA model should always be in eval mode
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    model, opt, loader = accelerator.prepare(model, opt, loader)

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    running_loss_a = 0
    running_loss_d = 0
    start_time = time()
    eval_batch = None
    best_action_loss = 1e8
    
    if accelerator.is_main_process:
        logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
        if not args.dynamics:
            raise NotImplementedError
        for x_cond, x, depth_cond, depth, action_cond, action, y in loader:

            x_cond = x_cond.squeeze(dim=1).to(device)
            x = x.squeeze(dim=1).to(device) # (B, 1, 4, 32,32)
            y = y.squeeze(dim=1).to(device) # text: (B,512) class:(B)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)

            if args.use_depth:
                depth_cond = depth_cond.to(device)
                depth = depth.to(device)
            
            if args.action_steps>0:
                action = action.to(device)
            if args.action_steps>0 and args.action_condition:
                action_cond = action_cond.to(device)

            model_kwargs = dict(y=y,x_cond=x_cond,action=action,depth_cond=depth_cond,depth=depth,action_cond=action_cond)
            if eval_batch == None:
                eval_batch = {
                    'input_img': x_cond,
                    'future_img': x,
                    'input_depth' : depth_cond,
                    'future_depth' : depth,
                    'rela_action' : action,
                    'action_cond': action_cond,
                    'y': y,
                }

            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            if args.action_steps>0:
                a_coffi = 1.0 if train_steps>args.action_loss_start else 0.0
                loss += loss_dict["loss_a"].mean()*args.action_loss_lambda*a_coffi
            if args.use_depth:
                loss += loss_dict["loss_depth"].mean()
            opt.zero_grad()
            accelerator.backward(loss)            
            opt.step()
            if not args.without_ema:
                update_ema(ema, model)

            # Log loss values:
            running_loss += loss_dict["loss"].mean().item()
            if args.action_steps>0:
                running_loss_a += loss_dict["loss_a"].mean().item()*args.action_loss_lambda*a_coffi
            if args.use_depth:
                running_loss_d += loss_dict["loss_depth"].mean().item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                # avg_loss = avg_loss.item() / accelerator.num_processes # why divide?
                avg_loss = avg_loss.item()
                avg_loss_a = torch.tensor(running_loss_a / log_steps, device=device)
                avg_loss_a = avg_loss_a.item()
                avg_loss_d = torch.tensor(running_loss_d / log_steps, device=device)
                avg_loss_d = avg_loss_d.item()
                if accelerator.is_main_process:
                    logger.info(f"(step={train_steps:07d}) Train Loss image: {avg_loss:.6f}, Train Loss action:{avg_loss_a:.6f}, Train Loss depth:{avg_loss_d:.6f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    if args.use_wandb:
                        log_payload = {
                            "train/loss_image": avg_loss,
                            "train/steps_per_sec": steps_per_sec,
                        }
                        if args.action_steps > 0:
                            log_payload["train/loss_action"] = avg_loss_a
                        if args.use_depth:
                            log_payload["train/loss_depth"] = avg_loss_d
                        wandb.log(log_payload, step=train_steps)
                # Reset monitoring variables:
                running_loss, running_loss_a, running_loss_d = 0, 0, 0
                log_steps = 0
                start_time = time()

            # evaluate dit
            if train_steps % args.eval_every == 1 and train_steps > 0:
                if accelerator.is_main_process:
                    logger.info("start evaluating model")
                    model.eval()
                    input_img = eval_batch['input_img']
                    target_img = eval_batch['future_img']
                    input_depth = eval_batch['input_depth']
                    target_depth = eval_batch['future_depth']
                    rela_action = eval_batch['rela_action']
                    action_cond = eval_batch['action_cond']
                    y = eval_batch['y']
                    #target_action = eval_batch['a']
                    z = torch.randn(size=target_img.shape, device=device)
                    noise_depth = torch.randn(size=target_depth.shape, device=device)
                    noise_action = torch.randn(size=rela_action.shape, device=device)
                    #noise_action = torch.randn(input_img.shape[0], 4, args.action_lens, device=device)
                    eval_model_kwargs = dict(y=y, x_cond=input_img,noised_action=noise_action,depth_cond=input_depth,noised_depth=noise_depth,action_cond=action_cond)
                    samples = eval_diffusion.p_sample_loop(
                        model, z.shape, z, clip_denoised=False, model_kwargs=eval_model_kwargs, progress=True,
                        device=device
                    )
                    if args.use_depth or args.action_steps>0:
                        img_samples, action_samples, depth_samples =samples
                    else:
                        img_samples = samples
                    img_mse_error = torch.nn.functional.mse_loss(target_img, img_samples)
                    img_mse_value = img_mse_error.detach().item()
                    logger.info(f"(step={train_steps:07d}) Train img mse: {img_mse_value:.6f}")
                    if args.use_depth:
                        depth_mse_error = torch.nn.functional.mse_loss(target_depth, depth_samples)
                        depth_mse_value = depth_mse_error.detach().item()
                        logger.info(f"(step={train_steps:07d}) Train depth mse: {depth_mse_value:.6f}")
                    else:
                        depth_mse_value = None
                    if args.action_steps>0:
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
                        eval_log = {"eval/loss_image": img_mse_value}
                        if depth_mse_value is not None:
                            eval_log["eval/loss_depth"] = depth_mse_value
                        if action_mse_value is not None:
                            eval_log["eval/loss_action"] = action_mse_value
                            eval_log["eval/best_action_loss"] = best_action_loss
                        wandb.log(eval_log, step=train_steps)
                    #img_samples = img_samples.reshape((N,pred_lens,-1,img_samples.shape[2],img_samples.shape[3]))
                    #depth_samples = depth_samples.reshape((N, pred_lens, -1, depth_samples.shape[2], depth_samples.shape[3]))
                    img_save_path = os.path.join(eval_dir, 'step_' + str(train_steps))
                    os.makedirs(img_save_path, exist_ok=True)
                    if args.use_depth:
                        depth_samples = depth_samples.cpu().detach().numpy()
                        input_depth = input_depth.cpu().detach().numpy()
                        target_depth = target_depth.cpu().detach().numpy()
                    for i in range(img_samples.shape[0]):
                        input_img_save = vae.decode(input_img[i:i + 1] / 0.18215).sample
                        save_image(input_img_save, os.path.join(img_save_path, str(i) + "_input.png"), nrow=4,
                                   normalize=True,
                                   value_range=(-1, 1))
                        if args.use_depth:
                            image = Image.fromarray((input_depth[i] * 100)[0].astype(np.uint8))
                            image.save(os.path.join(img_save_path, str(i) + "_input_depth.png"))
                        for j in range(pred_lens):
                            target_img_save = vae.decode(target_img[i:i+1,4*j:4*(j+1)] / 0.18215).sample
                            samples_img_save = vae.decode(img_samples[i:i+1,4*j:4*(j+1)] / 0.18215).sample

                            save_image(target_img_save, os.path.join(img_save_path, str(i) + '_' + str(j) + "_target.png"), nrow=4,
                                       normalize=True,
                                       value_range=(-1, 1))
                            save_image(samples_img_save, os.path.join(img_save_path, str(i) + '_' + str(j) + "_pred.png"), nrow=4,
                                       normalize=True,
                                       value_range=(-1, 1))
                            #print('depth_samples_shape:',depth_samples.shape)

                            if args.use_depth:
                                image = Image.fromarray((depth_samples[i,j:j+1]*100)[0].astype(np.uint8))
                                image.save(os.path.join(img_save_path, str(i) + '_' + str(j) + "_pred_depth.png"))
                                image = Image.fromarray((target_depth[i,j:j+1] * 100)[0].astype(np.uint8))
                                image.save(os.path.join(img_save_path, str(i) + '_' + str(j) + "_target_depth.png"))
                model.train()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.module.state_dict() if accelerator.num_processes > 1 else model.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    if accelerator.is_main_process:
        logger.info("Done!")
        if args.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    # Create argument parser with config file support
    parser = argparse.ArgumentParser(description="Train DiT model with config file support")
    
    # Config file argument
    parser.add_argument("--config", type=str, default="default.yaml", 
                       help="Path to YAML config file (default: configs/default.yaml)")
    
    # All other arguments (these can override config file values)
    parser.add_argument("--feature-path", type=str, help="Path to training data features")
    parser.add_argument("--results-dir", type=str, help="Directory to save checkpoints and logs")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), help="Model architecture")
    parser.add_argument("--image-size", type=int, choices=[256, 512], help="Image size (must be divisible by 8)")
    parser.add_argument("--num-classes", type=int, help="Number of classes for classifier-free guidance")
    parser.add_argument("--epochs", type=int, help="Total training epochs")
    parser.add_argument("--global-batch-size", type=int, help="Total batch size across all GPUs")
    parser.add_argument("--global-seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], help="VAE model type")
    parser.add_argument("--vae-path", type=str, help="Path to VAE model directory")
    parser.add_argument("--learning-rate", type=float, help="Learning rate for AdamW optimizer")
    parser.add_argument("--weight-decay", type=float, help="Weight decay for AdamW optimizer")
    parser.add_argument("--adam-beta1", type=float, help="AdamW beta1 parameter")
    parser.add_argument("--adam-beta2", type=float, help="AdamW beta2 parameter")
    parser.add_argument("--num-workers", type=int, help="Number of data loader workers")
    parser.add_argument("--log-every", type=int, help="Log training status every N steps")
    parser.add_argument("--ckpt-every", type=int, help="Save checkpoint every N steps")
    parser.add_argument("--eval-every", type=int, help="Evaluate model every N steps")
    parser.add_argument("--ckpt-wrapper", action="store_true", help="Wrapper for saving memory")
    parser.add_argument("--without-ema", action="store_true", help="Disable Exponential Moving Average")

    # Initialization
    parser.add_argument("--dit-init", type=str, help="Path to pretrained DiT weights")
    parser.add_argument("--rgb-init", type=str, help="Path to pretrained model for RGB components")

    # Model components
    parser.add_argument("--attn-mask", action="store_true", help="Use attention mask")
    parser.add_argument("--predict-horizon", type=int, help="Number of future frames to predict")
    parser.add_argument("--skip-step", type=int, help="Steps to skip between frames")

    # Text conditioning
    parser.add_argument("--dynamics", action="store_true", help="Enable dynamics modeling")
    parser.add_argument("--text-cond", action="store_true", help="Enable text conditioning")
    parser.add_argument("--clip-path", type=str, help="Path to CLIP model")
    parser.add_argument("--text-emb-size", type=int, help="Dimension of text embeddings")
    
    # Depth conditioning
    parser.add_argument("--use-depth", action="store_true", help="Enable depth conditioning")
    parser.add_argument("--d-hidden-size", type=int, help="Hidden size for depth encoder")
    parser.add_argument("--d-patch-size", type=int, help="Patch size for depth encoder")
    parser.add_argument("--depth-filter", action="store_true", help="Apply filter to depth images")

    # Action conditioning
    parser.add_argument("--learnable-action-pos", action="store_true", help="Use learnable positional embeddings for actions")
    parser.add_argument("--action-steps", type=int, help="Number of action steps to predict/condition on")
    parser.add_argument("--action-dim", type=int, help="Dimension of the action space")
    parser.add_argument("--action-scale", type=float, help="Scaling factor for actions")
    parser.add_argument("--absolute-action", action="store_true", help="Use absolute actions instead of relative")
    parser.add_argument("--action-condition", action="store_true", help="Condition on the current action/pose")

    # Loss configuration
    parser.add_argument("--action-loss-lambda", type=float, help="Weight for the action prediction loss")
    parser.add_argument("--action-loss-start", type=int, help="Step to start applying action loss")
    
    # Wandb logging
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, help="Wandb project name")
    parser.add_argument("--wandb-run-name", type=str, help="Wandb run name")

    # Parse command line arguments
    cli_args = parser.parse_args()
    
    # Load configuration from YAML file and merge with CLI args
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

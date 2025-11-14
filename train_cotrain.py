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
import json
import random

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
from dataset import CustomDataset2

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
            wandb_run = wandb.init(
                project=wandb_project,
                name=run_name,
                config=vars(args),
            )

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    pred_lens = args.predict_horizon
    if args.dit_init is not None:
        # TODO: initialize model with pretrained DiT weights, some layers are uncompitable
        assert args.model == "DiT-XL/2" # only DiT-XL-2 has pretrained ckpt
        args_dit = deepcopy(args)
        args_dit.dynamics=False
        args_dit.text_cond = False
        model = DiT_models[args.model](
            input_size=latent_size,
            num_classes=args.num_classes,
            args=args_dit,
        )
        # load model from args.dit_init
        state_dict = torch.load(args.dit_init, map_location='cpu')
        # print(state_dict.keys())
        # print(len(state_dict.keys()))
        # for k in state_dict.keys():
        #     print(k,torch.max(state_dict[k]),torch.min(state_dict[k]))   
        model = model.to('cpu')
        model.load_state_dict(state_dict)
        model = model.to('cpu')
        from models import LanguageEmbedder, FinalLayer
        with torch.no_grad():
            c = model.in_channels
            input_c = c*(1+args.predict_horizon)
            out_c = c*2*args.predict_horizon
            new_conv_in = nn.Conv2d(
                input_c, model.hidden_size, kernel_size=model.patch_size, stride=model.patch_size, bias=True
            )
            nn.init.constant_(new_conv_in.weight, 0)
            for i in range(args.predict_horizon): # copy weight for every noised image
                new_conv_in.weight[:, i*c:(i+1)*c, :, :].copy_(model.x_embedder.proj.weight)
            # new_conv_in.weight[:, :model.in_channels, :, :].copy_(model.x_embedder.proj.weight)
            model.x_embedder.proj = new_conv_in
            model.y_embedder = LanguageEmbedder(model.hidden_size,args)
            nn.init.constant_(model.y_embedder.embedding_table.bias, 0)
            nn.init.constant_(model.y_embedder.embedding_table.weight, 0)

            # (32,1156) -> (96,1156)
            new_final_layer = FinalLayer(model.hidden_size, model.patch_size,out_c,args)
            for i in range(args.predict_horizon):
                d = model.patch_size ** 2 *model.in_channels*2
                new_final_layer.linear.weight[i*d:(i+1)*d,:].copy_(model.final_layer.linear.weight)
                new_final_layer.linear.bias[i*d:(i+1)*d].copy_(model.final_layer.linear.bias)
                new_final_layer.adaLN_modulation[-1].weight.copy_(model.final_layer.adaLN_modulation[-1].weight)
                new_final_layer.adaLN_modulation[-1].bias.copy_(model.final_layer.adaLN_modulation[-1].bias)
            model.final_layer = new_final_layer

            # for block in model.blocks:
            #     nn.init.constant_(block.adaLN_modulation[1].weight,0)
            #     nn.init.constant_(block.adaLN_modulation[1].bias,0)
    elif args.rgb_init is not None:
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
    model_name = []
    model_shape = {}
    # if accelerator.is_main_process:
    #     for name, param in model.named_parameters():
    #         # if param.requires_grad:
    #         model_name.append(name)
    #         model_shape[name] = param.shape
            # if not "block" in name :
            #     # print grad
            #     print(name,param.requires_grad)

        # print("load",len(state_dict.keys()),state_dict.keys())
        # print("after",len(model_name))
    if not args.without_ema:
        ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    eval_diffusion = create_diffusion(str(250))
    # vae = AutoencoderKL.from_pretrained("/home/gyj/llm/sd-vae-ft-mse").to(device)
    if accelerator.is_main_process:
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    lr = float(getattr(args, 'learning_rate', 1e-4))
    weight_decay = float(getattr(args, 'weight_decay', 0.0))
    beta1 = float(getattr(args, 'adam_beta1', 0.9))
    beta2 = float(getattr(args, 'adam_beta2', 0.999))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(beta1, beta2))

    # Setup data:
    if args.dynamics:
        dataset = CustomDataset2(args.feature_path, args)
    else:
        raise NotImplementedError

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
        for x_cond, x, y, action, depth_cond, depth, action_cond, loss_mask in loader:
            x_cond = x_cond.squeeze(dim=1).to(device) if args.dynamics else None # (B,1,4,32,32)
            x = x.squeeze(dim=1).to(device) # (B, 1, 4, 32,32)
            y = y.squeeze(dim=1).to(device) # text: (B,512) class:(B)
            # action (B,action_steps,action_dim) 
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            loss_mask = loss_mask.squeeze(dim=1).to(device)

            if args.use_depth:
                depth_cond = depth_cond.to(device)
                depth = depth.to(device)
            
            if args.action_steps>0 and args.action_condition:
                action_cond = action_cond.to(device)

            model_kwargs = dict(y=y,x_cond=x_cond,depth_cond=depth_cond,depth=depth)
            if args.action_steps > 0:
                model_kwargs['action'] = action
            if args.action_steps > 0 and args.action_condition:
                model_kwargs['action_cond'] = action_cond
            if eval_batch == None:
                eval_batch = {
                    'input_img': x_cond,
                    'future_img': x,
                    'input_depth' : depth_cond,
                    'future_depth' : depth,
                    'rela_action' : action,
                    'action_cond': action_cond,
                    'y': y,
                    'loss_mask': loss_mask
                }

            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            # print("loss_mask",loss_mask.shape)
            # print("loss_dict",loss_dict["loss"].shape)
            loss = loss_dict["loss"].mean()
            if args.action_steps>0:
                # print("action_loss",loss_dict["loss_a"].shape)
                # print("loss_mask",loss_mask.shape)
                # element wise multiply
                a_coffi = 1.0 if train_steps>args.action_loss_start else 0.0
                loss += (loss_dict["loss_a"]*loss_mask).mean()*args.action_loss_lambda*a_coffi
            if args.use_depth:
                loss += (loss_dict["loss_depth"]*loss_mask).mean()
            opt.zero_grad()
            accelerator.backward(loss)
            # if accelerator.is_main_process:
            #     for name, param in model.named_parameters():
            #         name = name.replace("module.", "")
            #         print(name,param.shape)
            #         assert model_shape[name] == param.shape
            #         if "pos_embed" not in name:
            #             assert model_shape[name] == param.grad.shape
            
            opt.step()
            if not args.without_ema:
                update_ema(ema, model)

            # Log loss values:
            running_loss += loss_dict["loss"].mean().item()
            if args.action_steps>0:
                running_loss_a += (loss_dict["loss_a"]*loss_mask).mean().item()*args.action_loss_lambda*a_coffi
            if args.use_depth:
                running_loss_d += (loss_dict["loss_depth"]*loss_mask).mean().item()
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
            if train_steps > 0 and train_steps % args.eval_every == 0:
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
                    loss_mask = eval_batch['loss_mask']
                    #target_action = eval_batch['a']
                    z = torch.randn(size=target_img.shape, device=device)
                    noise_depth = torch.randn(size=target_depth.shape, device=device)
                    noise_action = torch.randn(size=rela_action.shape, device=device)
                    #noise_action = torch.randn(input_img.shape[0], 4, args.action_lens, device=device)
                    eval_model_kwargs = dict(y=y, x_cond=input_img,depth_cond=input_depth)
                    if args.use_depth:
                        eval_model_kwargs['noised_depth'] = noise_depth
                    if args.action_steps > 0:
                        eval_model_kwargs['noised_action'] = noise_action
                    if args.action_steps > 0 and args.action_condition:
                        eval_model_kwargs['action_cond'] = action_cond
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
                        print("action_samples",action_samples.shape,"action_gt",rela_action.shape, "loss_mask", loss_mask.shape)
                        action_mse_error = ((rela_action-action_samples)**2*loss_mask.unsqueeze(1).unsqueeze(1)).mean()
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
                        # "ema": ema.state_dict(),
                        # "opt": opt.state_dict(),
                        "args": args
                    }
                    # if not args.without_ema:
                    #     checkpoint["ema"] = ema.state_dict()
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    
    if accelerator.is_main_process:
        logger.info("Done!")


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path", type=str, default="features")
    parser.add_argument("--video-path", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--vae-path", type=str, help="Path to VAE model directory")
    parser.add_argument("--learning-rate", type=float, help="Learning rate for AdamW optimizer")
    parser.add_argument("--weight-decay", type=float, help="Weight decay for AdamW optimizer")
    parser.add_argument("--adam-beta1", type=float, help="AdamW beta1 parameter")
    parser.add_argument("--adam-beta2", type=float, help="AdamW beta2 parameter")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=30_000)

    parser.add_argument("--dynamics", action="store_true")
    parser.add_argument("--skip_step", type=int, default=10)
    parser.add_argument("--text_cond", action="store_true")
    parser.add_argument("--clip_path", type=str, default="/home/gyj/llm/clip-vit-base-patch32")
    parser.add_argument("--text_emb_size", type=int, default=512)

    parser.add_argument("--without_ema", action="store_true")
    parser.add_argument("--dit_init", type=str, default=None)
    parser.add_argument("--rgb_init", type=str, default=None)
    # action_steps,action_dim
    parser.add_argument("--action_steps", type=int, default=0)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--learnable_action_pos", action="store_true")
    # attn_mask
    parser.add_argument("--attn_mask", action="store_true")
    # predict_horizon
    parser.add_argument("--predict_horizon", type=int, default=1)
    # ckpt_wrapper
    parser.add_argument("--ckpt_wrapper", action="store_true")
    # use_depth
    parser.add_argument("--use_depth", action="store_true")
    # d_hidden_size
    parser.add_argument("--d_hidden_size", type=int, default=32)
    parser.add_argument("--d_patch_size", type=int, default=8)
    # use depth_filter
    parser.add_argument("--depth_filter", action="store_true")
    # action_scale
    parser.add_argument("--action_scale", type=float, default=10)
    # eval_every
    parser.add_argument("--eval_every", type=int, default=5000)
    # absolute_action
    parser.add_argument("--absolute_action", action="store_true")
    # action_condition
    parser.add_argument("--action_condition", action="store_true")
    # action_loss_lambda
    parser.add_argument("--action_loss_lambda", type=float, default=1.0)
    # action_loss_start
    parser.add_argument("--action_loss_start", type=int, default=0)
    # wandb logging
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)

    
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

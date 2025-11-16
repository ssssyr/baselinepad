# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Complete single GPU version of extract_features.py for MetaWorld with action data.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import argparse
import logging
import os
import json
from glob import glob
from time import time

from diffusers.models import AutoencoderKL

#################################################################################
#                             Image Preprocessing Functions                         #
#################################################################################

def center_crop_arr(pil_image, image_size):
    """
    Center crops a PIL image to the specified size.
    """
    while min(pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def main(args):
    """
    Extract features using single GPU with complete MetaWorld action data.
    """
    assert torch.cuda.is_available(), "Feature extraction currently requires at least one GPU."

    # Setup device:
    device_idx = 0  # Use first GPU
    device = f"cuda:{device_idx}"
    torch.cuda.set_device(device)
    seed = args.global_seed
    torch.manual_seed(seed)
    print(f"Starting seed={seed}, device={device}.")

    # Setup a feature folder:
    os.makedirs(args.features_path, exist_ok=True)

    # Create img encoder model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    vae = AutoencoderKL.from_pretrained("/home/syr/code/models/sd-vae-ft-mse/").to(device)
    # Create text encoder model:
    from transformers import AutoTokenizer, CLIPTextModelWithProjection
    model = CLIPTextModelWithProjection.from_pretrained("/home/syr/code/models/clip-vit-base-patch32").to(device)
    tokenizer = AutoTokenizer.from_pretrained("/home/syr/code/models/clip-vit-base-patch32/")

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    # Load dataset info (action and state data)
    print(f"Loading dataset info from {args.data_path}")
    dataset_info_path = os.path.join(args.data_path, "dataset_info.json")
    if not os.path.exists(dataset_info_path):
        print(f"Warning: No dataset_info.json found at {dataset_info_path}")
        print("Using dummy action data...")
        # Create dummy data structure
        json_all = [{"instruction": "press the button", "features": [[0,0,0,1]] * 1000, "action": [[0,0,0,1]] * 1000} for _ in range(50)]
    else:
        with open(dataset_info_path, "r") as f:
            json_all = json.load(f)

    # Get image directories
    image_paths = sorted(glob(os.path.join(args.data_path, "class_*/*.png")))
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {args.data_path}/class_*/")

    print(f"Found {len(image_paths)} images in {len(json_all)} classes")

    vae.eval()
    model.eval()

    train_steps = 0
    dataset_info = []

    with torch.no_grad():
        for traj_id in range(len(json_all)):
            # Create episode directory
            episode_dir = os.path.join(args.features_path, f"episode{traj_id:07}")
            os.makedirs(episode_dir, exist_ok=True)

            instruction = json_all[traj_id]["instruction"]
            action_list = json_all[traj_id]["features"]  # Using features as action (should be "action" in proper data)

            # Process text embedding once per episode
            text_inputs = tokenizer([instruction], padding=True, return_tensors="pt").to(device)
            text_embeds = model(**text_inputs).text_embeds
            text_embed_path = os.path.join(episode_dir, "text_clip.npy")
            np.save(text_embed_path, text_embeds.cpu().numpy())

            # Get images for this episode
            episode_images = sorted(glob(os.path.join(args.data_path, f"class_{traj_id:06d}/*.png")))

            print(f"Processing episode {traj_id}, instruction: '{instruction}', {len(episode_images)} images")

            for episode_steps, img_path in enumerate(episode_images):
                # Load and preprocess image
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(device)

                # Encode to latent
                x = vae.encode(img_tensor).latent_dist.sample().mul_(0.18215)

                # Save features
                feature_path = os.path.join(episode_dir, f"color_wrist_1_{episode_steps:04}.npy")
                np.save(feature_path, x.cpu().numpy())

                # Get action and state data
                if episode_steps < len(action_list):
                    action = action_list[episode_steps]
                    state = action_list[episode_steps]  # Use action as state if no separate state
                else:
                    # Default action/state if not available
                    action = [0, 0, 0, 1]
                    state = [0, 0, 0, 1]

                # Add to dataset info
                dataset_info.append({
                    "idx": str(train_steps),
                    "episode": str(traj_id),
                    "frame": str(episode_steps),
                    "wrist_1": f'episode{traj_id:07}/color_wrist_1_{episode_steps:04}.npy',
                    "label": str(traj_id),
                    "instruction": instruction,
                    "ins_emb_path": f'episode{traj_id:07}/text_clip.npy',
                    "action": action,
                    "state": state
                })

                train_steps += 1

                if train_steps % 100 == 0:
                    print(f"Processed {train_steps} steps...")

    print(f"Feature extraction complete! Processed {train_steps} total frames.")

    # Save dataset metadata
    print("Saving dataset metadata...")
    with open(os.path.join(args.features_path, "dataset_rgb_s_d.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"Dataset info saved to {os.path.join(args.features_path, 'dataset_rgb_s_d.json')}")
    print(f"Total samples: {len(dataset_info)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--features-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--global-batch-size", type=int, default=1)  # Process one image at a time for proper sequencing
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    main(args)
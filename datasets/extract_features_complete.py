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

    # Find all task subdirectories by looking for dataset_info.json
    task_json_paths = sorted(glob(os.path.join(args.data_path, "*/dataset_info.json")))
    if not task_json_paths:
        raise ValueError(f"No 'dataset_info.json' found in any subdirectories of {args.data_path}. Please check your data structure.")
    
    print(f"Found {len(task_json_paths)} tasks to process.")

    vae.eval()
    model.eval()

    # Global counters and lists for aggregating data from all tasks
    global_train_steps = 0
    global_episode_idx = 0
    all_dataset_info = []

    with torch.no_grad():
        # Outer loop: iterate over each task
        for task_json_path in task_json_paths:
            task_dir = os.path.dirname(task_json_path)
            task_name = os.path.basename(task_dir)
            print(f"\n{'='*60}\n--- Processing Task: {task_name} ---\n{'='*60}")

            with open(task_json_path, "r") as f:
                task_json_all = json.load(f)
            
            # Inner loop: iterate over each trajectory (episode) within the task
            for traj_id_in_task, traj_data in enumerate(task_json_all):
                # Create a unique directory for this episode using a global index
                episode_dir = os.path.join(args.features_path, f"episode{global_episode_idx:07}")
                os.makedirs(episode_dir, exist_ok=True)

                instruction = traj_data["instruction"]
                # IMPORTANT: Check if your JSON has an 'action' field. If not, this will use 'features'.
                action_list = traj_data.get("action", traj_data.get("features"))
                if not action_list:
                    print(f"  [Warning] No 'action' or 'features' found for episode {global_episode_idx}. Skipping.")
                    continue

                # Process and save text embedding once per episode
                text_inputs = tokenizer([instruction], padding=True, return_tensors="pt").to(device)
                text_embeds = model(**text_inputs).text_embeds
                text_embed_path = os.path.join(episode_dir, "text_clip.npy")
                np.save(text_embed_path, text_embeds.cpu().numpy())

                # Get all image paths for this specific trajectory
                # The trajectory folder name in your data is `class_{index:06d}`
                image_folder_path = os.path.join(task_dir, f"class_{traj_id_in_task:06d}")
                episode_images = sorted(glob(os.path.join(image_folder_path, "*.png")))

                if not episode_images:
                    print(f"  [Warning] No images found in {image_folder_path} for episode {global_episode_idx}. Skipping.")
                    continue

                print(f"  Processing episode {global_episode_idx} (Task: {task_name}, Traj: {traj_id_in_task}): '{instruction}', {len(episode_images)} images")

                # Loop through each frame in the trajectory
                for frame_idx, img_path in enumerate(episode_images):
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = transform(img).unsqueeze(0).to(device)

                    # Encode image to latent representation
                    x = vae.encode(img_tensor).latent_dist.sample().mul_(0.18215)

                    # Save the latent vector
                    feature_path = os.path.join(episode_dir, f"color_wrist_1_{frame_idx:04}.npy")
                    np.save(feature_path, x.cpu().numpy())

                    # Get action and state data for the current frame
                    if frame_idx < len(action_list):
                        action = action_list[frame_idx]
                        state = action_list[frame_idx]  # Using action as state, adjust if you have separate state data
                    else:
                        action = [0, 0, 0, 1]  # Default value if action data is missing for a frame
                        state = [0, 0, 0, 1]

                    # Append frame-level information to the global dataset list
                    all_dataset_info.append({
                        "idx": str(global_train_steps),
                        "episode": str(global_episode_idx),
                        "frame": str(frame_idx),
                        "wrist_1": f'episode{global_episode_idx:07}/color_wrist_1_{frame_idx:04}.npy',
                        "label": task_name, # Use task name as label
                        "instruction": instruction,
                        "ins_emb_path": f'episode{global_episode_idx:07}/text_clip.npy',
                        "action": action,
                        "state": state
                    })

                    global_train_steps += 1

                # Increment the global episode counter after processing all frames of a trajectory
                global_episode_idx += 1

            if global_train_steps % 500 == 0 and global_train_steps > 0:
                print(f"... processed {global_train_steps} total frames so far ...")

    print(f"\nFeature extraction complete! Processed {global_train_steps} total frames from {global_episode_idx} episodes.")

    # Save the aggregated dataset metadata
    print("\nSaving final dataset metadata...")
    final_json_path = os.path.join(args.features_path, "dataset_rgb_s_d.json")
    with open(final_json_path, "w") as f:
        json.dump(all_dataset_info, f, indent=2)

    print(f"Dataset info saved to {final_json_path}")
    print(f"Total samples: {len(all_dataset_info)}")

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






"""
Complete single GPU version of extract_features.py for MetaWorld with action data.
"""
import torch

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


def update_force_stats(stats, force):
    force = np.asarray(force, dtype=np.float64)
    stats["count"] += 1
    delta = force - stats["mean"]
    stats["mean"] += delta / stats["count"]
    delta2 = force - stats["mean"]
    stats["m2"] += delta * delta2





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

    
    device_idx = 0  
    device = f"cuda:{device_idx}"
    torch.cuda.set_device(device)
    seed = args.global_seed
    torch.manual_seed(seed)
    print(f"Starting seed={seed}, device={device}.")

    
    os.makedirs(args.features_path, exist_ok=True)

    
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    vae = AutoencoderKL.from_pretrained("None/").to(device)
    
    from transformers import AutoTokenizer, CLIPTextModelWithProjection
    model = CLIPTextModelWithProjection.from_pretrained("None").to(device)
    tokenizer = AutoTokenizer.from_pretrained("None/")

    
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    
    task_json_paths = sorted(glob(os.path.join(args.data_path, "*/dataset_info.json")))
    if not task_json_paths:
        single_path = os.path.join(args.data_path, "dataset_info.json")
        if os.path.exists(single_path):
            task_json_paths = [single_path]
        else:
            raise ValueError(f"No 'dataset_info.json' found under {args.data_path}. Please check your data structure.")
    
    print(f"Found {len(task_json_paths)} tasks to process.")

    vae.eval()
    model.eval()

    
    global_train_steps = 0
    global_episode_idx = 0
    all_dataset_info = []
    force_stats = {"count": 0, "mean": np.zeros(6, dtype=np.float64), "m2": np.zeros(6, dtype=np.float64)}

    with torch.no_grad():
        
        for task_json_path in task_json_paths:
            task_dir = os.path.dirname(task_json_path)
            if os.path.abspath(task_dir) == os.path.abspath(args.data_path):
                task_name = "single_task"
            else:
                task_name = os.path.basename(task_dir)
            print(f"\n{'='*60}\n--- Processing Task: {task_name} ---\n{'='*60}")

            with open(task_json_path, "r") as f:
                task_json_all = json.load(f)
            
            
            for traj_id_in_task, traj_data in enumerate(task_json_all):
                
                episode_dir = os.path.join(args.features_path, f"episode{global_episode_idx:07}")
                os.makedirs(episode_dir, exist_ok=True)

                instruction = traj_data["instruction"]
                
                action_list = traj_data.get("action", traj_data.get("features"))
                force_list = traj_data.get("force_features", None)
                if not action_list:
                    print(f"  [Warning] No 'action' or 'features' found for episode {global_episode_idx}. Skipping.")
                    continue

                
                text_inputs = tokenizer([instruction], padding=True, return_tensors="pt").to(device)
                text_embeds = model(**text_inputs).text_embeds
                text_embed_path = os.path.join(episode_dir, "text_clip.npy")
                np.save(text_embed_path, text_embeds.cpu().numpy())

                
                
                image_folder_path = os.path.join(task_dir, f"class_{traj_id_in_task:06d}")
                episode_images = sorted(glob(os.path.join(image_folder_path, "*.png")))

                if not episode_images:
                    print(f"  [Warning] No images found in {image_folder_path} for episode {global_episode_idx}. Skipping.")
                    continue

                print(f"  Processing episode {global_episode_idx} (Task: {task_name}, Traj: {traj_id_in_task}): '{instruction}', {len(episode_images)} images")

                frame_count = len(episode_images)
                if force_list is not None and len(force_list) != frame_count:
                    print(f"  [Warning] force_features length ({len(force_list)}) != images ({frame_count}) for episode {global_episode_idx}")

                
                for frame_idx, img_path in enumerate(episode_images):
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = transform(img).unsqueeze(0).to(device)

                    
                    x = vae.encode(img_tensor).latent_dist.sample().mul_(0.18215)

                    
                    feature_path = os.path.join(episode_dir, f"color_wrist_1_{frame_idx:04}.npy")
                    np.save(feature_path, x.cpu().numpy())

                    
                    if frame_idx < len(action_list):
                        action = action_list[frame_idx]
                        state = action_list[frame_idx]  
                    else:
                        action = [0, 0, 0, 1]  
                        state = [0, 0, 0, 1]

                    if force_list is not None and frame_idx < len(force_list):
                        force = force_list[frame_idx]
                    else:
                        force = [0, 0, 0, 0, 0, 0]

                    update_force_stats(force_stats, force)

                    
                    all_dataset_info.append({
                        "idx": str(global_train_steps),
                        "episode": str(global_episode_idx),
                        "frame": str(frame_idx),
                        "wrist_1": f'episode{global_episode_idx:07}/color_wrist_1_{frame_idx:04}.npy',
                        "label": task_name, 
                        "instruction": instruction,
                        "ins_emb_path": f'episode{global_episode_idx:07}/text_clip.npy',
                        "action": action,
                        "state": state,
                        "force": force
                    })

                    global_train_steps += 1

                
                global_episode_idx += 1

            if global_train_steps % 500 == 0 and global_train_steps > 0:
                print(f"... processed {global_train_steps} total frames so far ...")

    print(f"\nFeature extraction complete! Processed {global_train_steps} total frames from {global_episode_idx} episodes.")

    
    print("\nSaving final dataset metadata...")
    final_json_path = os.path.join(args.features_path, "dataset_rgb_s_d.json")
    with open(final_json_path, "w") as f:
        json.dump(all_dataset_info, f, indent=2)

    print(f"Dataset info saved to {final_json_path}")
    print(f"Total samples: {len(all_dataset_info)}")

    if force_stats["count"] > 0:
        denom = max(force_stats["count"] - 1, 1)
        var = force_stats["m2"] / denom
        std = np.sqrt(var)
        stats_payload = {
            "count": int(force_stats["count"]),
            "mean": force_stats["mean"].tolist(),
            "std": std.tolist()
        }
        stats_path = os.path.join(args.features_path, "force_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats_payload, f, indent=2)
        print(f"Force stats saved to {stats_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--features-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--global-batch-size", type=int, default=1)  
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    main(args)

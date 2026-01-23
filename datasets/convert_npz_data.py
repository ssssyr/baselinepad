
"""
Convert npz format robot data to training format.

Source (multi task): /mnt/sda/datasets/real_data/
  - task_folder_1/       (folder name = instruction)
    - episode_XXXX.npz   (image, action, robot_pose, gripper_state, force_torque)
    - episode_XXXX_vis/  (visualization images, optional)
  - task_folder_2/
    - ...

NPZ file structure:
  - image: (n, H, W, 3)           - camera images
  - action: (n, 7)                - robot actions [x,y,z,qx,qy,qz,gripper]
  - robot_pose: (n, 6)            - robot poses [x,y,z,roll,pitch,yaw]
  - gripper_state: (n,)           - gripper states
  - force_torque: (n, 6)          - force/torque data
  - cam_ts_hw, cam_ts_mono, ...   - timestamps

Output: Compatible with RobotDataset
  - dataset_rgb_s_d.json
  - force_stats.json
  - episodeXXXXXXX/
    - color_wrist_1_XXXX.npy      - VAE latents
    - text_clip.npy               - CLIP text embedding

Usage:
    python convert_npz_data.py \
        --input /mnt/sda/datasets/real_data \
        --output /mnt/sda/datasets/real_data_converted \
        --multi-task
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms
from diffusers.models import AutoencoderKL
from transformers import AutoTokenizer, CLIPTextModelWithProjection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)




def resize_image(pil_image, image_size):
    """Resize image to the specified size (stretch/squash to fit, preserving all content)."""
    return pil_image.resize((image_size, image_size), resample=Image.BICUBIC)




class NPZDataReader:
    """Read robot data from npz files."""

    def __init__(self, task_dir: str):
        self.task_dir = task_dir
        self.npz_files = sorted([f for f in os.listdir(task_dir) if f.endswith('.npz')])
        self.num_episodes = len(self.npz_files)
        logger.info(f"Found {self.num_episodes} episodes in {task_dir}")

    def load_episode_data(self, episode_idx: int) -> Dict[str, np.ndarray]:
        """
        Load all data for a single episode.

        Args:
            episode_idx: Episode index (0-based)

        Returns dict with:
            - image: [n_steps, H, W, 3]
            - action: [n_steps, 7]    -> last column is gripper
            - robot_pose: [n_steps, 6]
            - gripper_state: [n_steps]
            - force_torque: [n_steps, 6]
        """
        npz_path = os.path.join(self.task_dir, self.npz_files[episode_idx])
        data = np.load(npz_path, allow_pickle=True)

        return {
            "image": data["image"],
            "action": data["action"],
            "robot_pose": data["robot_pose"],
            "gripper_state": data["gripper_state"],
            "force_torque": data.get("force_torque", np.zeros((data["action"].shape[0], 6))),
        }




class NPZDataConverter:
    """Convert npz format robot data to training format."""

    def __init__(self, args):
        self.args = args
        self.input_dir = args.input
        self.output_dir = args.output
        self.instruction = args.instruction
        self.image_size = args.image_size
        self.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
        self.multi_task = args.multi_task

        
        os.makedirs(self.output_dir, exist_ok=True)

        
        self.npz_reader = None
        self.current_task_dir = None

        
        self._setup_models()
        self._setup_transform()

        
        self.force_stats = {"count": 0, "mean": np.zeros(6, dtype=np.float64), "m2": np.zeros(6, dtype=np.float64)}

    def _setup_for_task(self, task_dir: str, instruction: str):
        """Setup components for a specific task."""
        self.current_task_dir = task_dir
        self.instruction = instruction
        self.npz_reader = NPZDataReader(task_dir)

        
        with torch.no_grad():
            text_inputs = self.clip_tokenizer([instruction], padding=True, return_tensors="pt").to(self.device)
            self.text_embed = self.clip_model(**text_inputs).text_embeds.cpu().numpy()

    def _setup_models(self):
        """Load VAE and CLIP models."""
        logger.info("Loading VAE model...")
        self.vae = AutoencoderKL.from_pretrained(self.args.vae_path).to(self.device)
        self.vae.eval()

        logger.info("Loading CLIP model...")
        self.clip_model = CLIPTextModelWithProjection.from_pretrained(self.args.clip_path).to(self.device)
        self.clip_tokenizer = AutoTokenizer.from_pretrained(self.args.clip_path)
        self.clip_model.eval()

    def _setup_transform(self):
        """Setup image preprocessing pipeline."""
        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: resize_image(img, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])

    def _update_force_stats(self, force):
        """Update running force statistics (Welford's algorithm)."""
        force = np.asarray(force, dtype=np.float64)
        self.force_stats["count"] += 1
        delta = force - self.force_stats["mean"]
        self.force_stats["mean"] += delta / self.force_stats["count"]
        delta2 = force - self.force_stats["mean"]
        self.force_stats["m2"] += force * delta2

    def _encode_image(self, image: np.ndarray) -> np.ndarray:
        """Encode image to latent using VAE."""
        img_pil = Image.fromarray(image)
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            latent = self.vae.encode(img_tensor).latent_dist.sample().mul_(0.18215)

        return latent.cpu().numpy()  

    def _get_task_dirs(self) -> List[Tuple[str, str]]:
        """Get list of (task_dir, instruction) tuples.

        In single-task mode: returns [(input_dir, instruction)]
        In multi-task mode: scans subfolders and uses folder names as instructions
        """
        if not self.multi_task:
            return [(self.input_dir, self.instruction)]

        
        tasks = []
        for item in os.listdir(self.input_dir):
            item_path = os.path.join(self.input_dir, item)
            if os.path.isdir(item_path):
                
                npz_files = [f for f in os.listdir(item_path) if f.endswith('.npz')]
                if npz_files:
                    
                    tasks.append((item_path, item))
                    logger.info(f"Found task: {item} -> {item_path} ({len(npz_files)} episodes)")

        if not tasks:
            raise ValueError(f"No valid task directories found in {self.input_dir}")

        return tasks

    def convert(self):
        """Main conversion loop."""
        all_dataset_info = []
        global_frame_idx = 0
        global_episode_idx = 0

        tasks = self._get_task_dirs()
        logger.info(f"Processing {len(tasks)} task(s)")

        with torch.no_grad():
            for task_idx, (task_dir, instruction) in enumerate(tasks):
                logger.info(f"{'='*60}")
                logger.info(f"Task {task_idx + 1}/{len(tasks)}: {instruction}")
                logger.info(f"  Directory: {task_dir}")
                logger.info(f"{'='*60}")

                
                self._setup_for_task(task_dir, instruction)

                for ep_idx in range(self.npz_reader.num_episodes):
                    episode_id = self.npz_reader.npz_files[ep_idx].replace('.npz', '')
                    logger.info(f"Processing episode {ep_idx+1}/{self.npz_reader.num_episodes}: {episode_id}")

                    
                    episode_data = self.npz_reader.load_episode_data(ep_idx)

                    n_steps = len(episode_data["image"])
                    logger.info(f"  Images: {n_steps} frames")

                    
                    episode_dir = os.path.join(self.output_dir, f"episode{global_episode_idx:07d}")
                    os.makedirs(episode_dir, exist_ok=True)

                    
                    text_embed_path = os.path.join(episode_dir, "text_clip.npy")
                    np.save(text_embed_path, self.text_embed)

                    
                    for step_idx in range(n_steps):
                        
                        pose = episode_data["robot_pose"][step_idx]
                        x, y, z = pose[0], pose[1], pose[2]
                        grip = int(episode_data["gripper_state"][step_idx])
                        state = [float(x), float(y), float(z), float(grip)]

                        
                        force = episode_data["force_torque"][step_idx].tolist()

                        
                        self._update_force_stats(force)

                        
                        frame_img = episode_data["image"][step_idx]

                        
                        latent = self._encode_image(frame_img)

                        
                        latent_path = os.path.join(episode_dir, f"color_wrist_1_{step_idx:04d}.npy")
                        np.save(latent_path, latent)

                        
                        all_dataset_info.append({
                            "episode": global_episode_idx,
                            "frame": step_idx,
                            "wrist_1": f"episode{global_episode_idx:07d}/color_wrist_1_{step_idx:04d}.npy",
                            "state": state,
                            "force": force,
                            "ins_emb_path": f"episode{global_episode_idx:07d}/text_clip.npy"
                        })

                        global_frame_idx += 1

                    logger.info(f"  Saved {n_steps} frames for episode {episode_id}")
                    global_episode_idx += 1

        
        json_path = os.path.join(self.output_dir, "dataset_rgb_s_d.json")
        with open(json_path, "w") as f:
            json.dump(all_dataset_info, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved dataset info to {json_path}")
        logger.info(f"Total samples: {len(all_dataset_info)}")

        
        if self.force_stats["count"] > 0:
            denom = max(self.force_stats["count"] - 1, 1)
            var = self.force_stats["m2"] / denom
            std = np.sqrt(var)
            stats_payload = {
                "count": int(self.force_stats["count"]),
                "mean": self.force_stats["mean"].tolist(),
                "std": std.tolist()
            }
            stats_path = os.path.join(self.output_dir, "force_stats.json")
            with open(stats_path, "w") as f:
                json.dump(stats_payload, f, indent=2)
            logger.info(f"Saved force stats to {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert npz format robot data to training format")

    
    parser.add_argument("--input", type=str, default="/mnt/sda/datasets/real_data",
                        help="Input directory (single task) or parent directory (multi-task)")
    parser.add_argument("--output", type=str, default="/mnt/sda/datasets/real_data_converted",
                        help="Output directory for converted data")

    
    parser.add_argument("--vae-path", type=str, default="/home/syr/code/models/sd-vae-ft-mse",
                        help="Path to VAE model")
    parser.add_argument("--clip-path", type=str, default="/home/syr/code/models/clip-vit-base-patch32",
                        help="Path to CLIP model")

    
    parser.add_argument("--multi-task", action="store_true",
                        help="Enable multi-task mode: scan subdirectories and use folder names as instructions")
    parser.add_argument("--instruction", type=str, default="",
                        help="Task instruction for text embedding (single-task mode only)")

    
    parser.add_argument("--image-size", type=int, default=256,
                        help="Image size after center crop")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID")

    args = parser.parse_args()

    
    converter = NPZDataConverter(args)
    converter.convert()

    logger.info("Conversion complete!")

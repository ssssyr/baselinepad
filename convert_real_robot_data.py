#!/usr/bin/env python3
"""
Convert real robot zarr data to training format.

Source: /mnt/sda/datasets/newdata/newdata/
  - replay_buffer.zarr/  (robot state data)
  - videos/              (camera videos)

Output: Compatible with RobotDataset
  - dataset_rgb_s_d.json
  - force_stats.json
  - episodeXXXXXXX/
    - color_wrist_1_XXXX.npy
    - text_clip.npy

Usage:
    python convert_real_robot_data.py \
        --input /mnt/sda/datasets/newdata/newdata \
        --output /mnt/sda/datasets/converted \
        --instruction "夹起魔方放到盘子里"
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms
from diffusers.models import AutoencoderKL
from transformers import AutoTokenizer, CLIPTextModelWithProjection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==================== Image Preprocessing ====================

def center_crop_arr(pil_image, image_size):
    """Center crops a PIL image to the specified size."""
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


# ==================== Zarr Data Reader ====================

class ZarrDataReader:
    """Read robot state data from zarr format (flat storage with episode_ends)."""

    def __init__(self, zarr_path: str):
        import zarr
        self.zarr = zarr.open(zarr_path, 'r')

        # Get episode boundaries
        self.episode_ends = self.zarr['meta']['episode_ends'][:]
        self.num_episodes = len(self.episode_ends)

        logger.info(f"Found {self.num_episodes} episodes in zarr data")

    def get_episode_indices(self, episode_idx: int) -> tuple:
        """Get (start_idx, end_idx) for a given episode."""
        start_idx = 0 if episode_idx == 0 else self.episode_ends[episode_idx - 1]
        end_idx = self.episode_ends[episode_idx]
        return start_idx, end_idx

    def load_episode_data(self, episode_idx: int) -> Dict[str, np.ndarray]:
        """
        Load all data for a single episode.

        Args:
            episode_idx: Episode index (0-based)

        Returns dict with:
            - timestamp: [n_steps]
            - robot_eef_pose: [n_steps, 6]  -> use [:3] for x,y,z
            - gripper_target: [n_steps]     -> grip state (0/1)
            - gripper_force: [n_steps, 6]   -> force [fx,fy,fz,tx,ty,tz]
        """
        start_idx, end_idx = self.get_episode_indices(episode_idx)

        data = {
            "timestamp": self.zarr['data']['timestamp'][start_idx:end_idx],
            "robot_eef_pose": self.zarr['data']['robot_eef_pose'][start_idx:end_idx],
            "gripper_target": self.zarr['data']['gripper_target'][start_idx:end_idx],
            "gripper_force": self.zarr['data']['gripper_force'][start_idx:end_idx],
        }

        # Validate data
        n_steps = len(data["timestamp"])
        for key, val in data.items():
            if len(val) != n_steps:
                logger.warning(f"Episode {episode_idx}: {key} length mismatch")

        return data


# ==================== Video Frame Extractor ====================

class VideoFrameExtractor:
    """Extract frames from video with timestamp alignment."""

    def __init__(self, video_dir: str, target_fps: float = 10.0):
        self.video_dir = video_dir
        self.target_fps = target_fps

    def get_video_info(self, episode_id: str) -> Dict[str, Any]:
        """Get video metadata: fps, frame_count, duration."""
        video_path = os.path.join(self.video_dir, episode_id, "0.mp4")
        if not os.path.exists(video_path):
            return None

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()

        return {
            "path": video_path,
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration": duration
        }

    def extract_frame_at_time(self, video_path: str, time_sec: float) -> np.ndarray:
        """Extract frame closest to given time (seconds from start)."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        target_frame = int(time_sec * fps)
        target_frame = max(0, min(target_frame, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1))

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None

        # Convert BGR to RGB
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ==================== Main Converter ====================

class RealRobotDataConverter:
    """Convert real robot zarr + video data to training format."""

    def __init__(self, args):
        self.args = args
        self.input_dir = args.input
        self.output_dir = args.output
        self.instruction = args.instruction
        self.image_size = args.image_size
        self.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

        # Setup output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize components
        self.zarr_reader = ZarrDataReader(os.path.join(self.input_dir, "replay_buffer.zarr"))
        self.video_extractor = VideoFrameExtractor(os.path.join(self.input_dir, "videos"))

        # Setup models
        self._setup_models()
        self._setup_transform()

        # Force stats
        self.force_stats = {"count": 0, "mean": np.zeros(6, dtype=np.float64), "m2": np.zeros(6, dtype=np.float64)}

    def _setup_models(self):
        """Load VAE and CLIP models."""
        logger.info("Loading VAE model...")
        self.vae = AutoencoderKL.from_pretrained(args.vae_path).to(self.device)
        self.vae.eval()

        logger.info("Loading CLIP model...")
        self.clip_model = CLIPTextModelWithProjection.from_pretrained(args.clip_path).to(self.device)
        self.clip_tokenizer = AutoTokenizer.from_pretrained(args.clip_path)
        self.clip_model.eval()

        # Encode instruction once
        with torch.no_grad():
            text_inputs = self.clip_tokenizer([self.instruction], padding=True, return_tensors="pt").to(self.device)
            self.text_embed = self.clip_model(**text_inputs).text_embeds.cpu().numpy()  # (1, 512)

    def _setup_transform(self):
        """Setup image preprocessing pipeline."""
        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: center_crop_arr(img, self.image_size)),
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
        self.force_stats["m2"] += delta * delta2

    def _encode_image(self, image: np.ndarray) -> np.ndarray:
        """Encode image to latent using VAE."""
        img_pil = Image.fromarray(image)
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            latent = self.vae.encode(img_tensor).latent_dist.sample().mul_(0.18215)

        return latent.cpu().numpy()  # (1, 4, 32, 32)

    def convert(self):
        """Main conversion loop."""
        all_dataset_info = []
        global_frame_idx = 0

        with torch.no_grad():
            for ep_idx in range(self.zarr_reader.num_episodes):
                episode_id = str(ep_idx)  # Video directory uses string ID
                logger.info(f"Processing episode {ep_idx+1}/{self.zarr_reader.num_episodes}: {episode_id}")

                # Load zarr data
                episode_data = self.zarr_reader.load_episode_data(ep_idx)

                if episode_data["timestamp"] is None or len(episode_data["timestamp"]) == 0:
                    logger.warning(f"Episode {ep_idx}: No timestamp data, skipping")
                    continue

                # Get video info
                video_info = self.video_extractor.get_video_info(episode_id)
                if video_info is None:
                    logger.warning(f"Episode {ep_idx}: No video found, skipping")
                    continue

                logger.info(f"  Video: {video_info['frame_count']} frames, {video_info['fps']} fps, {video_info['duration']:.1f}s")
                logger.info(f"  Zarr: {len(episode_data['timestamp'])} steps")

                # Create episode directory
                episode_dir = os.path.join(self.output_dir, f"episode{ep_idx:07d}")
                os.makedirs(episode_dir, exist_ok=True)

                # Save text embedding (once per episode)
                text_embed_path = os.path.join(episode_dir, "text_clip.npy")
                np.save(text_embed_path, self.text_embed)

                # Get video start time (first timestamp)
                video_start_time = episode_data["timestamp"][0]

                # Process each time step
                n_steps = len(episode_data["timestamp"])
                for step_idx in range(n_steps):
                    # Extract state
                    pose = episode_data["robot_eef_pose"][step_idx]
                    x, y, z = pose[0], pose[1], pose[2]
                    grip = int(episode_data["gripper_target"][step_idx]) if episode_data["gripper_target"] is not None else 0
                    state = [float(x), float(y), float(z), float(grip)]

                    # Extract force
                    if episode_data["gripper_force"] is not None:
                        force = episode_data["gripper_force"][step_idx].tolist()
                    else:
                        force = [0.0] * 6

                    # Update force stats
                    self._update_force_stats(force)

                    # Get corresponding video frame
                    timestamp = episode_data["timestamp"][step_idx]
                    time_from_start = timestamp - video_start_time
                    frame_img = self.video_extractor.extract_frame_at_time(video_info["path"], time_from_start)

                    if frame_img is None:
                        logger.warning(f"  Step {step_idx}: Failed to extract frame at t={time_from_start:.2f}s")
                        continue

                    # Encode image
                    latent = self._encode_image(frame_img)

                    # Save latent
                    latent_path = os.path.join(episode_dir, f"color_wrist_1_{step_idx:04d}.npy")
                    np.save(latent_path, latent)

                    # Add to dataset info
                    all_dataset_info.append({
                        "episode": ep_idx,
                        "frame": step_idx,
                        "wrist_1": f"episode{ep_idx:07d}/color_wrist_1_{step_idx:04d}.npy",
                        "state": state,
                        "force": force,
                        "ins_emb_path": f"episode{ep_idx:07d}/text_clip.npy"
                    })

                    global_frame_idx += 1

                logger.info(f"  Saved {n_steps} frames for episode {episode_id}")

        # Save dataset info
        json_path = os.path.join(self.output_dir, "dataset_rgb_s_d.json")
        with open(json_path, "w") as f:
            json.dump(all_dataset_info, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved dataset info to {json_path}")
        logger.info(f"Total samples: {len(all_dataset_info)}")

        # Save force stats
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
    parser = argparse.ArgumentParser(description="Convert real robot zarr data to training format")

    # Input/Output
    parser.add_argument("--input", type=str, default="/mnt/sda/datasets/newdata/newdata",
                        help="Input directory containing replay_buffer.zarr and videos/")
    parser.add_argument("--output", type=str, default="/mnt/sda/datasets/converted",
                        help="Output directory for converted data")

    # Model paths
    parser.add_argument("--vae-path", type=str, default="/home/syr/code/models/sd-vae-ft-mse",
                        help="Path to VAE model")
    parser.add_argument("--clip-path", type=str, default="/home/syr/code/models/clip-vit-base-patch32",
                        help="Path to CLIP model")

    # Task description
    parser.add_argument("--instruction", type=str, default="夹起魔方放到盘子里",
                        help="Task instruction for text embedding")

    # Processing parameters
    parser.add_argument("--image-size", type=int, default=256,
                        help="Image size after center crop")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID")

    args = parser.parse_args()

    # Run conversion
    converter = RealRobotDataConverter(args)
    converter.convert()

    logger.info("Conversion complete!")

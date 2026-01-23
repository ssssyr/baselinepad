
"""
Convert real robot zarr data to training format.

Source (single task): None/
  - replay_buffer.zarr/  (robot state data)
  - videos/              (camera videos)

Source (multi task): None/
  - task_folder_1/       (folder name = instruction)
    - replay_buffer.zarr/
    - videos/
  - task_folder_2/
    - replay_buffer.zarr/
    - videos/

Output: Compatible with RobotDataset
  - dataset_rgb_s_d.json
  - force_stats.json
  - episodeXXXXXXX/
    - color_wrist_1_XXXX.npy
    - text_clip.npy

Usage (single task):
    python convert_real_robot_data.py \
        --input None/task_folder \
        --output None

Usage (multi task):
    python convert_real_robot_data.py \
        --input None \
        --output None \
        --multi-task
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




class ZarrDataReader:
    """Read robot state data from zarr format (flat storage with episode_ends)."""

    def __init__(self, zarr_path: str):
        import zarr
        self.zarr = zarr.open(zarr_path, 'r')

        
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

        
        n_steps = len(data["timestamp"])
        for key, val in data.items():
            if len(val) != n_steps:
                logger.warning(f"Episode {episode_idx}: {key} length mismatch")

        return data




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

        
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)




class RealRobotDataConverter:
    """Convert real robot zarr + video data to training format."""

    def __init__(self, args):
        self.args = args
        self.input_dir = args.input
        self.output_dir = args.output
        self.instruction = args.instruction
        self.image_size = args.image_size
        self.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
        self.multi_task = args.multi_task

        
        os.makedirs(self.output_dir, exist_ok=True)

        
        self.zarr_reader = None
        self.video_extractor = None
        self.current_task_dir = None

        
        self._setup_models()
        self._setup_transform()

        
        self.force_stats = {"count": 0, "mean": np.zeros(6, dtype=np.float64), "m2": np.zeros(6, dtype=np.float64)}

    def _setup_for_task(self, task_dir: str, instruction: str):
        """Setup components for a specific task."""
        self.current_task_dir = task_dir
        self.instruction = instruction
        self.zarr_reader = ZarrDataReader(os.path.join(task_dir, "replay_buffer.zarr"))
        self.video_extractor = VideoFrameExtractor(os.path.join(task_dir, "videos"))

        
        with torch.no_grad():
            text_inputs = self.clip_tokenizer([self.instruction], padding=True, return_tensors="pt").to(self.device)
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

        
        if not self.multi_task:
            with torch.no_grad():
                text_inputs = self.clip_tokenizer([self.instruction], padding=True, return_tensors="pt").to(self.device)
                self.text_embed = self.clip_model(**text_inputs).text_embeds.cpu().numpy()  

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

        return latent.cpu().numpy()  

    def _get_task_dirs(self) -> List[tuple]:
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
                
                zarr_path = os.path.join(item_path, "replay_buffer.zarr")
                videos_path = os.path.join(item_path, "videos")
                if os.path.exists(zarr_path) and os.path.exists(videos_path):
                    
                    tasks.append((item_path, item))
                    logger.info(f"Found task: {item} -> {item_path}")

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

                for ep_idx in range(self.zarr_reader.num_episodes):
                    episode_id = str(ep_idx)  
                    logger.info(f"Processing episode {ep_idx+1}/{self.zarr_reader.num_episodes}: {episode_id}")

                    
                    episode_data = self.zarr_reader.load_episode_data(ep_idx)

                    if episode_data["timestamp"] is None or len(episode_data["timestamp"]) == 0:
                        logger.warning(f"Episode {ep_idx}: No timestamp data, skipping")
                        continue

                    
                    video_info = self.video_extractor.get_video_info(episode_id)
                    if video_info is None:
                        logger.warning(f"Episode {ep_idx}: No video found, skipping")
                        continue

                    logger.info(f"  Video: {video_info['frame_count']} frames, {video_info['fps']} fps, {video_info['duration']:.1f}s")
                    logger.info(f"  Zarr: {len(episode_data['timestamp'])} steps")

                    
                    episode_dir = os.path.join(self.output_dir, f"episode{global_episode_idx:07d}")
                    os.makedirs(episode_dir, exist_ok=True)

                    
                    text_embed_path = os.path.join(episode_dir, "text_clip.npy")
                    np.save(text_embed_path, self.text_embed)

                    
                    video_start_time = episode_data["timestamp"][0]

                    
                    n_steps = len(episode_data["timestamp"])
                    for step_idx in range(n_steps):
                        
                        pose = episode_data["robot_eef_pose"][step_idx]
                        x, y, z = pose[0], pose[1], pose[2]
                        grip = int(episode_data["gripper_target"][step_idx]) if episode_data["gripper_target"] is not None else 0
                        state = [float(x), float(y), float(z), float(grip)]

                        
                        if episode_data["gripper_force"] is not None:
                            force = episode_data["gripper_force"][step_idx].tolist()
                        else:
                            force = [0.0] * 6

                        
                        self._update_force_stats(force)

                        
                        timestamp = episode_data["timestamp"][step_idx]
                        time_from_start = timestamp - video_start_time
                        frame_img = self.video_extractor.extract_frame_at_time(video_info["path"], time_from_start)

                        if frame_img is None:
                            logger.warning(f"  Step {step_idx}: Failed to extract frame at t={time_from_start:.2f}s")
                            continue

                        
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
    parser = argparse.ArgumentParser(description="Convert real robot zarr data to training format")

    
    parser.add_argument("--input", type=str, default="None",
                        help="Input directory (single task) or parent directory (multi-task)")
    parser.add_argument("--output", type=str, default="None",
                        help="Output directory for converted data")

    
    parser.add_argument("--vae-path", type=str, default="None",
                        help="Path to VAE model")
    parser.add_argument("--clip-path", type=str, default="None",
                        help="Path to CLIP model")

    
    parser.add_argument("--multi-task", action="store_true",
                        help="Enable multi-task mode: scan subdirectories and use folder names as instructions")
    parser.add_argument("--instruction", type=str, default="夹起魔方放到盘子里",
                        help="Task instruction for text embedding (single-task mode only)")

    
    parser.add_argument("--image-size", type=int, default=256,
                        help="Image size after center crop")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID")

    args = parser.parse_args()

    
    converter = RealRobotDataConverter(args)
    converter.convert()

    logger.info("Conversion complete!")

#!/usr/bin/env python3
"""
process_real_robot_data.py

处理 UR10 真实机器人收集的数据，转换为训练所需的格式。

输入: /home/syr/code/baselinepad/real/scripts/robot_data/episode_*.npz
输出: 与 extract_features_complete.py 相同的格式

主要处理步骤:
1. 从 NPZ 文件加载图像、机器人状态、力矩数据
2. 将图像从 1280x720 用 Letterbox 方法处理到 256x256（保留所有信息，不裁剪）
3. 使用 VAE 编码图像为潜在表示 (1, 4, 32, 32)
4. 使用 CLIP 编码任务指令为文本嵌入 (1, 512)
5. 转换 7 DOF 动作为 4 DOF (x, y, z, gripper)
6. 保存为 dataset_rgb_s_d.json 和 force_stats.json

图像处理说明:
- 使用 Letterbox 方法：等比例缩放 + 黑边填充
- 原始 1280x720 → 缩放到 256x144 → 上下填充黑边到 256x256
- 保留所有图像信息，保持宽高比不变形
"""

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import numpy as np
from PIL import Image
import argparse
import os
import json
from pathlib import Path
from tqdm import tqdm

from diffusers.models import AutoencoderKL
from transformers import AutoTokenizer, CLIPTextModelWithProjection

# ==============================================================================
# 配置
# ==============================================================================

INPUT_DATA_DIR = "/home/syr/code/baselinepad/real/scripts/robot_data"
OUTPUT_FEATURE_DIR = "/home/syr/code/baselinepad/datasets/processed_real_robot"

VAE_PATH = "/home/syr/code/models/sd-vae-ft-mse"
CLIP_PATH = "/home/syr/code/models/clip-vit-base-patch32"

# 任务配置（单一任务模式）
TASK_NAME = "ur10_robot_task"
TASK_INSTRUCTION = "Pick up the Rubik's Cube and place it on the plate."  # 夹起魔方放到盘子里

# 图像处理
IMAGE_SIZE = 256

# ==============================================================================
# 辅助函数
# ==============================================================================

def update_force_stats(stats, force):
    """使用 Welford 算法在线计算力矩数据的均值和标准差"""
    force = np.asarray(force, dtype=np.float64)
    stats["count"] += 1
    delta = force - stats["mean"]
    stats["mean"] += delta / stats["count"]
    delta2 = force - stats["mean"]
    stats["m2"] += delta * delta2


def letterbox_resize(pil_image, target_size):
    """
    Letterbox 调整图像大小：保持宽高比，用黑边填充

    保留所有图像信息，不裁剪任何部分。

    Args:
        pil_image: PIL Image
        target_size: 目标尺寸（正方形）

    Returns:
        PIL Image: 处理后的图像 (target_size x target_size)
    """
    original_w, original_h = pil_image.size

    # 计算缩放比例（使长边等于目标尺寸）
    scale = target_size / max(original_w, original_h)

    # 等比例缩放
    new_w = int(round(original_w * scale))
    new_h = int(round(original_h * scale))
    resized = pil_image.resize((new_w, new_h), resample=Image.BICUBIC)

    # 创建黑色背景
    result = Image.new("RGB", (target_size, target_size), (0, 0, 0))

    # 计算粘贴位置（居中）
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2

    # 粘贴缩放后的图像
    result.paste(resized, (paste_x, paste_y))

    return result


# ==============================================================================
# 主处理函数
# ==============================================================================

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="处理 UR10 真实机器人数据")
    parser.add_argument("--input-dir", type=str, default=INPUT_DATA_DIR, help="输入数据目录")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_FEATURE_DIR, help="输出特征目录")
    parser.add_argument("--instruction", type=str, default=TASK_INSTRUCTION, help="任务指令")
    parser.add_argument("--task-name", type=str, default=TASK_NAME, help="任务名称")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE, help="输出图像大小")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA 设备")
    parser.add_argument("--skip-encoding", action="store_true", help="跳过 VAE/CLIP 编码（仅调试）")
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 检查 CUDA
    if not args.skip_encoding and not torch.cuda.is_available():
        print("警告: CUDA 不可用，将使用 CPU（速度较慢）")

    # 设置设备
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 初始化模型
    if not args.skip_encoding:
        print(f"加载 VAE 模型: {VAE_PATH}")
        vae = AutoencoderKL.from_pretrained(VAE_PATH).to(device)
        vae.eval()

        print(f"加载 CLIP 模型: {CLIP_PATH}")
        text_model = CLIPTextModelWithProjection.from_pretrained(CLIP_PATH).to(device)
        tokenizer = AutoTokenizer.from_pretrained(CLIP_PATH)
        text_model.eval()

        # 图像变换（使用 Letterbox 方法）
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: letterbox_resize(pil_image, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])

        # 获取任务指令的 CLIP 嵌入（所有 episodes 共享）
        print(f"任务指令: {args.instruction}")
        with torch.no_grad():
            text_inputs = tokenizer([args.instruction], padding=True, return_tensors="pt").to(device)
            text_embeds = text_model(**text_inputs).text_embeds
            text_embedding = text_embeds.cpu().numpy()  # (1, 512)
        print(f"文本嵌入形状: {text_embedding.shape}")
    else:
        vae = None
        text_model = None
        tokenizer = None
        transform = None
        text_embedding = None
        print("警告: 跳过编码模式，仅处理数据结构")

    # 查找所有 episode 文件
    input_dir = Path(args.input_dir)
    episode_files = sorted(input_dir.glob("episode_*.npz"))
    metadata_files = sorted(input_dir.glob("episode_*_metadata.json"))

    if not episode_files:
        raise ValueError(f"在 {input_dir} 中未找到 episode_*.npz 文件")

    print(f"\n找到 {len(episode_files)} 个 episode 文件")

    # 全局计数器和列表
    global_episode_idx = 0
    global_frame_idx = 0
    all_dataset_info = []
    force_stats = {"count": 0, "mean": np.zeros(6, dtype=np.float64), "m2": np.zeros(6, dtype=np.float64)}

    # 处理每个 episode
    with torch.no_grad():
        for episode_file in tqdm(episode_files, desc="处理 episodes"):
            # 从文件名提取 episode 索引
            episode_idx = int(episode_file.stem.split("_")[1])

            # 加载 episode 数据
            try:
                episode_data = np.load(episode_file)
            except Exception as e:
                print(f"\n警告: 无法加载 {episode_file.name}: {e}")
                continue

            # 提取数据
            images = episode_data["image"]  # (T, 720, 1280, 3)
            robot_pose = episode_data["robot_pose"]  # (T, 6) - [x, y, z, rx, ry, rz]
            gripper_state = episode_data["gripper_state"]  # (T,)
            raw_action = episode_data["action"]  # (T, 7) - [vx, vy, vz, wx, wy, wz, gripper_cmd]
            force_torque = episode_data.get("force_torque", np.zeros((len(images), 6)))  # (T, 6)

            num_frames = len(images)
            print(f"\n  Episode {episode_idx}: {num_frames} 帧")

            # 创建 episode 目录
            episode_dir = os.path.join(args.output_dir, f"episode{global_episode_idx:07d}")
            os.makedirs(episode_dir, exist_ok=True)

            # 保存文本嵌入（所有 episodes 共享同一指令）
            if text_embedding is not None:
                text_embed_path = os.path.join(episode_dir, "text_clip.npy")
                np.save(text_embed_path, text_embedding)

            # 处理每一帧
            for frame_idx in range(num_frames):
                # 1. 处理图像
                img = images[frame_idx]
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)

                # 转换为 PIL Image
                pil_image = Image.fromarray(img)

                if not args.skip_encoding:
                    # 应用变换并编码
                    img_tensor = transform(pil_image).unsqueeze(0).to(device)
                    x = vae.encode(img_tensor).latent_dist.sample().mul_(0.18215)
                    latent = x.cpu().numpy()  # (1, 4, 32, 32)
                else:
                    # 调试模式: 创建虚拟 latent
                    latent = np.zeros((1, 4, 32, 32), dtype=np.float32)

                # 保存 latent
                latent_path = os.path.join(episode_dir, f"color_wrist_1_{frame_idx:04}.npy")
                np.save(latent_path, latent)

                # 2. 提取 4 DOF: [x, y, z, gripper]
                # gripper 从 raw_action[:, 6] 获取（命令值）
                action = [
                    float(robot_pose[frame_idx, 0]),  # x
                    float(robot_pose[frame_idx, 1]),  # y
                    float(robot_pose[frame_idx, 2]),  # z
                    float(raw_action[frame_idx, 6])    # gripper 命令
                ]
                state = action.copy()  # state 与 action 相同

                # 3. 提取力矩数据 (6 DOF)
                force = force_torque[frame_idx].tolist()
                update_force_stats(force_stats, force)

                # 4. 添加到 dataset 信息
                all_dataset_info.append({
                    "idx": str(global_frame_idx),
                    "episode": f"episode{global_episode_idx:07d}",
                    "frame": str(frame_idx),
                    "wrist_1": f"episode{global_episode_idx:07d}/color_wrist_1_{frame_idx:04}.npy",
                    "label": args.task_name,
                    "instruction": args.instruction,
                    "ins_emb_path": f"episode{global_episode_idx:07d}/text_clip.npy",
                    "action": action,
                    "state": state,
                    "force": force
                })

                global_frame_idx += 1

            global_episode_idx += 1

            # 每 500 帧打印进度
            if global_frame_idx % 500 == 0 and global_frame_idx > 0:
                print(f"  ... 已处理 {global_frame_idx} 帧 ...")

    # ========================================================================
    # 保存最终结果
    # ========================================================================

    print(f"\n处理完成! 共处理 {global_frame_idx} 帧，来自 {global_episode_idx} 个 episodes")

    # 保存 dataset metadata
    print("\n保存数据集 metadata...")
    final_json_path = os.path.join(args.output_dir, "dataset_rgb_s_d.json")
    with open(final_json_path, "w") as f:
        json.dump(all_dataset_info, f, indent=2)
    print(f"Dataset 信息已保存到: {final_json_path}")
    print(f"总样本数: {len(all_dataset_info)}")

    # 保存力矩统计信息
    if force_stats["count"] > 0:
        denom = max(force_stats["count"] - 1, 1)
        var = force_stats["m2"] / denom
        std = np.sqrt(var)
        stats_payload = {
            "count": int(force_stats["count"]),
            "mean": force_stats["mean"].tolist(),
            "std": std.tolist()
        }
        stats_path = os.path.join(args.output_dir, "force_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats_payload, f, indent=2)
        print(f"力矩统计已保存到: {stats_path}")
        print(f"  力矩均值: {stats_payload['mean']}")
        print(f"  力矩标准差: {stats_payload['std']}")

    print("\n全部完成!")


if __name__ == "__main__":
    main()

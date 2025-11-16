# -*- coding: utf-8 -*-
"""
Meta-World v2 数据采集脚本（最小化：不保存 action）
- 每个任务采集 N 条轨迹
- 固定相机 'corner2'，离屏渲染 PNG 帧
- 每条轨迹的 JSON 条目仅包含：
    {
      "instruction": <str>,
      "features": [[x,y,z,grip], ...],   # 世界坐标下的绝对状态
      "success": 0/1
    }
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

# 让 mujoco-py 离屏渲染
os.environ.setdefault("MUJOCO_GL", "egl")

# Meta-World v2 环境与策略
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from metaworld.policies import (
    SawyerAssemblyV2Policy,
    SawyerBasketballV2Policy,
    SawyerBinPickingV2Policy,
    SawyerBoxCloseV2Policy,
    SawyerButtonPressTopdownV2Policy,
    SawyerButtonPressV2Policy,
    SawyerCoffeeButtonV2Policy,
    SawyerCoffeePullV2Policy,
    SawyerCoffeePushV2Policy,
    SawyerDialTurnV2Policy,
)

# ---------------- 配 置 ---------------- #
TASKS_TO_COLLECT: Dict[str, str] = {
    "button-press-v2": "press the button",
    # 需要更多任务就取消注释/添加：
    # "assembly-v2": "assemble the peg",
    # "basketball-v2": "shoot the basketball into the hoop",
    # ...
}

NUM_TRAJECTORIES_PER_TASK = 50          # 每任务采集轨迹数
KEEP_ONLY_SUCCESS = False               # 仅保留成功轨迹（会重采直到够数）
CAMERA_NAME = "corner2"                 # ['corner','corner2','corner3','corner4','topview','behindGripper','gripperPOV']
IMAGE_RESOLUTION = (256, 256)           # (H, W)
OUTPUT_DIR = Path("/mnt/sda/datasets/metaworld")  # 输出根目录

# 任务名 -> 专家策略
POLICY_MAPPING = {
    "assembly-v2": SawyerAssemblyV2Policy,
    "basketball-v2": SawyerBasketballV2Policy,
    "bin-picking-v2": SawyerBinPickingV2Policy,
    "box-close-v2": SawyerBoxCloseV2Policy,
    "button-press-topdown-v2": SawyerButtonPressTopdownV2Policy,
    "button-press-v2": SawyerButtonPressV2Policy,
    "coffee-button-v2": SawyerCoffeeButtonV2Policy,
    "coffee-pull-v2": SawyerCoffeePullV2Policy,
    "coffee-push-v2": SawyerCoffeePushV2Policy,
    "dial-turn-v2": SawyerDialTurnV2Policy,
}
# ------------------------------------- #

def render_rgb(env, h: int, w: int, camera: str) -> np.ndarray:
    """离屏渲染一帧 RGB（HxWx3, uint8）"""
    frame = env.sim.render(width=w, height=h, camera_name=camera)
    return frame  # 如遇画面倒置，可改：return frame[::-1]

def collect_one_trajectory(
    env,
    policy,
    traj_idx: int,
    task_dir: Path,
    image_resolution: Tuple[int, int],
    camera_name: str,
) -> Tuple[bool, int, List[List[float]]]:
    """
    采集一条轨迹：保存 PNG 帧，并返回 (success, num_steps, states)
      - states: 每步 4 维绝对状态 [x,y,z,grip]（从 obs[:4] 取）
    目录结构：task_dir / f"class_{traj_idx:06d}"/ frame_0000.png, ...
    """
    traj_dir = task_dir / f"class_{traj_idx:06d}"
    traj_dir.mkdir(parents=True, exist_ok=True)

    env.seed(traj_idx)
    obs = env.reset()

    H, W = image_resolution
    max_len = int(getattr(env, "max_path_length", 500))

    states: List[List[float]] = []
    info_last = {}
    for step_idx in range(max_len):
        # 图像
        img = render_rgb(env, H, W, camera_name)
        Image.fromarray(img).save(traj_dir / f"frame_{step_idx:04d}.png")
        # 绝对状态（世界系）
        states.append((obs[:4]).tolist())

        # 与环境交互：仍然用专家动作，但不保存动作
        action = policy.get_action(obs)
        obs, _, done, info = env.step(np.asarray(action, dtype=np.float32))
        info_last = info
        if int(info.get("success", 0)) == 1 or done:
            break

    return (int(info_last.get("success", 0)) == 1), len(states), states

def main():
    print("=== Meta-World v2 采集（最小化：不保存 action） ===")
    print(f"输出目录: {OUTPUT_DIR.resolve()}")
    print(f"相机: {CAMERA_NAME}, 分辨率: {IMAGE_RESOLUTION}, 每任务轨迹数: {NUM_TRAJECTORIES_PER_TASK}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for task_name, instruction in TASKS_TO_COLLECT.items():
        env_key = task_name + "-goal-observable"
        if task_name not in POLICY_MAPPING or env_key not in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE:
            print(f"⚠️ 跳过 {task_name}: 未找到专家或环境类（检查 v2 安装）")
            continue

        policy = POLICY_MAPPING[task_name]()
        env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_key]
        env = env_cls()

        task_dir = OUTPUT_DIR / task_name
        task_dir.mkdir(parents=True, exist_ok=True)

        dataset_info: List[dict] = []
        saved, attempt = 0, 0
        pbar = tqdm(total=NUM_TRAJECTORIES_PER_TASK, desc=f"Collecting {task_name}", ncols=100)
        try:
            while saved < NUM_TRAJECTORIES_PER_TASK:
                success, nsteps, states = collect_one_trajectory(
                    env=env,
                    policy=policy,
                    traj_idx=attempt,
                    task_dir=task_dir,
                    image_resolution=IMAGE_RESOLUTION,
                    camera_name=CAMERA_NAME,
                )

                # 构造仅含所需键的条目
                traj_entry = {
                    "instruction": instruction,
                    "features": states,     # [[x,y,z,grip], ...] 世界坐标
                    "success": int(success)
                }

                if KEEP_ONLY_SUCCESS and not success:
                    # 不保留失败轨迹：清理刚才写的图片目录
                    traj_dir = task_dir / f"class_{attempt:06d}"
                    for png in traj_dir.glob("*.png"):
                        try:
                            png.unlink()
                        except FileNotFoundError:
                            pass
                    try:
                        traj_dir.rmdir()
                    except OSError:
                        # 目录可能非空，忽略
                        pass
                    attempt += 1
                    continue

                dataset_info.append(traj_entry)
                saved += 1
                attempt += 1
                pbar.update(1)
        finally:
            pbar.close()
            env.close()

        with open(task_dir / "dataset_info.json", "w") as f:
            json.dump(dataset_info, f, indent=2)

        print(f"✅ {task_name}: 保存 {saved} 条轨迹到 {task_dir}")

    print("\n=== 全部任务完成 ===")

if __name__ == "__main__":
    main()

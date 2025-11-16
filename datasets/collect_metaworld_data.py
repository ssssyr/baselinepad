# -*- coding: utf-8 -*-
"""
Meta-World v2 数据采集脚本（按任务各收集 50 条轨迹）
- 固定 camera_name='corner2' 离屏渲染（MUJOCO_GL=egl）
- 逐帧保存视频（mp4）
- JSON 记录动作与 4 维末端姿态 [ee_x, ee_y, ee_z, gripper]
"""

import os
import json
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm
import mediapy as media

# 1) 先设置 EGL 以便 mujoco-py 离屏渲染
os.environ.setdefault("MUJOCO_GL", "egl")

# 2) v2 接口：从 metaworld v2 导入环境字典与专家策略
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
TASKS_TO_COLLECT = [
    "button-press-v2",  # 简单的按钮按压任务，适合测试
    # "assembly-v2",
    # "basketball-v2",
    # "bin-picking-v2",
    # "box-close-v2",
    # "button-press-topdown-v2",
    # "coffee-button-v2",
    # "coffee-pull-v2",
    # "coffee-push-v2",
    # "dial-turn-v2",
]

NUM_TRAJECTORIES_PER_TASK = 50
CAMERA_NAME = "corner2"
IMAGE_RESOLUTION = (224, 224)  # (H, W)
FPS = 20
OUTPUT_DIR = Path("/mnt/sda/datasets/metaworld")

# 仅保存成功轨迹？（成功由 info["success"]==1 判定）
KEEP_ONLY_SUCCESS = False  # 设为 True 则会重采直到凑够 50 条成功轨迹

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


def render_rgb(env, height, width, camera_name):
    """用 mujoco-py 的 sim.render() 从指定相机取帧（RGB, HxWx3, uint8）"""
    # mujoco-py 的渲染使用 (width, height) 形参
    frame = env.sim.render(width=width, height=height, camera_name=camera_name)
    # OpenGL 读出的帧通常已是正确朝向；若你看到上下颠倒，可取消下一行翻转：
    # frame = frame[::-1, :, :]
    return frame


def collect_one_trajectory(env, policy, traj_idx, task_dir):
    """在已有 env 上采集一条轨迹，返回是否成功（用于只保成功的场景）"""
    # 设定种子保证多样性 & 可复现
    env.seed(traj_idx)
    obs = env.reset()

    frames, actions, robot_states = [], [], []

    # 步数兜底：有些版本没暴露 max_path_length；默认取 500
    max_len = int(getattr(env, "max_path_length", 500))

    info_last = {}
    for _ in range(max_len):
        # 1) 取像素帧（固定 corner2 视角）
        H, W = IMAGE_RESOLUTION
        img = render_rgb(env, H, W, CAMERA_NAME)
        # 若需要严格 224x224，可再次 resize（mujoco-py 已按宽高渲染，一般不必）
        if img.shape[0] != H or img.shape[1] != W:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        frames.append(img)

        # 2) 记录 4 维末端状态 [x, y, z, gripper]
        robot_states.append(obs[:4].copy())

        # 3) 专家动作
        a = policy.get_action(obs)
        actions.append(a.tolist())

        # 4) 交互一步（v2: done 单一布尔）
        obs, reward, done, info = env.step(a)
        info_last = info
        # 成功或结束则退出
        if int(info.get("success", 0)) == 1 or done:
            break

    # 判定是否成功（供“只保成功”逻辑使用）
    success = int(info_last.get("success", 0)) == 1

    # 保存本条轨迹（视频 + JSON）
    traj_dir = task_dir / f"trajectory_{traj_idx:03d}"
    traj_dir.mkdir(parents=True, exist_ok=True)

    media.write_video(traj_dir / "video.mp4", frames, fps=FPS)

    with open(traj_dir / "trajectory_data.json", "w") as f:
        json.dump(
            {
                "task_name": task_dir.name,
                "trajectory_index": traj_idx,
                "num_steps": len(frames),
                "success": int(success),
                "actions": actions,  # [T, 4]
                "robot_states": np.array(robot_states).tolist(),  # [T, 4]
                "camera_name": CAMERA_NAME,
                "image_resolution": IMAGE_RESOLUTION,
            },
            f,
            indent=2,
        )

    return success


def main():
    print("=== Meta-World v2 数据采集 ===")
    print(f"输出目录: {OUTPUT_DIR.resolve()}")
    print(f"相机视角: {CAMERA_NAME}, 分辨率: {IMAGE_RESOLUTION}, 每任务: {NUM_TRAJECTORIES_PER_TASK} 条\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for task_name in TASKS_TO_COLLECT:
        # 找到对应的 v2（goal-observable）环境类与专家策略
        env_key = task_name + "-goal-observable"
        if task_name not in POLICY_MAPPING or env_key not in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE:
            print(f"⚠️  跳过 {task_name}：未找到专家或环境类（v2 安装不完整？）")
            continue

        policy = POLICY_MAPPING[task_name]()  # 脚本化专家策略
        env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_key]

        # 仅创建一次环境，内部复用；最后统一关闭
        env = env_cls()
        task_dir = OUTPUT_DIR / task_name
        task_dir.mkdir(parents=True, exist_ok=True)

        count_saved = 0
        traj_idx = 0
        pbar = tqdm(total=NUM_TRAJECTORIES_PER_TASK, desc=f"Collecting {task_name}", ncols=100)

        try:
            while count_saved < NUM_TRAJECTORIES_PER_TASK:
                success = collect_one_trajectory(env, policy, traj_idx, task_dir)

                # 只保存成功：失败则不计数、继续重采
                if KEEP_ONLY_SUCCESS and not success:
                    traj_idx += 1
                    continue

                count_saved += 1
                traj_idx += 1
                pbar.update(1)
        finally:
            pbar.close()
            env.close()

        print(f"✅ {task_name}: 保存 {count_saved} 条到 {task_dir}")

    print("\n=== 所有任务完成 ===")


if __name__ == "__main__":
    main()

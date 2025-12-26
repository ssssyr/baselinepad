# -*- coding: utf-8 -*-
"""
Meta-World v2 数据采集脚本（50个任务：corner3相机视角 + 力信号）
- 采集所有 50 个 Meta-World v2 任务
- 固定相机 'corner3'，离屏渲染 PNG 帧
- 每条轨迹的 JSON 条目仅包含：
   {
     "instruction": <str>,
     "features": [[x,y,z,grip], ...],          # 世界坐标（4维）
     "force_features": [[fx,fy,fz,tx,ty,tz], ...],  # EE坐标系六维力（6维）
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
from scipy.spatial.transform import Rotation

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
    SawyerButtonPressTopdownWallV2Policy,
    SawyerButtonPressV2Policy,
    SawyerButtonPressWallV2Policy,
    SawyerCoffeeButtonV2Policy,
    SawyerCoffeePullV2Policy,
    SawyerCoffeePushV2Policy,
    SawyerDialTurnV2Policy,
    SawyerDisassembleV2Policy,
    SawyerDoorCloseV2Policy,
    SawyerDoorLockV2Policy,
    SawyerDoorOpenV2Policy,
    SawyerDoorUnlockV2Policy,
    SawyerDrawerCloseV2Policy,
    SawyerDrawerOpenV2Policy,
    SawyerFaucetCloseV2Policy,
    SawyerFaucetOpenV2Policy,
    SawyerHammerV2Policy,
    SawyerHandInsertV2Policy,
    SawyerHandlePressSideV2Policy,
    SawyerHandlePressV2Policy,
    SawyerHandlePullSideV2Policy,
    SawyerHandlePullV2Policy,
    SawyerLeverPullV2Policy,
    SawyerPegInsertionSideV2Policy,
    SawyerPegUnplugSideV2Policy,
    SawyerPickOutOfHoleV2Policy,
    SawyerPickPlaceV2Policy,
    SawyerPickPlaceWallV2Policy,
    SawyerPlateSlideBackSideV2Policy,
    SawyerPlateSlideBackV2Policy,
    SawyerPlateSlideSideV2Policy,
    SawyerPlateSlideV2Policy,
    SawyerPushBackV2Policy,
    SawyerPushV2Policy,
    SawyerPushWallV2Policy,
    SawyerReachV2Policy,
    SawyerReachWallV2Policy,
    SawyerShelfPlaceV2Policy,
    SawyerSoccerV2Policy,
    SawyerStickPullV2Policy,
    SawyerStickPushV2Policy,
    SawyerSweepIntoV2Policy,
    SawyerSweepV2Policy,
    SawyerWindowCloseV2Policy,
    SawyerWindowOpenV2Policy,
)

# ---------------- 配 置 ---------------- #

# Meta-World v2 采集任务：包含所有50个任务（带力信号）
TASKS_TO_COLLECT: Dict[str, str] = {
    "assembly-v2": "assemble the peg",
    "button-press-v2": "press the button",
    "button-press-wall-v2": "press the wall-mounted button",
    "button-press-topdown-v2": "press the button from the top",
    "button-press-topdown-wall-v2": "press the wall-mounted button from the top",
    "coffee-button-v2": "press the coffee machine button",
    "basketball-v2": "shoot the basketball into the hoop",
    "bin-picking-v2": "pick the object from the bin",
    "box-close-v2": "close the box lid",
    "coffee-pull-v2": "pull the coffee mug",
    "coffee-push-v2": "push the coffee mug",
    "dial-turn-v2": "turn the dial",
    "disassemble-v2": "disassemble the object",
    "door-close-v2": "close the door",
    "door-lock-v2": "lock the door",
    "door-open-v2": "open the door",
    "door-unlock-v2": "unlock the door",
    "drawer-close-v2": "close the drawer",
    "drawer-open-v2": "open the drawer",
    "faucet-close-v2": "close the faucet",
    "faucet-open-v2": "open the faucet",
    "hammer-v2": "hammer object",
    "hand-insert-v2": "insert hand into slot",
    "handle-press-side-v2": "press side handle",
    "handle-press-v2": "press the handle",
    "handle-pull-side-v2": "pull side handle",
    "handle-pull-v2": "pull the handle",
    "lever-pull-v2": "pull the lever",
    "peg-insert-side-v2": "insert peg from side",
    "peg-unplug-side-v2": "unplug side peg",
    "pick-out-of-hole-v2": "pick object out of hole",
    "pick-place-v2": "pick and place object",
    "pick-place-wall-v2": "pick and place object to wall target",
    "plate-slide-back-side-v2": "slide plate back from side",
    "plate-slide-back-v2": "slide plate to back",
    "plate-slide-side-v2": "slide plate to side",
    "plate-slide-v2": "slide plate",
    "push-back-v2": "push object to back",
    "push-v2": "push object",
    "push-wall-v2": "push object to wall target",
    "reach-v2": "reach target",
    "reach-wall-v2": "reach wall target",
    "shelf-place-v2": "place object on shelf",
    "soccer-v2": "kick soccer ball into goal",
    "stick-pull-v2": "pull stick",
    "stick-push-v2": "push stick",
    "sweep-into-v2": "sweep object into bin",
    "sweep-v2": "sweep object",
    "window-close-v2": "close the window",
    "window-open-v2": "open the window",
}

NUM_TRAJECTORIES_PER_TASK = 50          # 每任务采集轨迹数
KEEP_ONLY_SUCCESS = False               # 仅保留成功轨迹（会重采直到够数）
CAMERA_NAME = "corner3"                 # ['corner','corner2','corner3','corner4','topview','behindGripper','gripperPOV']
IMAGE_RESOLUTION = (256, 256)           # (H, W)
OUTPUT_DIR = Path("/mnt/sda/datasets/metaworld_corner3_all_with_force")  # 输出根目录（所有50个任务，带力信号）

# 任务名 -> 专家策略（包含所有50个任务）
POLICY_MAPPING = {
    "assembly-v2": SawyerAssemblyV2Policy,
    "button-press-v2": SawyerButtonPressV2Policy,
    "button-press-wall-v2": SawyerButtonPressWallV2Policy,
    "button-press-topdown-v2": SawyerButtonPressTopdownV2Policy,
    "button-press-topdown-wall-v2": SawyerButtonPressTopdownWallV2Policy,
    "coffee-button-v2": SawyerCoffeeButtonV2Policy,
    "basketball-v2": SawyerBasketballV2Policy,
    "bin-picking-v2": SawyerBinPickingV2Policy,
    "box-close-v2": SawyerBoxCloseV2Policy,
    "coffee-pull-v2": SawyerCoffeePullV2Policy,
    "coffee-push-v2": SawyerCoffeePushV2Policy,
    "dial-turn-v2": SawyerDialTurnV2Policy,
    "disassemble-v2": SawyerDisassembleV2Policy,
    "door-close-v2": SawyerDoorCloseV2Policy,
    "door-lock-v2": SawyerDoorLockV2Policy,
    "door-open-v2": SawyerDoorOpenV2Policy,
    "door-unlock-v2": SawyerDoorUnlockV2Policy,
    "drawer-close-v2": SawyerDrawerCloseV2Policy,
    "drawer-open-v2": SawyerDrawerOpenV2Policy,
    "faucet-close-v2": SawyerFaucetCloseV2Policy,
    "faucet-open-v2": SawyerFaucetOpenV2Policy,
    "hammer-v2": SawyerHammerV2Policy,
    "hand-insert-v2": SawyerHandInsertV2Policy,
    "handle-press-side-v2": SawyerHandlePressSideV2Policy,
    "handle-press-v2": SawyerHandlePressV2Policy,
    "handle-pull-side-v2": SawyerHandlePullSideV2Policy,
    "handle-pull-v2": SawyerHandlePullV2Policy,
    "lever-pull-v2": SawyerLeverPullV2Policy,
    "peg-insert-side-v2": SawyerPegInsertionSideV2Policy,
    "peg-unplug-side-v2": SawyerPegUnplugSideV2Policy,
    "pick-out-of-hole-v2": SawyerPickOutOfHoleV2Policy,
    "pick-place-v2": SawyerPickPlaceV2Policy,
    "pick-place-wall-v2": SawyerPickPlaceWallV2Policy,
    "plate-slide-back-side-v2": SawyerPlateSlideBackSideV2Policy,
    "plate-slide-back-v2": SawyerPlateSlideBackV2Policy,
    "plate-slide-side-v2": SawyerPlateSlideSideV2Policy,
    "plate-slide-v2": SawyerPlateSlideV2Policy,
    "push-back-v2": SawyerPushBackV2Policy,
    "push-v2": SawyerPushV2Policy,
    "push-wall-v2": SawyerPushWallV2Policy,
    "reach-v2": SawyerReachV2Policy,
    "reach-wall-v2": SawyerReachWallV2Policy,
    "shelf-place-v2": SawyerShelfPlaceV2Policy,
    "soccer-v2": SawyerSoccerV2Policy,
    "stick-pull-v2": SawyerStickPullV2Policy,
    "stick-push-v2": SawyerStickPushV2Policy,
    "sweep-into-v2": SawyerSweepIntoV2Policy,
    "sweep-v2": SawyerSweepV2Policy,
    "window-close-v2": SawyerWindowCloseV2Policy,
    "window-open-v2": SawyerWindowOpenV2Policy,
}
# ------------------------------------- #

def render_rgb(env, h: int, w: int, camera: str) -> np.ndarray:
    """离屏渲染一帧 RGB（HxWx3, uint8）"""
    frame = env.sim.render(width=w, height=h, camera_name=camera)
    return frame  # 如遇画面倒置，可改：return frame[::-1]

def get_ee_force_torque(env) -> np.ndarray:
    """
    获取EE坐标系下的六维力 [fx, fy, fz, tx, ty, tz]

    Args:
        env: Meta-World环境实例

    Returns:
        np.ndarray: 6维数组 [fx, fy, fz, tx, ty, tz]，EE坐标系
    """
    # 获取传感器数据索引和维度
    force_idx = env.model.sensor_name2id("ee_force")
    torque_idx = env.model.sensor_name2id("ee_torque")

    # 获取传感器的数据维度（force和torque都是3维）
    force_adr = env.model.sensor_adr[force_idx]  # 传感器在sensordata中的起始位置
    torque_adr = env.model.sensor_adr[torque_idx]

    # 获取世界坐标系下的力和力矩（MuJoCo传感器默认输出世界坐标系）
    force_world = env.sim.data.sensordata[force_adr:force_adr+3].copy()  # [fx, fy, fz]
    torque_world = env.sim.data.sensordata[torque_adr:torque_adr+3].copy()  # [tx, ty, tz]

    # 获取EE（hand body）的姿态四元数 [w, x, y, z]
    body_id = env.model.body_name2id("hand")
    quat = env.sim.data.body_xquat[body_id].copy()  # [w, x, y, z]

    # 将四元数转换为旋转矩阵
    # 注意：MuJoCo四元数格式是 [w, x, y, z]，scipy需要 [x, y, z, w]
    rotation = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
    R_world_to_ee = rotation.as_matrix().T  # 转置得到世界到EE的变换

    # 转换到EE坐标系
    force_ee = R_world_to_ee @ force_world
    torque_ee = R_world_to_ee @ torque_world

    return np.concatenate([force_ee, torque_ee])  # [fx, fy, fz, tx, ty, tz]

def collect_one_trajectory(
    env,
    policy,
    traj_idx: int,
    task_dir: Path,
    image_resolution: Tuple[int, int],
    camera_name: str,
) -> Tuple[bool, int, List[List[float]], List[List[float]]]:
    """
    采集一条轨迹：保存 PNG 帧，并返回 (success, num_steps, states, forces)
      - states: 每步 4 维绝对状态 [x,y,z,grip]（从 obs[:4] 取）
      - forces: 每步 6 维 EE 坐标系力 [fx,fy,fz,tx,ty,tz]
    目录结构：task_dir / f"class_{traj_idx:06d}"/ frame_0000.png, ...
    """
    traj_dir = task_dir / f"class_{traj_idx:06d}"
    traj_dir.mkdir(parents=True, exist_ok=True)

    env.seed(traj_idx)
    obs = env.reset()

    H, W = image_resolution
    max_len = int(getattr(env, "max_path_length", 500))

    states: List[List[float]] = []
    forces: List[List[float]] = []
    info_last = {}
    for step_idx in range(max_len):
        # 图像
        img = render_rgb(env, H, W, camera_name)
        Image.fromarray(img).save(traj_dir / f"frame_{step_idx:04d}.png")
        # 绝对状态（世界系）
        states.append((obs[:4]).tolist())  # [x, y, z, grip]

        # 获取EE坐标系下的六维力（单独存储）
        forces.append(get_ee_force_torque(env).tolist())  # [fx, fy, fz, tx, ty, tz]

        # 与环境交互：使用专家策略动作（不保存动作）
        action = policy.get_action(obs)
        obs, _, done, info = env.step(np.asarray(action, dtype=np.float32))
        info_last = info
        if int(info.get("success", 0)) == 1 or done:
            break

    return (int(info_last.get("success", 0)) == 1), len(states), states, forces

def main():
    print("=== Meta-World v2 全部任务数据采集（带力信号）===")
    print(f"输出目录: {OUTPUT_DIR.resolve()}")
    print(f"相机: {CAMERA_NAME}, 分辨率: {IMAGE_RESOLUTION}, 每任务轨迹数: {NUM_TRAJECTORIES_PER_TASK}")
    print(f"任务总数: {len(TASKS_TO_COLLECT)}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for task_idx, (task_name, instruction) in enumerate(TASKS_TO_COLLECT.items()):
        print(f"\n[{task_idx+1}/{len(TASKS_TO_COLLECT)}] 开始处理任务: {task_name}")
        env_key = task_name + "-goal-observable"

        # 检查策略映射
        if task_name not in POLICY_MAPPING:
            print(f"⚠️ 跳过 {task_name}: 未找到专家策略")
            continue

        # 检查环境类
        if env_key not in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE:
            print(f"⚠️ 跳过 {task_name}: 环境类 {env_key} 不存在")
            continue

        print(f"✓ 任务 {task_name} 检查通过，开始采集...")

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
                success, nsteps, states, forces = collect_one_trajectory(
                    env=env,
                    policy=policy,
                    traj_idx=attempt,
                    task_dir=task_dir,
                    image_resolution=IMAGE_RESOLUTION,
                    camera_name=CAMERA_NAME,
                )

                traj_entry = {
                    "instruction": instruction,
                    "features": states,      # [[x,y,z,grip], ...] 世界坐标（4维）
                    "force_features": forces,  # [[fx,fy,fz,tx,ty,tz], ...] EE坐标系（6维）
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
        print(f"任务 {task_name} 完成，准备进行下一个任务...")

    print(f"\n=== 全部 {len(TASKS_TO_COLLECT)} 个任务完成 ===")

if __name__ == "__main__":
    main()

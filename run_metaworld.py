"""
run_metaworld.py

Design goals:
- Safe to import (batch scripts can import run_single_rollout without triggering rollouts).
- Standalone execution (python run_metaworld.py) runs tasks defined in evaluation/run_cfg.py.
- When save_video=True: ALWAYS save a video whether success or failure.
- If success is achieved: stop early (do NOT run full max_steps), then save video and return.

Stability improvements included:
- env.close() in finally to avoid MuJoCo/EGL resource leaks during long runs.
- torch.no_grad() around inference to avoid graph/memory accumulation.
- motion planner avoids divide-by-zero and checks for NaN/Inf actions.
"""

import os
import time
import random
import gc

import cv2
import mediapy
import numpy as np
import torch
from tqdm import tqdm
from scipy.spatial.transform import Rotation

# Must be set BEFORE importing metaworld/mujoco backends
os.environ.setdefault("MUJOCO_GL", "egl")

from metaworld.envs import (
    ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
    ALL_V2_ENVIRONMENTS_GOAL_HIDDEN,  # kept for compatibility (not used here)
)

from evaluation.agent import DiffusionAgent
from evaluation.run_cfg import INSTRUCTIONS, META_CONFIG


def set_random_seed(seed=None):
    """Set random seed for reproducibility or randomness."""
    if seed is None:
        seed = int(time.time() * 1000000) % (2**32)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"🎲 Set random seed: {seed}")
    return seed


def add_bound(rgb, color="red"):
    """Draw a colored border on an RGB image array (H,W,3)."""
    width = 10
    c = 0 if color == "red" else 1

    rgb = rgb.copy()
    # top
    rgb[:width, :, 1:3] = 100
    rgb[:width, :, c] = 255
    # bottom
    rgb[-width:, :, 1:3] = 100
    rgb[-width:, :, c] = 255
    # left
    rgb[:, :width, 1:3] = 100
    rgb[:, :width, c] = 255
    # right
    rgb[:, -width:, 1:3] = 100
    rgb[:, -width:, c] = 255
    return rgb


def merge_img(obs, predict, img_word):
    """Merge observation + prediction + header row."""
    img_1 = cv2.resize(obs, (256, 256), interpolation=cv2.INTER_AREA)
    image = np.concatenate((add_bound(img_1, color="green"), predict), axis=1)
    image = np.concatenate((np.asarray(img_word), image), axis=0)
    return image


def plot_word():
    from PIL import Image as PILImage, ImageDraw, ImageFont

    fnt_title = ImageFont.truetype("evaluation/TIMES.ttf", int(600 / 20))
    img_word = PILImage.new("RGB", (256 * 4, 40), color="white")
    draw = ImageDraw.Draw(img_word)
    draw.text((50, 10), "Observations", font=fnt_title, fill="green")
    draw.text((572, 10), "Predictions", font=fnt_title, fill="red")
    return img_word


def get_ee_force_torque(env):
    """
    Return end-effector force/torque in EE frame if sensors exist; else zeros.
    Output shape: (6,)
    """
    try:
        force_idx = env.model.sensor_name2id("ee_force")
        torque_idx = env.model.sensor_name2id("ee_torque")

        force_adr = env.model.sensor_adr[force_idx]
        torque_adr = env.model.sensor_adr[torque_idx]

        force_world = env.sim.data.sensordata[force_adr : force_adr + 3].copy()
        torque_world = env.sim.data.sensordata[torque_adr : torque_adr + 3].copy()

        body_id = env.model.body_name2id("hand")
        quat = env.sim.data.body_xquat[body_id].copy()  # mujoco: w,x,y,z

        rotation = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        R_world_to_ee = rotation.as_matrix().T

        force_ee = R_world_to_ee @ force_world
        torque_ee = R_world_to_ee @ torque_world
        return np.concatenate([force_ee, torque_ee]).astype(np.float32)
    except Exception:
        return np.zeros(6, dtype=np.float32)


def motion_planner(
    target_xyz,
    target_gripper,
    curr_xyz,
    curr_gripper,
    env,
    image_3,
    thirdview,
    predict_img=None,
    img_word=None,
    save_video=True,
):
    """
    A simple motion planner to reach the target pose from current pose.

    Returns:
        info (dict), img (np.ndarray)  # img is last rendered frame
    """
    stage = 0
    grasp_moment = False
    motion_steps = 50
    success_flag = False
    img = None

    # Determine whether to run grasp stage
    if np.abs(float(target_gripper) - float(curr_gripper)) > 0.2:
        grasp_moment = True
        print("prepare grasp!!")

    info = {"success": 0.0}

    for _ in range(motion_steps):
        a = -np.ones(4, dtype=np.float32)

        if stage == 0:
            # gripper control
            if target_gripper < 0.75 and curr_gripper < 0.75:
                a[3] = 0.7

            delta = (target_xyz - curr_xyz).astype(np.float32)
            dist = float(np.linalg.norm(delta))

            # Safe normalization
            if dist < 1e-8:
                a[:3] = 0.0
            else:
                velocity = 0.6 if dist > 0.03 else 0.3
                a[:3] = (delta / dist) * velocity

            if not np.isfinite(a).all():
                print(f"[motion_planner] Non-finite action: {a}, dist={dist}, delta={delta}")
                break

            obs, r, done, info = env.step(a)
            img = env.render(offscreen=True, camera_name=thirdview, resolution=[224, 224], depth=False)

            if save_video:
                if predict_img is not None and img_word is not None:
                    image_3.append(merge_img(img, predict_img, img_word))
                else:
                    image_3.append(img)

            o = env._get_obs()
            curr_xyz, curr_gripper = o[:3], float(o[3])

            if info.get("success", 0):
                success_flag = True
                break

            # Target reached -> next stage or exit
            if float(np.linalg.norm(target_xyz - curr_xyz)) < 0.005:
                stage += 1 if grasp_moment else motion_steps

        elif stage < 20:
            # grasping stage
            if target_gripper < 0.82:
                a = np.array([0, 0, 0, 0.7], dtype=np.float32)
                obs, r, done, info = env.step(a)

                img = env.render(offscreen=True, camera_name=thirdview, resolution=[224, 224], depth=False)

                if save_video:
                    if predict_img is not None and img_word is not None:
                        image_3.append(merge_img(img, predict_img, img_word))
                    else:
                        image_3.append(img)

                if info.get("success", 0):
                    success_flag = True
                    break

                stage += 1
            else:
                break
        else:
            break

    if success_flag:
        info["success"] = 1.0

    return info, img


def _save_video_if_needed(task, traj_idx, frames, META_CONFIG):
    """Save mp4 if frames exist. Always best-effort, never raises."""
    video_dir = META_CONFIG.get("video_dir", "./videos")
    out_dir = os.path.join(video_dir, "rollout_metaworld")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{task}_{traj_idx}.mp4")
    try:
        if frames and len(frames) > 0:
            mediapy.write_video(out_path, frames, fps=20)
            print(f"[video] saved to: {out_path}")
        else:
            print(f"[video] no frames collected; skip saving: {out_path}")
    except Exception as e:
        print(f"[video] failed to save {out_path}: {repr(e)}")


def run_single_rollout(agent, task, selected_id, traj_idx, META_CONFIG, INSTRUCTIONS, save_video=True):
    """
    Run a single rollout for a task.

    IMPORTANT video behavior:
    - If save_video=True: ALWAYS save video whether success or failure.
    - If success happens early: stop immediately, then save video and return True.

    Args:
        agent: DiffusionAgent
        task: str
        selected_id: int (kept for compatibility; used in env seed as before)
        traj_idx: int
        META_CONFIG, INSTRUCTIONS: from evaluation/run_cfg.py
        save_video: bool

    Returns:
        bool: True if success, else False
    """
    thirdview = META_CONFIG["thirdview_camera"]

    env = None
    frames = []
    img_word = None
    success = False

    try:
        # Use random_seed from config if available, otherwise use time-based seed
        base_seed = META_CONFIG.get("random_seed", None)
        if base_seed is not None:
            # Combine base_seed with selected_id and traj_idx for reproducible per-rollout seeds
            seed = base_seed + selected_id * 1000 + traj_idx
        else:
            seed = None  # Use time-based random seed
        set_random_seed(seed)

        env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task + "-goal-observable"]
        # Use traj_idx for env seed to ensure consistent initialization across batch/standalone runs
        env = env_cls(seed=traj_idx + 100)

        # Keep your original assembly handling
        if task.startswith("assembly"):
            env.random_init = False

        print(f"task name {task} traj_idx {traj_idx}")

        visualize_prediction = bool(META_CONFIG.get("visualize_prediction", False))
        if save_video and visualize_prediction:
            img_word = plot_word()

        obs = env.reset()
        img = env.render(offscreen=True, camera_name=thirdview, resolution=[224, 224], depth=False)

        max_steps = int(META_CONFIG["max_steps"])

        for plan_step in tqdm(range(max_steps), disable=not save_video):
            text = INSTRUCTIONS[task]

            state4 = env._get_obs()[:4]
            curr_xyz, curr_gripper = state4[:3], float(state4[3])

            force = get_ee_force_torque(env)

            with torch.no_grad():
                samples, sample_a, _ = agent.action(text, img, None, state4, force)

            predict_img = None
            if save_video and visualize_prediction:
                with torch.no_grad():
                    predict_img = agent.decode_rgb(img, samples)
                predict_img = add_bound(predict_img)

            # Select target action (use 2nd frame of predicted sequence)
            if getattr(agent.args, "action_steps", 0) > 0 and sample_a is not None:
                a_seq = sample_a.reshape(agent.args.action_steps, agent.args.action_dim)
                if save_video:
                    print(f"🧭 Full predicted action seq (xyzg per step):\n{np.array2string(a_seq, precision=3, floatmode='fixed')}")
                    print(f"🧭 Gripper seq: {np.array2string(a_seq[:,3], precision=3, floatmode='fixed')}, current gripper: {curr_gripper:.3f}")

                target = a_seq[0] / agent.args.action_scale  # Use 2nd frame (index 1)
                target_xyz, target_gripper = target[:3], float(target[3])
            else:
                target = sample_a / agent.args.action_scale
                target_xyz, target_gripper = target[0, 0, :3], float(target[0, 0, 3])

            if save_video:
                print(f"🧭 Target used (step1 xyzg): {np.array2string(target_xyz, precision=3, floatmode='fixed')}, target_gripper: {target_gripper:.3f}")

            info, img = motion_planner(
                target_xyz,
                target_gripper,
                curr_xyz,
                curr_gripper,
                env,
                frames,
                thirdview,
                predict_img=predict_img,
                img_word=img_word,
                save_video=save_video,
            )

            # Early stop on success (your requirement)
            if info.get("success", 0):
                success = True
                if save_video:
                    print(task, traj_idx, "success (early stop)")
                break

        # ALWAYS save video if requested (success or failure)
        if save_video:
            _save_video_if_needed(task, traj_idx, frames, META_CONFIG)

        return success

    finally:
        # Critical cleanup for batch stability
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        env = None
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


def build_agent_from_meta(META_CONFIG):
    """Helper for standalone execution."""
    return DiffusionAgent(
        ckpt_path=META_CONFIG["ckpt_path"],
        vae_path=META_CONFIG["vae_path"],
        clip_path=META_CONFIG["clip_path"],
        denoise_steps=META_CONFIG["denoise_steps"],
        device_id=META_CONFIG.get("gpu_id", 0),
    )


def main():
    # Do not modify task definitions: read from run_cfg as requested.
    task_list = META_CONFIG.get("task_list", list(INSTRUCTIONS.keys()))
    rollout_num = int(META_CONFIG.get("rollout_num", 1))

    # Standalone should save videos
    save_video = True

    agent = build_agent_from_meta(META_CONFIG)

    success_num = np.zeros(len(task_list), dtype=np.int32)

    for selected_id, task in enumerate(task_list):
        success_num[selected_id] = 0
        for traj_idx in range(rollout_num):
            ok = run_single_rollout(
                agent,
                task,
                selected_id,
                traj_idx,
                META_CONFIG,
                INSTRUCTIONS,
                save_video=save_video,
            )
            if ok:
                success_num[selected_id] += 1

    for i, t in enumerate(task_list):
        print(t, int(success_num[i]))


if __name__ == "__main__":
    main()

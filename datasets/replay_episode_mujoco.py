#!/usr/bin/env python3
"""Replay a recorded MetaWorld episode in MuJoCo using stored poses.

Loads `dataset_rgb_s_d.json`, pulls the pose targets from one episode, and
drives the button-press environment toward those poses while saving a video
for inspection.
"""

import argparse
import json
import os
from typing import List, Optional, Tuple

import mediapy
import numpy as np

# Headless rendering support
os.environ.setdefault("MUJOCO_GL", "egl")
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE  # noqa: E402


def load_episode_steps(
    dataset_dir: str, episode_idx: int, limit: Optional[int], field: str
) -> List[dict]:
    json_path = os.path.join(dataset_dir, "dataset_rgb_s_d.json")
    with open(json_path, "r") as f:
        steps = json.load(f)

    episode_steps = [s for s in steps if int(s["episode"]) == episode_idx]
    if not episode_steps:
        raise ValueError(f"Episode {episode_idx} not found in {json_path}")

    episode_steps = sorted(episode_steps, key=lambda x: int(x.get("frame", x.get("idx", 0))))
    if limit is not None:
        episode_steps = episode_steps[:limit]

    missing = [i for i, s in enumerate(episode_steps) if field not in s]
    if missing:
        raise KeyError(f"Field '{field}' missing in episode {episode_idx} at indices {missing[:5]}")
    return episode_steps


def move_to_pose(
    env,
    target_pose: np.ndarray,
    camera_name: str,
    frame_buffer: List[np.ndarray],
    resolution: int,
    reach_eps: float = 3e-3,
    max_steps: int = 60,
    record_intermediate: bool = True,
    gripper_open_thresh: float = 0.75,
    gripper_close_cmd: float = 0.7,
) -> Tuple[np.ndarray, float, dict]:
    """Simple controller: move end-effector toward target xyz and align gripper."""
    obs = env._get_obs()
    curr_xyz = np.array(obs[:3], dtype=np.float32)
    curr_gripper = float(obs[3])

    target_xyz = np.array(target_pose[:3], dtype=np.float32)
    target_gripper = float(target_pose[3])

    grasp_moment = abs(target_gripper - curr_gripper) > 0.2
    stage = 0
    info: dict = {}

    for _ in range(max_steps):
        action = np.zeros(4, dtype=np.float32)
        delta = target_xyz - curr_xyz
        dist = np.linalg.norm(delta)

        if stage == 0:
            if dist > 1e-6:
                velocity = 0.6 if dist > 0.03 else 0.3
                action[:3] = np.clip(delta / (dist + 1e-9) * velocity, -1.0, 1.0)
            # Gripper: for button press, keep完全打开（-1）; 若需要闭合再执行正向指令
            if target_gripper >= gripper_open_thresh:
                action[3] = -1.0
            else:
                action[3] = gripper_close_cmd
            if dist < reach_eps:
                stage = 1 if grasp_moment else 2
        elif stage == 1:
            action[3] = gripper_close_cmd
            if abs(target_gripper - curr_gripper) < 0.02:
                stage = 2

        obs, reward, done, info = env.step(action)
        curr_xyz, curr_gripper = env._get_obs()[:3], env._get_obs()[3]

        if camera_name and record_intermediate:
            frame = env.render(
                offscreen=True,
                camera_name=camera_name,
                resolution=[resolution, resolution],
                depth=False,
            )
            frame_buffer.append(frame)

        new_delta = target_xyz - curr_xyz
        new_dist = np.linalg.norm(new_delta)
        if stage == 2 and new_dist < reach_eps:
            break

    if camera_name and not record_intermediate:
        frame_buffer.append(
            env.render(
                offscreen=True,
                camera_name=camera_name,
                resolution=[resolution, resolution],
                depth=False,
            )
        )

    return np.array(curr_xyz, dtype=np.float32), float(curr_gripper), info


def rollout_episode(
    dataset_dir: str,
    episode_idx: int,
    video_path: str,
    camera_name: str = "corner3",
    resolution: int = 256,
    max_frames: Optional[int] = None,
    seed: int = 0,
    fps: int = 20,
    field: str = "state",
    controller_steps: int = 60,
    record_intermediate: bool = True,
) -> None:
    steps = load_episode_steps(dataset_dir, episode_idx, max_frames, field)
    env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["button-press-v2-goal-observable"]
    env = env_cls(seed=seed)
    env.reset()

    frames: List[np.ndarray] = []
    if camera_name and record_intermediate:
        frames.append(
            env.render(
                offscreen=True,
                camera_name=camera_name,
                resolution=[resolution, resolution],
                depth=False,
            )
        )

    print(f"Rolling out episode {episode_idx} with {len(steps)} targets using field '{field}'...")
    for i, step in enumerate(steps):
        pose = np.array(step[field], dtype=np.float32)
        reached_xyz, reached_grip, info = move_to_pose(
            env,
            pose,
            camera_name,
            frames,
            resolution,
            max_steps=controller_steps,
            record_intermediate=record_intermediate,
        )
        pos_err = np.linalg.norm(pose[:3] - reached_xyz)
        grip_err = abs(pose[3] - reached_grip)
        success_flag = info.get("success", 0)
        print(
            f"[{i:03d}] target {pose} -> reached {np.append(reached_xyz, reached_grip)} "
            f"| pos_err={pos_err:.4f} grip_err={grip_err:.4f} success={success_flag}"
        )

    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    mediapy.write_video(video_path, frames, fps=fps)
    print(f"Saved rollout video to {video_path} ({len(frames)} frames, {fps} fps)")

    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a dataset episode in MuJoCo.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="/mnt/sda/datasets/metaworldcorner3-features_button_press_v2",
        help="Path to feature dataset folder containing dataset_rgb_s_d.json.",
    )
    parser.add_argument("--episode", type=int, default=0, help="Episode index to replay.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Where to save the rendered video (mp4). Defaults to output/replay_epXXXXXXX.mp4.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for env reset.")
    parser.add_argument(
        "--camera",
        type=str,
        default="corner3",
        help="MuJoCo camera name for rendering.",
    )
    parser.add_argument("--resolution", type=int, default=256, help="Render resolution.")
    parser.add_argument(
        "--max-targets",
        type=int,
        default=None,
        help="Optionally limit number of targets from the episode.",
    )
    parser.add_argument("--fps", type=int, default=20, help="FPS for the output video.")
    parser.add_argument(
        "--field",
        type=str,
        default="state",
        choices=["state", "action"],
        help="Which field to use as target pose.",
    )
    parser.add_argument(
        "--controller-steps",
        type=int,
        default=60,
        help="Max controller steps per target pose.",
    )
    parser.add_argument(
        "--render-intermediate",
        action="store_true",
        help="Record every controller step instead of only the final pose per target.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    default_out = os.path.join("output", f"replay_ep{args.episode:07}.mp4")
    out_path = args.output or default_out
    rollout_episode(
        dataset_dir=args.dataset,
        episode_idx=args.episode,
        video_path=out_path,
        camera_name=args.camera,
        resolution=args.resolution,
        max_frames=args.max_targets,
        seed=args.seed,
        fps=args.fps,
        field=args.field,
        controller_steps=args.controller_steps,
        record_intermediate=args.render_intermediate,
    )

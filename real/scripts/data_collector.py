#!/usr/bin/env python3
"""
data_collector.py

A script for collecting multi-modal data from the UR10 robot setup for training.
- Uses event-driven camera thread (only new frames) + high-frequency robot thread.
- Aligns data by cam_ts_mono (same clock as robot), matching nearest robot state.
- Causal ordering: observation (o_t) -> action (a_t) with proper timestamps.
- Saves episodes with metadata for reproducibility.

Key improvements:
- Camera: uses cam_ts_mono for robot alignment (same clock domain)
- Three-tier deduplication: frame_id > cam_ts_hw > image_hash
- Ring buffer: preserves recent 2 seconds for alignment
- Multiple timestamps: cam_ts_hw, cam_ts_mono, robot_ts, action_ts
- Alignment threshold: skips if robot-camera time diff > 50ms
- Metadata saved: camera serial, fps, color space, etc.
"""

import os
import sys
import time
import threading
import json
import shutil
import multiprocessing
from pathlib import Path
from collections import deque
from typing import Optional, Tuple, Dict, Any
import numpy as np
import click
import cv2

# Set multiprocessing start method to fork (inherits parent process permissions)
multiprocessing.set_start_method('fork')

# Add project root to Python path to allow absolute imports
project_root = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(project_root))
# Add third_party directory for realsense_interface
sys.path.insert(0, str(project_root / "real" / "third_party"))

from real.configs.ur10_config import CONFIG
from real.hardware.camera_manager import CameraManager
from real.hardware.ur10_manager import UR10Manager
from real.scripts.gamepad_handler import GamepadHandler


# ==============================================================================
# Time Alignment Configuration
# ==============================================================================

ALIGNMENT_THRESHOLD_S = 0.05
RING_BUFFER_SIZE_S = 2.0
CAM_TS_HW_THRESHOLD_S = 0.005


# ==============================================================================
# Data Structures
# ==============================================================================

class TimestampedData:
    __slots__ = ['timestamp', 'data']
    def __init__(self, timestamp: float, data: Any):
        self.timestamp = timestamp
        self.data = data

class RingBuffer:
    def __init__(self, max_duration_seconds: float):
        self.max_duration = max_duration_seconds
        self.buffer: deque = deque()
        self._lock = threading.Lock()

    def put(self, timestamp: float, data: Any):
        with self._lock:
            cutoff_time = timestamp - self.max_duration
            while self.buffer and self.buffer[0].timestamp < cutoff_time:
                self.buffer.popleft()
            self.buffer.append(TimestampedData(timestamp, data))

    def get_nearest(self, target_ts: float, max_dt: Optional[float] = None) -> Optional[TimestampedData]:
        with self._lock:
            if not self.buffer:
                return None
            best_entry, best_diff = None, float('inf')
            for entry in self.buffer:
                diff = abs(entry.timestamp - target_ts)
                if diff < best_diff:
                    best_diff, best_entry = diff, entry
            if max_dt is not None and best_diff > max_dt:
                return None
            return best_entry

    def newest_timestamp(self) -> Optional[float]:
        with self._lock:
            return self.buffer[-1].timestamp if self.buffer else None


# ==============================================================================
# Main Data Collector
# ==============================================================================

class DataCollector:
    def __init__(self, output_dir, use_camera=True, high_freq_hz=100, low_freq_hz=10):
        self.output_dir = Path(output_dir)
        self.use_camera = use_camera
        self.high_freq_hz = high_freq_hz
        self.low_freq_hz = low_freq_hz
        self.stop_event = threading.Event()
        self.camera_buffer = RingBuffer(max_duration_seconds=RING_BUFFER_SIZE_S)
        self.robot_buffer = RingBuffer(max_duration_seconds=RING_BUFFER_SIZE_S)
        self._last_frame_id = -1
        self._last_cam_ts_hw = None
        self._last_image_hash = None

        self.episode_count = self._count_existing_episodes()

        self.stats = {k: 0 for k in ['frames_aligned', 'frames_skipped_cam', 'frames_skipped_robot', 
                                     'frames_skipped_align', 'frames_dedup_id', 'frames_dedup_ts', 'frames_dedup_hash']}

    def _count_existing_episodes(self) -> int:
        count = 0
        if self.output_dir.exists():
            for f in sorted(self.output_dir.glob("episode_*.npz")):
                count += 1
        return count

    def _is_duplicate_frame(self, frame_id: int, cam_ts_hw: float, image_hash: str) -> bool:
        if frame_id >= 0:
            if frame_id == self._last_frame_id:
                self.stats['frames_dedup_id'] += 1
                return True
            self._last_frame_id, self._last_cam_ts_hw, self._last_image_hash = frame_id, cam_ts_hw, image_hash
            return False
        if cam_ts_hw > 0:
            if self._last_cam_ts_hw is not None and abs(cam_ts_hw - self._last_cam_ts_hw) < CAM_TS_HW_THRESHOLD_S:
                self.stats['frames_dedup_ts'] += 1
                return True
            self._last_cam_ts_hw, self._last_image_hash = cam_ts_hw, image_hash
            return False
        if self._last_image_hash is not None and image_hash == self._last_image_hash:
            self.stats['frames_dedup_hash'] += 1
            return True
        self._last_image_hash = image_hash
        return False

    def _collect_camera_data(self, camera_manager: CameraManager):
        print("[Camera Thread] Started")
        while not self.stop_event.is_set():
            try:
                result = camera_manager.get_latest_frame_with_meta(convert_to_rgb=True, enable_debug_dump=True)
                if result is None or result[0] is None: time.sleep(0.001); continue
                image, metadata = result
                if self._is_duplicate_frame(metadata.get('frame_id', -1), metadata.get('cam_ts_hw', 0.0), metadata.get('image_hash', '')): continue
                cam_data = {'image': image, **metadata}
                self.camera_buffer.put(timestamp=metadata.get('cam_ts_mono', time.monotonic()), data=cam_data)
            except Exception as e:
                print(f"[Camera Thread] Error: {e}")
            time.sleep(0.001)
        print("[Camera Thread] Stopped")

    def _collect_robot_data(self, robot_manager: UR10Manager):
        print(f"[Robot Thread] Started ({self.high_freq_hz} Hz)")
        while not self.stop_event.is_set():
            t_start = time.monotonic()
            try:
                pose, gripper, ts = robot_manager.get_tcp_pose_with_ts()
                ft, _ = robot_manager.get_force_torque_with_ts()
                self.robot_buffer.put(timestamp=ts, data={'pose': pose, 'gripper_state': gripper, 'force_torque': ft})
            except Exception as e:
                print(f"[Robot Thread] Error: {e}")
            t_elapsed = time.monotonic() - t_start
            t_sleep = (1.0 / self.high_freq_hz) - t_elapsed
            if t_sleep > 0: time.sleep(t_sleep)
        print("[Robot Thread] Stopped")

    def _save_episode(self, episode_data: list, metadata: dict):
        if not episode_data: print("No data in episode to save."); return
        episode_idx = self.episode_count
        filepath = self.output_dir / f"episode_{episode_idx:04d}.npz"
        data_dict = {k: [] for k in ['image', 'cam_ts_hw', 'cam_ts_mono', 'cam_ts_recv', 'robot_ts', 'action_ts', 
                                     'step_ts', 'action', 'robot_pose', 'gripper_state', 'force_torque', 'color_space', 'frame_id']}
        for step in episode_data:
            for k, v in step.items():
                if k == 'robot_state':
                    data_dict['robot_pose'].append(v['pose']); data_dict['gripper_state'].append(int(round(v['gripper_state']))); data_dict['force_torque'].append(v['force_torque'])
                elif k in data_dict: data_dict[k].append(v)
        for key, value in data_dict.items(): data_dict[key] = np.array(value)
        np.savez_compressed(filepath, **data_dict)
        vis_dir = self.output_dir / f"episode_{episode_idx:04d}_vis"
        vis_dir.mkdir(exist_ok=True)
        # 保存纯图像（不添加任何文字信息）
        for i, step_data in enumerate(episode_data):
            # 直接保存原始图像，不添加任何文字
            image_rgb = step_data['image']
            # OpenCV 需要 BGR 格式保存，图像是 RGB，需要转换
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(vis_dir / f"frame_{i:05d}.png"), image_bgr)
        print(f"\n{'='*50}\n** Episode saved to: {filepath.name} **")
        metadata.update({'n_steps': len(episode_data), 'episode_idx': episode_idx})
        with open(self.output_dir / f"episode_{episode_idx:04d}_metadata.json", 'w') as f: json.dump(metadata, f, indent=2)
        self.episode_count += 1

    def _delete_last_episode(self):
        if self.episode_count > 0:
            episode_idx_to_delete = self.episode_count - 1
            npz_path = self.output_dir / f"episode_{episode_idx_to_delete:04d}.npz"
            meta_path = self.output_dir / f"episode_{episode_idx_to_delete:04d}_metadata.json"
            vis_path = self.output_dir / f"episode_{episode_idx_to_delete:04d}_vis"
            try:
                if npz_path.exists(): npz_path.unlink()
                if meta_path.exists(): meta_path.unlink()
                if vis_path.exists(): shutil.rmtree(vis_path)
                print(f"** Deleted last episode: {npz_path.name} and associated files **")
                self.episode_count -= 1
            except OSError as e:
                print(f"Error deleting files: {e}")
        else:
            print("No episodes to delete.")

    def _draw_visualization(self, image, is_recording, episode_len, episode_count, robot_state, vel_cmd, stats, cam_data=None, action=None, timestamps=None):
        vis_img = image.copy(); h, w, _ = vis_img.shape
        status_text = f"REC: {episode_len} steps" if is_recording else f"PAUSED ({episode_len} in buffer)"
        status_color = (0, 0, 255) if is_recording else (128, 128, 128)
        cv2.circle(vis_img, (30, 30), 15, status_color, -1)
        cv2.putText(vis_img, status_text, (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        cv2.putText(vis_img, f"Episodes: {episode_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # 左侧列：机器人当前状态 (STATE - 实际位置)
        if robot_state:
            pose = robot_state['pose']  # 当前位置和姿态
            gripper = robot_state['gripper_state']  # 当前夹爪状态 (0或1)
            ft = robot_state['force_torque']
            text_lines = [
                f"=== STATE (Current) ===",
                f"Pos: [{pose[0]:7.3f}, {pose[1]:7.3f}, {pose[2]:7.3f}]",
                f"Rot: [{pose[3]:7.3f}, {pose[4]:7.3f}, {pose[5]:7.3f}]",
                f"Gripper: {int(round(gripper))} ({'OPEN' if gripper > 0.5 else 'CLOSED'})",
                f"Force: [{ft[0]:6.1f}, {ft[1]:6.1f}, {ft[2]:6.1f}]",
                f"Torque: [{ft[3]:6.1f}, {ft[4]:6.1f}, {ft[5]:6.1f}]",
            ]
            for i, line in enumerate(text_lines):
                cv2.putText(vis_img, line, (10, 110 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # 右侧列：动作命令 (CMD - 目标速度/夹爪)
        if action is not None:
            lin_vel_cmd = action[:3]  # 线速度命令
            ang_vel_cmd = action[3:6]  # 角速度命令
            grip_cmd = action[6]  # 夹爪命令 (0或1)
            text_lines = [
                f"=== CMD (Target) ===",
                f"LinVel: [{lin_vel_cmd[0]:6.2f}, {lin_vel_cmd[1]:6.2f}, {lin_vel_cmd[2]:6.2f}]",
                f"AngVel: [{ang_vel_cmd[0]:6.2f}, {ang_vel_cmd[1]:6.2f}, {ang_vel_cmd[2]:6.2f}]",
                f"GripCmd: {int(round(grip_cmd))} ({'OPEN' if grip_cmd > 0.5 else 'CLOSED'})",
            ]
            for i, line in enumerate(text_lines):
                cv2.putText(vis_img, line, (w // 2, 110 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # 底部左侧：相机数据
        if cam_data:
            cam_ts_hw = cam_data.get('cam_ts_hw', 0)
            cam_ts_mono = cam_data.get('cam_ts_mono', 0)
            cam_ts_recv = cam_data.get('cam_ts_recv', 0)
            frame_id = cam_data.get('frame_id', -1)
            color_space = cam_data.get('color_space', 'N/A')
            text_lines = [
                f"=== CAMERA ===",
                f"Frame ID: {frame_id}",
                f"ColorSp: {color_space}",
                f"TS_HW:   {cam_ts_hw:.6f}",
                f"TS_MONO: {cam_ts_mono:.6f}",
                f"TS_RECV: {cam_ts_recv:.6f}",
            ]
            for i, line in enumerate(text_lines):
                cv2.putText(vis_img, line, (10, h - 145 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1)

        # 底部右侧：时间戳和对齐统计
        if timestamps:
            robot_ts = timestamps.get('robot_ts', 0)
            action_ts = timestamps.get('action_ts', 0)
            time_diff = abs(cam_ts_mono - robot_ts) if cam_data else 0
            text_lines = [
                f"=== TIMESTAMPS ===",
                f"Robot TS:  {robot_ts:.6f}",
                f"Action TS: {action_ts:.6f}",
                f"Cam-Robot dt: {time_diff*1000:.1f}ms",
                f"",
                f"=== ALIGN STATS ===",
                f"OK: {stats['frames_aligned']} Skip: {stats['frames_skipped_align']}",
            ]
            for i, line in enumerate(text_lines):
                cv2.putText(vis_img, line, (w // 2, h - 145 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)

        return vis_img

    def run(self):
        print("Initializing hardware...")
        camera = CameraManager(**CONFIG['camera']) if self.use_camera else None
        robot = UR10Manager(robot_ip=CONFIG['robot']['ip'], gripper_ip=CONFIG['robot'].get('gripper_ip'), control_freq=self.high_freq_hz)
        gamepad = GamepadHandler(config=CONFIG['gamepad'])
        with robot, (camera or robot):
            print("Hardware initialized. Starting collector threads...")
            robot_thread = threading.Thread(target=self._collect_robot_data, args=(robot,), daemon=True); robot_thread.start()
            camera_thread = threading.Thread(target=self._collect_camera_data, args=(camera,), daemon=True) if self.use_camera and camera else None
            if camera_thread: camera_thread.start()
            if self.episode_count > 0: print(f"\nFound {self.episode_count} existing episode(s) in {self.output_dir}")
            print("\nMain loop running... Press BACK to exit.")
            episode_data, is_recording, episode_start_time = [], False, None
            target_gripper_state = robot.get_tcp_pose()[1]
            is_resetting = False
            win_name = "Data Collection"; cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            time.sleep(0.5); print("Ready!\n")
            while not self.stop_event.is_set():
                loop_start_ts = time.monotonic()
                gamepad.update()
                control_buttons = gamepad.get_control_buttons()
                if control_buttons['exit']: break
                if control_buttons['toggle_recording']:
                    is_recording = not is_recording
                    if is_recording: episode_start_time = time.time()
                    print(f"** Recording {'STARTED' if is_recording else 'PAUSED'} **"); time.sleep(0.3)
                if control_buttons['save_episode']:
                    if episode_data:
                        metadata = {'duration_s': time.time() - episode_start_time if episode_start_time else 0, **CONFIG['camera']}
                        self._save_episode(episode_data, metadata)
                        episode_data, self.stats = [], {k: 0 for k in self.stats}
                        is_recording = False
                        print("** Recording stopped. Press START to record next episode. **")
                    else: print("No data to save.")
                    time.sleep(0.3)
                if control_buttons.get('reset_pose'):
                    is_resetting = True
                    print("** Resetting to initial pose... **"); robot.set_tcp_speed([0]*6); time.sleep(0.5)
                    try:
                        robot.robot_controller.rtde_c.stopScript(); time.sleep(1.0)
                        print("  RTDE script restarted")
                    except Exception as e:
                        print(f"  Warning: stopScript failed: {e}")
                    robot.move_to_pose_sync(np.array(CONFIG['robot']['initial_pose']), timeout=10.0)
                    is_resetting = False
                    print("** Pose reset complete. **"); time.sleep(0.3)
                if control_buttons.get('delete_last'):
                    self._delete_last_episode(); time.sleep(0.3)
                if not is_resetting:
                    lin_vel, ang_vel = gamepad.get_teleop_velocity()
                    robot.set_tcp_speed(np.concatenate([lin_vel, ang_vel]))
                gripper_input = gamepad.get_gripper_input()
                if gripper_input is not None:
                    robot.set_gripper(gripper_input); target_gripper_state = gripper_input
                action_ts = time.monotonic()
                if self.use_camera and camera:
                    newest_cam_ts = self.camera_buffer.newest_timestamp()
                    if newest_cam_ts is None: time.sleep(0.01); continue
                    cam_entry = self.camera_buffer.get_nearest(newest_cam_ts)
                    if cam_entry is None: self.stats['frames_skipped_cam'] += 1; time.sleep(0.01); continue
                    cam_ts_mono, cam_data = cam_entry.timestamp, cam_entry.data
                    robot_entry = self.robot_buffer.get_nearest(cam_ts_mono, max_dt=ALIGNMENT_THRESHOLD_S)
                    if robot_entry is None: self.stats['frames_skipped_align'] += 1; time.sleep(0.01); continue
                    robot_ts, robot_state = robot_entry.timestamp, robot_entry.data
                else:
                    newest_robot_ts = self.robot_buffer.newest_timestamp()
                    if newest_robot_ts is None: time.sleep(0.01); continue
                    robot_entry = self.robot_buffer.get_nearest(newest_robot_ts)
                    if robot_entry is None: time.sleep(0.01); continue
                    robot_ts, robot_state = robot_entry.timestamp, robot_entry.data
                    cam_data = {'image': np.zeros((CONFIG['camera']['height'], CONFIG['camera']['width'], 3), dtype=np.uint8), 'color_space': 'RGB', 'frame_id': -1, 'cam_ts_mono': robot_ts}
                # 构建动作命令和时间戳（用于显示）
                action = np.concatenate([lin_vel, ang_vel, [target_gripper_state]])
                timestamps = {'robot_ts': robot_ts, 'action_ts': action_ts}
                if is_recording:
                    step_data = {'action': action, 'robot_state': robot_state, 'action_ts': action_ts, 'robot_ts': robot_ts, **cam_data}
                    episode_data.append(step_data)
                    self.stats['frames_aligned'] += 1
                vis_img = self._draw_visualization(cam_data['image'], is_recording, len(episode_data), self.episode_count, robot_state, lin_vel, self.stats, cam_data, action, timestamps)
                cv2.imshow(win_name, vis_img)
                if cv2.waitKey(1) & 0xFF == 27: break
                loop_elapsed = time.monotonic() - loop_start_ts
                t_sleep = (1.0 / self.low_freq_hz) - loop_elapsed
                if t_sleep > 0: time.sleep(t_sleep)
            if episode_data:
                print("\nUnsaved data detected. Saving final episode before exiting...")
                metadata = {'duration_s': time.time() - episode_start_time if episode_start_time else 0, 'reason_for_save': 'auto-save on exit', **CONFIG['camera']}
                self._save_episode(episode_data, metadata)
            print("\nShutting down...")
            self.stop_event.set()
            if camera_thread: camera_thread.join(timeout=2.0)
            robot_thread.join(timeout=2.0)
            cv2.destroyAllWindows()

@click.command()
@click.option('--output-dir', '-o', default=None, help="Directory to save collected data.")
@click.option('--no-camera', is_flag=True, default=False, help="Run without camera for testing teleop.")
def main(output_dir, no_camera):
    collection_config = CONFIG['data_collection']
    if output_dir is None: output_dir = collection_config['default_output_dir']
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Data will be saved to: {output_path.absolute()}")
    collector_kwargs = {'high_freq_hz': collection_config.get('high_freq_hz', 100), 'low_freq_hz': collection_config.get('low_freq_hz', 10)}
    collector = DataCollector(output_dir=output_path, use_camera=not no_camera, **collector_kwargs)
    try:
        collector.run()
    except KeyboardInterrupt:
        print("\nData collection interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        collector.stop_event.set()
        print("Shutting down.")

if __name__ == "__main__":
    main()

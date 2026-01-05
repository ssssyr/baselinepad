"""
Main script for running the UR10 robot arm with Diffusion Policy in the real world.
"""

import os
import sys
import time
import json
import numpy as np
import torch
import cv2
from pathlib import Path

# Add project root to the Python path to allow imports from other packages.
# This assumes 'main.py' is in 'baselinepad/real/' and the project root is 'baselinepad/'.
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

# --- Local Imports from the new structure ---
from configs.ur10_config import CONFIG
from hardware.camera_manager import CameraManager
from hardware.ur10_manager import UR10Manager

# Import DiffusionAgent from local real/ directory (for real robot deployment)
try:
    from agent import DiffusionAgent
except ImportError as e:
    print(f"Error: Failed to import DiffusionAgent from local agent.py.")
    print(f"Current directory: {Path(__file__).parent}")
    raise e

def build_agent():
    """Initializes the DiffusionAgent from the central configuration."""
    print("Building agent...")
    try:
        agent = DiffusionAgent(
            ckpt_path=CONFIG['model']['ckpt_path'],
            vae_path=CONFIG['model']['vae_path'],
            clip_path=CONFIG['model']['clip_path'],
            denoise_steps=CONFIG['model']['denoise_steps'],
            device_id=CONFIG['model']['gpu_id'],
            use_fp16=CONFIG['model'].get('use_fp16', False),
        )
        print("Agent built successfully.")
        return agent
    except Exception as e:
        print(f"Error building agent: {e}")
        print("Please ensure model paths in 'ur10_config.py' are correct.")
        raise

def transform_coordinates(camera_coords, cam_to_base_transform):
    """Transforms a 3D point from the camera's optical frame to the robot's base frame."""
    if camera_coords is None:
        return None
    # Create a homogeneous coordinate for the point
    point_in_camera_frame = np.append(camera_coords, 1)
    # Apply the transformation
    point_in_base_frame = np.array(cam_to_base_transform) @ point_in_camera_frame
    # Return the Cartesian coordinates
    return point_in_base_frame[:3]

def is_pose_safe(pose, workspace_limits=None):
    """
    Checks if a target pose is within the robot's safe workspace.

    Args:
        pose (np.ndarray): The target pose [x, y, z, rx, ry, rz].
        workspace_limits (dict): Optional custom workspace limits.

    Returns:
        bool: True if the pose is safe, False otherwise.
    """
    x, y, z = pose[:3]

    # Default UR10 workspace limits (in meters)
    # Adjust these values based on your actual robot setup
    default_limits = {
        'x_min': -0.27, 'x_max': 0.5,
        'y_min': 0.5, 'y_max': 1.1,
        'z_min': -0.1,  'z_max': 0.142,
    }

    limits = workspace_limits or default_limits

    # Check each axis
    if not (limits['x_min'] <= x <= limits['x_max']):
        print(f"警告：X坐标 {x:.3f} 超出范围 [{limits['x_min']}, {limits['x_max']}]")
        return False
    if not (limits['y_min'] <= y <= limits['y_max']):
        print(f"警告：Y坐标 {y:.3f} 超出范围 [{limits['y_min']}, {limits['y_max']}]")
        return False
    if not (limits['z_min'] <= z <= limits['z_max']):
        print(f"警告：Z坐标 {z:.3f} 超出范围 [{limits['z_min']}, {limits['z_max']}]")
        return False

    return True


def smooth_motion_planner(robot, target_xyz, target_gripper, curr_xyz, curr_gripper,
                          verbose=False):
    """
    类似仿真环境的 motion_planner，平滑地移动到目标位置。

    推理一次后，分多步平滑执行，避免"动一下卡一下"的问题。

    Args:
        robot: UR10Manager 实例
        target_xyz: 目标位置 [x, y, z]
        target_gripper: 目标夹爪状态 (0=closed, 1=open)
        curr_xyz: 当前位置 [x, y, z]
        curr_gripper: 当前夹爪状态
        verbose: 是否打印调试信息

    Returns:
        success: 是否成功到达目标
    """
    stage = 0  # 0=移动阶段, 1=抓取阶段
    grasp_moment = abs(float(target_gripper) - float(curr_gripper)) > 0.2
    motion_steps = 50  # 最多50步

    if grasp_moment and verbose:
        print("准备执行抓取动作...")

    for step in range(motion_steps):
        # 获取当前位置
        current_pose, current_gripper_val = robot.get_tcp_pose()
        curr_xyz = current_pose[:3]
        curr_gripper = float(current_gripper_val)

        if stage == 0:
            # ============ 移动阶段 ============
            delta = (target_xyz - curr_xyz).astype(np.float32)
            dist = float(np.linalg.norm(delta))

            # 到达目标位置？
            if dist < 0.005:  # 5mm 阈值
                if verbose:
                    print(f"  步骤 {step + 1}: 到达目标位置 (dist={dist:.4f}m)")
                if grasp_moment:
                    stage = 1  # 进入抓取阶段
                else:
                    return True  # 完成

            # 计算步进位移（速度控制）
            if dist > 1e-8:
                # 距离远用大速度，近用小速度
                velocity = 0.05 if dist > 0.03 else 0.01
                step_delta = (delta / dist) * velocity
                step_xyz = curr_xyz + step_delta
            else:
                step_xyz = curr_xyz

            # 保持当前姿态，只改变位置
            step_pose = np.concatenate([step_xyz, current_pose[3:]])

            if verbose and step % 10 == 0:
                print(f"  步骤 {step + 1}: dist={dist:.4f}m, 移动到 {np.round(step_xyz, 3)}")

            # 执行移动（超时1.5秒，避免误报）
            robot.move_to_pose_sync(step_pose, timeout=1.5)

        elif stage < 20:
            # ============ 抓取阶段 ============
            # 闭夹爪动作
            if target_gripper < 0.75:
                # Robot API: 1=open, 0=closed，需要转换
                # target_gripper < 0.75 表示要闭合
                robot.set_gripper(0.0)
                if verbose:
                    print(f"  步骤 {step + 1}: 执行抓取（闭合夹爪）")
            else:
                if verbose:
                    print(f"  步骤 {step + 1}: 夹爪保持张开")
            stage += 1
        else:
            break

        # 短暂延时，让机器人执行
        time.sleep(0.05)

    return True


def main():
    """Main control loop for real-world execution."""
    print("--- UR10 Real-World Deployment Script ---")
    
    camera = None
    robot = None

    try:
        # 1. Initialize Hardware
        print("\n1. Initializing Hardware...")
        camera = CameraManager(
            serial_number=CONFIG['camera']['serial_number'] or None, # Pass None to auto-detect
            width=CONFIG['camera']['width'],
            height=CONFIG['camera']['height'],
            fps=CONFIG['camera']['fps']
        )
        robot = UR10Manager(
            robot_ip=CONFIG['robot']['ip'],
            gripper_ip=CONFIG['robot'].get('gripper_ip')
        )
        
        with camera, robot:
            # 2. Load AI Agent
            print("\n2. Loading AI Agent...")
            agent = build_agent()

            # 3. Move to initial pose from config.json
            print("\n3. Moving to initial pose from config.json...")
            config_path = os.path.join(os.path.dirname(__file__), "scripts", "config.json")
            with open(config_path, "r") as f:
                teleop_config = json.load(f)
            initial_pose = np.array(teleop_config["robot"]["initial_pose"])
            print(f"Initial pose: {initial_pose}")

            if is_pose_safe(initial_pose):
                robot.move_to_pose_sync(initial_pose)
                # NOTE: Training data uses opposite encoding (0=open, 1=closed)
                # To match the training data's initial state (gripper=0=open),
                # we send 1.0 to the robot API which expects 1=open
                robot.set_gripper(1.0)  # Robot API: 1.0 = open
                print("Moved to initial pose successfully, gripper opened.")
            else:
                print("警告：初始位姿超出工作范围，跳过移动")

            print("\nInitialization complete. Starting main control loop.")
            cv2.namedWindow('UR10 Diffusion Policy - Current + 3 Predicted Frames', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('UR10 Diffusion Policy - Current + 3 Predicted Frames', 1400, 400)

            for step in range(CONFIG['task']['max_steps']):
                print(f"\n--- Step {step + 1}/{CONFIG['task']['max_steps']} ---")

                # a. Perception: Get latest data (使用带时间戳的方法，确保同步)
                rgb_image, cam_meta = camera.get_latest_frame_with_meta(enable_debug_dump=False)
                current_pose, current_gripper, robot_ts = robot.get_tcp_pose_with_ts()
                force_torque, ft_ts = robot.get_force_torque_with_ts()

                if rgb_image is None:
                    print("Warning: Failed to get image frame. Skipping step.")
                    time.sleep(0.1)
                    continue

                # 检查时间同步（图像和机器人位置的时间差应该很小）
                cam_ts = cam_meta['cam_ts_mono']
                time_diff = abs(cam_ts - robot_ts)
                if time_diff > 0.1:  # 超过100ms认为不同步
                    print(f"警告：图像和机器人位置时间差过大 ({time_diff*1000:.1f}ms)，可能存在同步问题")
                    print(f"  相机时间戳: {cam_ts:.3f}s, 机器人时间戳: {robot_ts:.3f}s")

                # b. Decision: Get action from policy
                # The agent needs state information in the correct format.
                # This is a placeholder for the actual state construction based on run_metaworld.py
                robot_state = np.concatenate([current_pose[:3], [current_gripper]])
                
                with torch.no_grad():
                    # DiffusionAgent.action signature: action(self, text, rgb=None, depth=None, state=None, force=None)
                    samples, sample_a, _ = agent.action(
                        text=CONFIG['task']['task_instruction'],
                        rgb=rgb_image,
                        depth=None,
                        state=robot_state,
                        force=force_torque,
                    )

                # c. Visualization: Decode and display predictions
                # samples contains predicted latents for future frames
                vis_image = agent.decode(rgb_image, samples, prefix="", save=False)
                # Add step counter on image
                vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
                cv2.putText(vis_image, f'Step: {step + 1}/{CONFIG["task"]["max_steps"]}',
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('UR10 Diffusion Policy - Current + 3 Predicted Frames', vis_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("User requested quit.")
                    break
                
                # Extract target from agent's prediction (使用第2帧，和仿真一致)
                # Use default action_scale if not present in checkpoint
                action_scale = getattr(agent.args, 'action_scale', 1.0)

                # 将 (1, 1, 12) reshape 成 (3, 4)，然后取第2帧
                action_steps = getattr(agent.args, 'action_steps', 3)
                action_dim = getattr(agent.args, 'action_dim', 4)
                a_seq = sample_a.reshape(action_steps, action_dim)

                target = a_seq[1] / action_scale  # 第2帧 (索引1)
                target_xyz, target_gripper = target[:3], float(target[3])

                # Note: Model outputs are already in robot base frame (xyz, gripper)
                # No camera_to_base transformation needed

                # Apply gripper threshold (clamp) based on training logic
                # During training: gripper < threshold → close, gripper >= threshold → open
                # This matches the motion_planner logic in run_metaworld.py
                gripper_threshold = CONFIG['task'].get('gripper_threshold', 0.75)
                if target_gripper < gripper_threshold:
                    target_gripper = 0.0  # Fully closed
                else:
                    target_gripper = 1.0  # Fully open

                print(f"Gripper: model_output={target[3]:.3f} → clamped={target_gripper:.1f} (threshold={gripper_threshold})")

                # Combine with orientation from current pose to form target pose
                target_pose = np.concatenate([target_xyz, current_pose[3:]])

                # c. Safety Check: Verify pose is within workspace
                # 注意：有 smooth_motion_planner 分步执行，不需要位移限制
                if not is_pose_safe(target_pose):
                    print(f"警告：目标位姿超出工作范围，跳过此步骤")
                    time.sleep(0.1)
                    continue

                # e. Execution: 使用平滑运动规划器（类似仿真环境）
                print(f"开始平滑移动到目标位置...")
                smooth_motion_planner(
                    robot=robot,
                    target_xyz=target_xyz,
                    target_gripper=target_gripper,
                    curr_xyz=current_pose[:3],
                    curr_gripper=current_gripper,
                    verbose=True
                )
                print("平滑移动完成\n")

    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Shutting down.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        # Cleanup OpenCV windows
        cv2.destroyAllWindows()
        # Cleanup is handled by 'with' statement context managers
        print("\nScript finished. Hardware should be disconnected.")


if __name__ == "__main__":
    main()

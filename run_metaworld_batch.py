import numpy as np
try:
    import mediapy
    HAS_MEDIAPY = True
except ImportError:
    HAS_MEDIAPY = False
from PIL import Image
import json
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import cv2
import random
import torch
import time
from tabulate import tabulate

os.environ["MUJOCO_GL"] = "egl"
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE, ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)

from evaluation.agent import DiffusionAgent
from evaluation.run_cfg import INSTRUCTIONS, META_CONFIG

def set_random_seed(seed=None):
    """Set random seed for reproducibility or randomness"""
    if seed is None:
        seed = int(time.time() * 1000000) % (2**32)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return seed

def add_bound(rgb, color="red"):
    width = 10
    c = 0 if color == "red" else 1
    rgb[:width, :, 1:3] = 100
    rgb[:width, :, c] = 255

    rgb[-width:, :, 1:3] = 100
    rgb[-width:, :, c] = 255

    rgb[:, :width, 1:3] = 100
    rgb[:, :width, c] = 255
    rgb[:, -width:, 1:3] = 100
    rgb[:, -width:, c] = 255
    return rgb

def merge_img(obs, predict, img_word):
    img_1 = cv2.resize(obs, (256, 256), interpolation=cv2.INTER_AREA)
    image = np.concatenate((add_bound(img_1, color="green"), predict), axis=1)
    image = np.concatenate((img_word, image), axis=0)
    return image

def plot_word():
    from PIL import Image, ImageDraw, ImageFont
    fnt_titile = ImageFont.truetype("evaluation/TIMES.ttf", int(600/20))
    img_word = Image.new('RGB', (256*4, 40), color='white')
    draw = ImageDraw.Draw(img_word)
    task = "Observations"
    task2 = "Predictions"
    draw.text((50, 10), task, font=fnt_titile, fill='green')
    draw.text((572, 10), task2, font=fnt_titile, fill='red')

    return img_word

# motion planner for metaworld tasks
def motion_planner(target_xyz, target_gripper, curr_xyz, curr_gripper, env, image_3, thirdview, predict_img=None, img_word=None, save_video=False):
    # a simple motion planner to reach the target pose, starting from the current pose
    # stage (0) Move to the target pose with a constant velocity (0.6 or 0.3)
    # stage (1) If grasp, then close the gripper
    stage = 0
    grasp_moment = False

    # check whether the gripper should closed
    if np.abs(target_gripper - curr_gripper) > 0.2:
        grasp_moment = True
        print("prepare grasp!!")

    # start motion planner with max 50 steps
    motion_steps = 50
    success_flag = False
    for i in range(motion_steps):
        a = -np.ones(4)
        if stage == 0:  # moving to target pose with a constant velocity
            if target_gripper < 0.75 and curr_gripper < 0.75:
                a[3] = 0.7
            velocity = 0.6 if np.linalg.norm(target_xyz-curr_xyz) > 0.03 else 0.3
            a[:3] = (target_xyz-curr_xyz)/np.linalg.norm(target_xyz-curr_xyz)*velocity

            # step the env
            obs, r, done, info = env.step(a)
            if save_video:
                img = env.render(offscreen=True, camera_name=thirdview, resolution=[224,224], depth=False)
                if predict_img is not None:
                    img_all = merge_img(img, predict_img, img_word)
                image_3.append(img_all)
            curr_xyz, curr_gripper = env._get_obs()[:3], env._get_obs()[3]

            # early break if env already reports success
            if info.get("success", 0):
                success_flag = True
                break

            # check if the target pose is reached
            if stage == 0 and np.linalg.norm(target_xyz-curr_xyz) < 0.005:
                stage += 1 if grasp_moment else motion_steps

        elif stage < 20:  # grasping stage
            if target_gripper < 0.82:
                a = np.array([0, 0, 0, 0.7])  # close the gripper
                obs, r, done, info = env.step(a)
                if save_video:
                    img = env.render(offscreen=True, camera_name=thirdview, resolution=[224,224], depth=False)
                    if predict_img is not None:
                        img_all = merge_img(img, predict_img, img_word)
                    image_3.append(img_all)
                if info.get("success", 0):
                    success_flag = True
                    break
                stage += 1
            else:
                break
        else:
            break
    # If success happened mid-loop, ensure the returned info reflects it
    if success_flag:
        info["success"] = 1.0

    if save_video:
        return info, img
    else:
        return info, None

def run_single_task(agent, task, selected_id, INSTRUCTIONS, META_CONFIG, rollout_num=10, save_video=False):
    """Run a single task for specified number of rollouts and return success count"""
    thirdview = META_CONFIG['thirdview_camera']
    use_depth = META_CONFIG['use_depth']
    success_count = 0

    # For visualization (only if save_video is True)
    img_word = plot_word() if save_video else None

    print(f"\n{'='*60}")
    print(f"Starting Task: {task} (ID: {selected_id})")
    print(f"{'='*60}")

    for traj_idx in range(rollout_num):
        current_seed = set_random_seed()
        print(f"task name: {task}, traj_idx: {traj_idx}, seed: {current_seed}")

        env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task + "-goal-observable"]
        env = env_cls(seed=selected_id + 100)

        image_3 = []
        obs = env.reset()

        if save_video:
            img = env.render(offscreen=True, camera_name=thirdview, resolution=[224,224], depth=False)

        for plan_step in tqdm(range(META_CONFIG['max_steps']), desc=f"  {task} traj {traj_idx}"):
            state = obs
            if save_video:
                rgb = img
            else:
                rgb = env.render(offscreen=True, camera_name=thirdview, resolution=[224,224], depth=False)
            depth = None if not META_CONFIG['use_depth'] else depth
            text = INSTRUCTIONS[task]
            state = env._get_obs()[:4]
            curr_xyz, curr_gripper = state[:3], state[3]

            # plan next target with PAD agent
            samples, sample_a, sample_depth = agent.action(text, rgb, depth, state)

            predict_img = None
            if save_video and META_CONFIG['visualize_prediction']:
                predict_img = agent.decode_rgb(rgb, samples)
                predict_img = add_bound(predict_img)

            # Use the first predicted step as the immediate target pose
            if agent.args.action_steps > 0 and sample_a is not None:
                a_seq = sample_a.reshape(agent.args.action_steps, agent.args.action_dim)
                target_step = 1 if agent.args.action_steps > 1 else 0
                target = a_seq[target_step] / agent.args.action_scale
                target_xyz, target_gripper = target[:3], target[3]
            else:
                target = sample_a / agent.args.action_scale
                target_xyz, target_gripper = target[0, 0, :3], target[0, 0, 3]

            # motion planner to reach the target pose
            info, img = motion_planner(target_xyz, target_gripper, curr_xyz, curr_gripper, env, image_3, thirdview,
                                       predict_img=predict_img, img_word=img_word, save_video=save_video)

            if info['success']:
                print(f"  {task} traj_idx {traj_idx} - SUCCESS!")
                success_count += 1
                break

        # Save video if enabled
        if save_video:
            if not HAS_MEDIAPY:
                print("Warning: mediapy not available, skipping video save")
                continue
            video_dir = META_CONFIG['video_dir']
            os.makedirs(f'{video_dir}/rollout_metaworld', exist_ok=True)
            mediapy.write_video(f'{video_dir}/rollout_metaworld/{task}_{traj_idx}.mp4', image_3, fps=20)

    return success_count

def main():
    # Configuration
    ROLLOUT_NUM = 10  # Change from 5 to 10
    SAVE_VIDEO = False  # Don't save videos for batch testing

    # Get all 50 tasks from INSTRUCTIONS
    task_list = list(INSTRUCTIONS.keys())
    print(f"Total tasks to run: {len(task_list)}")
    print(f"Rollouts per task: {ROLLOUT_NUM}")
    print(f"Save videos: {SAVE_VIDEO}")

    # build agent
    # Override gpu_id to 0 since we use CUDA_VISIBLE_DEVICES
    agent = DiffusionAgent(
        ckpt_path=META_CONFIG['ckpt_path'],
        vae_path=META_CONFIG['vae_path'],
        clip_path=META_CONFIG['clip_path'],
        denoise_steps=META_CONFIG['denoise_steps'],
        device_id=0  # Always use 0 when CUDA_VISIBLE_DEVICES is set
    )

    # Store results
    results = []

    # Run all tasks
    print("\n" + "="*80)
    print("STARTING BATCH TESTING")
    print("="*80)

    for selected_id, task in enumerate(task_list):
        success_count = run_single_task(
            agent, task, selected_id, INSTRUCTIONS, META_CONFIG,
            rollout_num=ROLLOUT_NUM,
            save_video=SAVE_VIDEO
        )
        success_rate = success_count / ROLLOUT_NUM
        results.append({
            'ID': selected_id,
            'Task': task,
            'Success': success_count,
            'Total': ROLLOUT_NUM,
            'Rate': f"{success_rate:.1%}"
        })

    # Print results table
    print("\n" + "="*80)
    print("BATCH TESTING RESULTS")
    print("="*80)

    # Print table using tabulate
    headers = ['ID', 'Task', 'Success/Total', 'Success Rate']
    table_data = [[r['ID'], r['Task'], f"{r['Success']}/{r['Total']}", r['Rate']] for r in results]
    print(tabulate(table_data, headers=headers, tablefmt='grid'))

    # Calculate and print overall statistics
    total_success = sum(r['Success'] for r in results)
    total_trials = sum(r['Total'] for r in results)
    overall_rate = total_success / total_trials

    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    print(f"Total tasks: {len(task_list)}")
    print(f"Total trials: {total_trials}")
    print(f"Total successes: {total_success}")
    print(f"Overall success rate: {overall_rate:.2%}")
    print("="*80)

    # Save results to file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"batch_results_{timestamp}.txt"
    with open(results_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BATCH TESTING RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(tabulate(table_data, headers=headers, tablefmt='grid') + "\n\n")
        f.write("="*80 + "\n")
        f.write("OVERALL STATISTICS\n")
        f.write("="*80 + "\n")
        f.write(f"Total tasks: {len(task_list)}\n")
        f.write(f"Total trials: {total_trials}\n")
        f.write(f"Total successes: {total_success}\n")
        f.write(f"Overall success rate: {overall_rate:.2%}\n")
        f.write("="*80 + "\n")

    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    main()

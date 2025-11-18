
import mediapy
import numpy as np
from PIL import Image
import json
from PIL import Image
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import cv2
import random
import torch
import time

os.environ["MUJOCO_GL"] = "egl"
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE, ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)

from evaluation.agent import DiffusionAgent
from evaluation.run_cfg import INSTRUCTIONS, META_CONFIG

def set_random_seed(seed=None):
    """Set random seed for reproducibility or randomness"""
    if seed is None:
        seed = int(time.time() * 1000000) % (2**32)  # Use microsecond timestamp as seed
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"🎲 Set random seed: {seed}")
    return seed 

def add_bound(rgb,color="red"):
    width=10
    c = 0 if color=="red" else 1
    rgb[:width,:,1:3]=100
    rgb[:width,:,c]=255
    
    rgb[-width:,:,1:3]=100
    rgb[-width:,:,c]=255
    
    rgb[:,:width,1:3]=100
    rgb[:,:width,c]=255
    rgb[:,-width:,1:3]=100
    rgb[:,-width:,c]=255
    return rgb

def merge_img(obs, predict,img_word):
    img_1 = cv2.resize(obs, (256,256), interpolation=cv2.INTER_AREA)
    image = np.concatenate((add_bound(img_1,color="green"), predict),axis=1)
    image = np.concatenate((img_word,image),axis=0)
    return image

def plot_word():
    from PIL import Image, ImageDraw, ImageFont
    fnt_titile = ImageFont.truetype("evaluation/TIMES.ttf", int(600/20))
    img_word = Image.new('RGB', (256*4,40), color = 'white')
    draw = ImageDraw.Draw(img_word)
    task = "Observations"
    task2 = "Predictions"
    draw.text((50,10), task, font=fnt_titile, fill='green')
    draw.text((572,10), task2, font=fnt_titile, fill='red')

    return img_word

# motion planner for metaworld tasks
def motion_planner(target_xyz, target_gripper, curr_xyz, curr_gripper, env, image_3, thirdview, predict_img=None, img_word=None):
    # a simple motion planner to reach the target pose, starting from the current pose
    # stage (0) Move to the target pose with a constant velocity (0.6 or 0.3)
    # stage (1) If grasp, then close the gripper
    stage = 0 
    grasp_moment = False

    # check whether the gripper should closed
    if np.abs(target_gripper - curr_gripper) > 0.2:
        # target_xyz[2] -=0.04 if target_xyz[2]>0.1 else 0.01
        grasp_moment = True
        print("prepare grasp!!")
    
    # start motion planner with max 50 steps
    motion_steps = 50
    for i in range(motion_steps):                
        a = -np.ones(4)
        if stage == 0: # moving to target pose with a constant velocity
            if target_gripper < 0.75 and curr_gripper < 0.75:
                a[3] = 0.7
            velocity = 0.6 if np.linalg.norm(target_xyz-curr_xyz) > 0.03 else 0.3
            a[:3] = (target_xyz-curr_xyz)/np.linalg.norm(target_xyz-curr_xyz)*velocity

            # step the env
            obs, r, done, info = env.step(a)
            img = env.render(offscreen=True, camera_name=thirdview, resolution=[224,224],depth=False)
            if predict_img is not None:
                img_all= merge_img(img,predict_img,img_word)
            image_3.append(img_all)
            curr_xyz,curr_gripper = env._get_obs()[:3], env._get_obs()[3]

            # check if the target pose is reached
            if stage==0 and (np.linalg.norm(target_xyz-curr_xyz) < 0.005 or curr_xyz[2]<0.05):
                stage += 1 if grasp_moment else motion_steps

        elif stage < 20: # grasping stage
            if target_gripper < 0.82:
                a = np.array([0,0,0,0.7]) # close the gripper
                obs, r, done, info = env.step(a)
                img = env.render(offscreen=True, camera_name=thirdview, resolution=[224,224],depth=False)
                if predict_img is not None:
                    img_all= merge_img(img,predict_img,img_word)
                image_3.append(img_all)
                stage += 1
            else:
                break
        else:
            break
    return info,img


# rollout tasks
task_list = [key for key in INSTRUCTIONS.keys()]
task_list = META_CONFIG['task_list']
success_num = np.zeros(len(task_list))
thirdview = META_CONFIG['thirdview_camera']
firstview = META_CONFIG['firstview_camera']
ckpt_path = META_CONFIG['ckpt_path']
use_depth = META_CONFIG['use_depth']

# build agent
agent = DiffusionAgent(ckpt_path=ckpt_path,vae_path=META_CONFIG['vae_path'], clip_path=META_CONFIG['clip_path'], denoise_steps=META_CONFIG['denoise_steps'])
if META_CONFIG['visualize_prediction']:
    img_word = plot_word()
else:
    img_word = None
    predict_img = None

# start rollout
for selected_id, task in enumerate(task_list):
    env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task+"-goal-observable"]
    env = env_cls(seed=selected_id+100)

    for traj_idx in range(META_CONFIG['rollout_num']):
        # Set different random seed for each trajectory to ensure diverse sampling
        current_seed = set_random_seed()
        print("task name", task, 'traj_idx', traj_idx)
        image_3 = []

        obs = env.reset()
        img = env.render(offscreen=True, camera_name=thirdview, resolution=[224,224],depth=False)
        # image_3.append(img)
        for plan_step in tqdm(range(META_CONFIG['max_steps'])):
            # prepare input data
            grasp_moment = False
            state = obs
            rgb = img
            depth =  depth if use_depth else None
            text = INSTRUCTIONS[task]
            # state = env._get_obs()[:4]
            state = env._get_obs()[:4]

            # plan next target with PAD agent
            samples,sample_a,sample_depth = agent.action(text, rgb, depth, state)

            if META_CONFIG['visualize_prediction']:
                predict_img = agent.decode_rgb(rgb, samples) # np.array shape (256,256*3)
                predict_img = add_bound(predict_img)

            target = sample_a/agent.args.action_scale
            target_xyz, target_gripper = target[0,0,:3], target[0,0,3] # target pose
            curr_xyz, curr_gripper = state[:3], state[3] # current pose
            
            # motion planner to reach the target pose, starting from the current pose
            info, img = motion_planner(target_xyz, target_gripper, curr_xyz, curr_gripper, env, image_3, thirdview, predict_img=predict_img, img_word=img_word)

            # 针对所有任务的详细进度显示
            obj_to_target = info.get('obj_to_target', 0.0)
            near_object = info.get('near_object', 0.0)
            grasp_success = info.get('grasp_success', 0.0)
            grasp_reward = info.get('grasp_reward', 0.0)
            in_place_reward = info.get('in_place_reward', 0.0)

            # 计算进度百分比（对于有obj_to_target的任务）
            if obj_to_target > 0:
                progress_pct = max(0, min(100, (1.0 - obj_to_target/0.15) * 100))  # 假设初始距离约0.15米
                print(f"Step {plan_step+1}: success={info['success']:.1f}, obj_to_target={obj_to_target:.4f}m ({progress_pct:.1f}%)")
            else:
                print(f"Step {plan_step+1}: success={info['success']:.1f}, obj_to_target=N/A")

            print(f"  └─ near_object={near_object:.3f}, grasp_success={grasp_success:.3f}, grasp_reward={grasp_reward:.3f}, in_place_reward={in_place_reward:.3f}")

            # 详细的成功判断和反馈
            if plan_step == META_CONFIG['max_steps'] - 1:  # 最后一步
                if info['success']:
                    print(f"🎉 {task} traj_idx {traj_idx} FULL SUCCESS!")
                    if obj_to_target > 0:
                        print(f"   Final obj_to_target: {obj_to_target:.4f}m")
                elif obj_to_target > 0:
                    # 根据任务类型设置不同的进度阈值
                    if task.startswith('button-press'):
                        threshold = 0.06
                    elif task.startswith('basketball'):
                        threshold = 0.08
                    else:
                        threshold = 0.05

                    if obj_to_target <= threshold:
                        print(f"👍 {task} traj_idx {traj_idx} GOOD PROGRESS (obj_to_target: {obj_to_target:.4f}m)")
                    else:
                        print(f"⚠️ {task} traj_idx {traj_idx} Need more progress (obj_to_target: {obj_to_target:.4f}m)")
                        print(f"   Target: <=0.04m, Current progress: {threshold - obj_to_target:.4f}m away")
                else:
                    print(f"📊 {task} traj_idx {traj_idx} COMPLETED (success={info['success']:.1f})")

            if info['success']:
                print(task, traj_idx, 'success')
                success_num[selected_id] += 1
                break
        
        # save video
        # ckpt_path = ckpt_path.split('.')[0]
        video_dir = META_CONFIG['video_dir']
        os.makedirs(f'{video_dir}/rollout_metaworld', exist_ok=True)
        mediapy.write_video(f'{video_dir}/rollout_metaworld/{task}_{traj_idx}.mp4', image_3, fps=20)
    

for i in range(len(task_list)):
    print(task_list[i], success_num[i])

# running command:
# CUDA_VISIBLE_DEVICES=1 python sample_pose.py
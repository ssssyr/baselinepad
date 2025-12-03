import mediapy
import numpy as np
from PIL import Image
import json
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
        seed = int(time.time() * 1000000) % (2**32)
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"🎲 Set random seed: {seed}")
    return seed 

def add_bound(rgb, color="red"):
    width = 10
    c = 0 if color == "red" else 1
    rgb[:width, :, 1:3] = 100
    rgb[:width, :, c] = 255
    
    rgb[-width:, :, 1:3] = 100
    rgb[-width, :, c] = 255
    
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
    # 确保字体文件存在，否则可能报错，这里加个简单的容错或者请确保路径正确
    try:
        fnt_titile = ImageFont.truetype("evaluation/TIMES.ttf", int(600/20))
    except:
        # fallback default font
        fnt_titile = ImageFont.load_default()
        
    img_word = Image.new('RGB', (256*4, 40), color='white')
    draw = ImageDraw.Draw(img_word)
    task = "Observations"
    task2 = "Predictions"
    draw.text((50, 10), task, font=fnt_titile, fill='green')
    draw.text((572, 10), task2, font=fnt_titile, fill='red')

    return img_word

# [MODIFIED] 重构为单步动作计算函数，不再包含循环和env.step
def calculate_single_action(target_xyz, target_gripper, curr_xyz, curr_gripper):
    """
    根据 Diffusion 预测的 target 计算当前这一步的动作 (velocity + gripper)。
    实现纯粹的 P-Controller 逻辑。
    """
    a = -np.ones(4) # 默认张开夹爪 (-1)

    # 1. 位置控制 (Position Control)
    # 使用简单的 P 控制器：动作 = 方向 * 速度
    dist = np.linalg.norm(target_xyz - curr_xyz)
    
    # 保持原有的启发式速度逻辑：离得远就快点，近了就慢点
    velocity = 0.6 if dist > 0.03 else 0.3
    
    # 防止除以零
    if dist < 1e-6:
        a[:3] = 0
    else:
        a[:3] = (target_xyz - curr_xyz) / dist * velocity

    # 2. 夹爪控制 (Gripper Control)
    # 如果 Diffusion 预测的夹爪值小于阈值（通常是 0.75 或 0.5），则执行闭合
    # 注意：MetaWorld 中正值通常是闭合，具体取决于环境配置，这里沿用原代码的 0.7
    if target_gripper < 0.75: 
        a[3] = 0.7 
        
    return a


# rollout tasks
task_list = META_CONFIG['task_list']
success_num = np.zeros(len(task_list))
thirdview = META_CONFIG['thirdview_camera']
firstview = META_CONFIG['firstview_camera']
ckpt_path = META_CONFIG['ckpt_path']
use_depth = META_CONFIG['use_depth']

# build agent
agent = DiffusionAgent(
    ckpt_path=ckpt_path,
    vae_path=META_CONFIG['vae_path'],
    clip_path=META_CONFIG['clip_path'],
    denoise_steps=META_CONFIG['denoise_steps'],
    device_id=META_CONFIG.get('gpu_id', 0)
)

if META_CONFIG['visualize_prediction']:
    img_word = plot_word()
else:
    img_word = None
    predict_img = None

# [MODIFIED] 增加最大步数，因为现在是单步执行，不再是 chunking 执行
# 原来的 30 步是指 "30次规划"，每次规划可能会跑 50 步物理仿真。
# 现在改为 RHC，总步数需要增加以覆盖同样的物理时间。
MAX_RHC_STEPS = 500 

# start rollout
for selected_id, task in enumerate(task_list):
    # 实例化环境
    env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task+"-goal-observable"]
    env = env_cls(seed=selected_id+100)

    for traj_idx in range(META_CONFIG['rollout_num']):
        current_seed = set_random_seed()
        print(f"task name: {task}, traj_idx: {traj_idx}")
        image_3 = []

        obs = env.reset()
        # 初始化渲染
        img = env.render(offscreen=True, camera_name=thirdview, resolution=[224, 224], depth=False)
        
        success_flag = False
        
        # [MODIFIED] 主控制循环：Receding Horizon Control
        # 每次循环：观测 -> 预测 -> 执行一步
        for step_idx in tqdm(range(MAX_RHC_STEPS), desc="RHC Steps"):
            # 1. 准备输入数据
            state = obs[:4] # (x, y, z, gripper)
            curr_xyz, curr_gripper = state[:3], state[3]
            
            rgb = img
            depth = depth if use_depth else None
            text = INSTRUCTIONS[task]

            # 2. PAD Agent 预测 (Inference)
            # 注意：这里每次都重新预测，利用了最新的观测信息
            samples, sample_a, sample_depth = agent.action(text, rgb, depth, state)

            # 可视化预测结果
            if META_CONFIG['visualize_prediction']:
                predict_img = agent.decode_rgb(rgb, samples) 
                predict_img = add_bound(predict_img)
                # 合并当前观测和预测图像
                img_all = merge_img(img, predict_img, img_word)
                image_3.append(img_all)
            else:
                image_3.append(img) # 仅保存观测

            # 3. 解析预测动作作为当前目标 (Target)
            # 我们只取预测序列的第一个点 (T+k) 作为局部目标
            if agent.args.action_steps > 0 and sample_a is not None:
                a_seq = sample_a.reshape(agent.args.action_steps, agent.args.action_dim)
                target = a_seq[0] / agent.args.action_scale # 取第一帧
                target_xyz, target_gripper = target[:3], target[3]
            else:
                # 兼容旧逻辑
                target = sample_a / agent.args.action_scale
                target_xyz, target_gripper = target[0, 0, :3], target[0, 0, 3]

            # 4. 计算控制指令 (Low-level Control)
            # 根据当前位置和预测的目标位置，计算这一步的 action
            action = calculate_single_action(target_xyz, target_gripper, curr_xyz, curr_gripper)

            # 5. 执行一步物理仿真 (Execute)
            obs, r, done, info = env.step(action)
            img = env.render(offscreen=True, camera_name=thirdview, resolution=[224, 224], depth=False)

            # 6. 状态监控与成功判定
            obj_to_target = info.get('obj_to_target', 0.0)
            
            # 如果环境判定成功，提前退出
            if info.get("success", 0):
                success_flag = True
                print(f"🎉 {task} traj_idx {traj_idx} SUCCESS at step {step_idx}!")
                success_num[selected_id] += 1
                break
            
            # (可选) 打印调试信息，每20步打印一次以免刷屏
            if step_idx % 20 == 0:
                print(f"Step {step_idx}: dist={obj_to_target:.4f}, grip_cmd={action[3]:.2f}, pred_grip={target_gripper:.2f}")

        if not success_flag:
            print(f"❌ {task} traj_idx {traj_idx} FAILED after {MAX_RHC_STEPS} steps.")

        # 保存视频
        video_dir = META_CONFIG['video_dir']
        os.makedirs(f'{video_dir}/rollout_metaworld', exist_ok=True)
        save_path = f'{video_dir}/rollout_metaworld/{task}_{traj_idx}.mp4'
        mediapy.write_video(save_path, image_3, fps=20)
        print(f"Video saved to {save_path}")

# 打印最终统计
for i in range(len(task_list)):
    print(f"Task: {task_list[i]}, Success Rate: {success_num[i]}/{META_CONFIG['rollout_num']}")
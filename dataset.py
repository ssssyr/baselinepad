"""
Dataset classes and helper functions for training scripts.
Extracted from train_cotrain.py and train_robot.py for better code organization.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import cv2
import os
import json
import random


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class CustomDataset2(Dataset):
    def __init__(self, features_dir, args):
        self.features_dir = features_dir
        self.video_path = args.video_path
        self.args = args

        # robotic data
        self.condition_files, self.features_files, self.cond_depth_files, self.depth_files, \
        self.labels, self.ins_emb_files, self.cond_action, self.action_list = self.process_dataset(features_dir,skip_step=args.skip_step,video_only=False)
        
        # video data
        self.condition_files_v, self.features_files_v, self.cond_depth_files_v, self.depth_files_v, \
        self.labels_v, self.ins_emb_files_v, self.cond_action_v, self.action_list_v = self.process_dataset(self.video_path,skip_step=6,video_only=True)

    def process_dataset(self,features_dir,skip_step=4,video_only=False):
        condition_file = []
        features_file = []
        cond_depth_file = []
        depth_file = []
        labels = []
        ins_emb_file = []
        cond_action = []
        action_list = []

        features_dirs = features_dir.split("+")
        episode_info = []
        for dir in features_dirs:
            step_info = []
            # load json information, currently very dirty
            json_path = os.path.join(dir, "dataset_rgb_s_d.json")
            if not os.path.exists(json_path):
                json_path = os.path.join(dir, "dataset_info_traj.json")
            with open(json_path, "r") as f:
                episode_info_json = json.load(f)
            if video_only:
                episode_info_f = []
                for ii, traj in enumerate(episode_info_json):
                    for step in traj[str(ii)]:
                        episode_info_f.append(step)
            else:
                episode_info_f = episode_info_json

            for step in episode_info_f:
                if 'wrist_1' in step.keys(): # for metaworld data
                    step["wrist_1"] = os.path.join(dir, step["wrist_1"])
                elif 'path' in step.keys(): # for bridge data
                    step['wrist_1'] = os.path.join(dir, step['path'])
                
                if 'state' not in step.keys(): # for bridge data
                    step['state'] = np.array(step['action'])
                
                step["depth_1"] = os.path.join(dir, step["depth_1"]) if 'depth_1' in step.keys() else None
                step["ins_emb_path"] = os.path.join(dir, step["ins_emb_path"])

                step_info.append(step)
            episode_info += [step for step in step_info if int(step["episode"])%10!=9]
        
        # start prepare input
        for idx, episode in enumerate(episode_info):
            # episode: {"idx":train_steps, "episode": traj_id, "frame": episode_steps, "path": f'episode{traj_id:07}/frame{episode_steps:04}.npy', "lable": traj_id}
            cond_traj_idx = episode_info[idx]["episode"]
            if idx+skip_step >= len(episode_info):
                break
            pred_traj_idx = episode_info[idx+skip_step]["episode"]
            
            # if idx+step>traj length, just use last frame 
            if cond_traj_idx == pred_traj_idx:
                condition_file.append(episode_info[idx]["wrist_1"])
                cond_depth_file.append(episode_info[idx]["depth_1"]) if self.args.use_depth else None
                if self.args.action_steps>0:
                    action = episode_info[idx]['state']
                    cond_action.append(action)
                else:
                    cond_action.append(None)
                features = []
                depths = []
                actions = []
                cur_idx = idx
                for i in range(self.args.predict_horizon):
                    pre_idx = cur_idx+skip_step
                    # TODO
                    if pre_idx>=len(episode_info) or episode_info[pre_idx]["episode"] != cond_traj_idx:
                        features.append(features[-1])
                        depths.append(depths[-1]) if self.args.use_depth else None
                        actions.append(actions[-1]) if self.args.action_steps>0 else None
                    else:
                        features.append(episode_info[pre_idx]["wrist_1"])
                        depths.append(episode_info[pre_idx]["depth_1"]) if self.args.use_depth else None
                        if self.args.action_steps>0:
                            action = episode_info[pre_idx]['state'] #(1,7)
                            # print(action)
                            actions.append(action)
                        else:
                            actions.append(None)
                    cur_idx = pre_idx

                # features_file.append(episode_info[idx+skip_step]["path"] if cond_traj_idx == pred_traj_idx else features_file[-1])
                features_file.append(features) # [[x,x,x],[x,x,x],[x,x,x]]
                labels.append(int(cond_traj_idx))
                ins_emb_file.append(episode_info[idx]["ins_emb_path"])
                depth_file.append(depths)
                action_list.append(actions)
        print("length of dataset", len(condition_file))
        return condition_file, features_file, cond_depth_file, depth_file, labels, ins_emb_file, cond_action, action_list        

    def __len__(self):
        assert len(self.features_files) == len(self.labels), \
            "Number of feature files and label files should be same"
        return max(len(self.features_files),len(self.features_files_v))

    def filter(self, depth):
        depth = cv2.resize(depth, (32,32), interpolation=cv2.INTER_NEAREST)
        return depth
    
    def filter2(self, depth):
        depth = np.clip(depth,1000,5000)/5000
        depth = np.array(depth*256,dtype=np.uint8)
        depth = cv2.medianBlur(depth, 15)
        depth = cv2.resize(depth,(32,32),interpolation=cv2.INTER_NEAREST)/256
        return depth

    def __getitem__(self, idx):
        
        
        robot_sample = random.random() > 0.3
        loss_mask = np.array([1.0]) if robot_sample else np.array([0.0])
        if robot_sample:
            idx = idx % len(self.features_files)
            condition_files = self.condition_files
            features_files = self.features_files
            ins_emb_files = self.ins_emb_files
            cond_depth_files = self.cond_depth_files
            depth_files = self.depth_files
            action_list = self.action_list
            cond_action_list = self.cond_action
        else:
            idx = idx % len(self.features_files_v)
            condition_files = self.condition_files_v
            features_files = self.features_files_v
            ins_emb_files = self.ins_emb_files_v
            cond_depth_files = self.cond_depth_files_v
            depth_files = self.depth_files_v
            action_list = self.action_list_v
            cond_action_list = self.cond_action_v

        # rgb image
        condition_file = condition_files[idx]
        conditions = np.load(condition_file)
        feature_file = features_files[idx]
        features = []
        for i in range(len(feature_file)):
            features.append(np.load(feature_file[i]))
        features = np.concatenate(features,axis=1)

        # text info
        if self.args.text_cond:
            text_file = ins_emb_files[idx]
            labels = np.load(text_file)
        else:
            labels = np.array([self.labels[idx]],dtype=np.int32)
        
        # depth image
        if self.args.use_depth and robot_sample:
            cond_depth_file = cond_depth_files[idx]
            cond_depth = np.load(cond_depth_file)
            cond_depth = self.filter(cond_depth) if not self.args.depth_filter else self.filter2(cond_depth)
            cond_depth = cond_depth[np.newaxis]

            depth_file = depth_files[idx]
            depths = []
            for i in range(len(depth_file)):
                d = np.load(depth_file[i])
                d = self.filter(d) if not self.args.depth_filter else self.filter2(d)
                depths.append(d)
            depths = np.stack(depths)
        else:
            cond_depth = np.zeros((1,32,32))
            depths = np.zeros((self.args.predict_horizon,32,32))

        # actions
        if self.args.action_steps>0 and robot_sample:
            if self.args.absolute_action:
                action = np.array(action_list[idx])
            else:
                action = np.array(action_list[idx])-np.array(cond_action_list[idx])
            action = action[:self.args.action_steps,:]
            action = action*self.args.action_scale
            cond_action = np.array(cond_action_list[idx]).reshape(1,-1)*self.args.action_scale

            # whether condition on current action
            if self.args.action_condition:
                action = action.reshape(1,-1)
                assert action.shape[-1] == self.args.action_dim*self.args.action_steps
                assert cond_action.shape[-1] == self.args.action_dim
            else:
                assert action.shape[-1] == self.args.action_dim
        else:
            action = np.zeros((1,self.args.action_dim*self.args.action_steps))
            cond_action = np.zeros((1,self.args.action_dim))

        return torch.from_numpy(conditions), torch.from_numpy(features), torch.from_numpy(labels), torch.from_numpy(action).float(), torch.from_numpy(cond_depth).float(), torch.from_numpy(depths).float(), torch.from_numpy(cond_action).float(), torch.from_numpy(loss_mask).float()


class RobotDataset(Dataset):
    def __init__(self, features_dir, args):

        # You need to implement a new dataset class if youre dataset structure is different
        ################################ Default dataset structrue:############################
        #   dataset_rgb_s_d.json
        #   episode 0
        #       clip_emb
        #       step 0.npy
        #       step 1.npy
        #       ...
        #   episode 1
        #       clip_emb
        #       step 0.npy
        #       step 1.npy
        #       ...
        #   episode 2
        ####################################################################################
        
        self.features_dir = features_dir
        self.args = args
        # rgb
        self.cond_rgb_file = []
        self.rgb_file = []
        # depth
        self.cond_depth_file = []
        self.depth_file = []
        # robot pose
        self.cond_action = []
        self.action = []
        # instruction
        self.ins_emb_file = []

        skip_step = args.skip_step # prediction skip step

        features_dirs = features_dir.split("+")
        step_infos = []
        for dir in features_dirs:
            step_info = []
            with open(os.path.join(dir, "dataset_rgb_s_d.json"), "r") as f:
                step_infos_f = json.load(f)
            for step in step_infos_f:
                step["wrist_1"] = os.path.join(dir, step["wrist_1"])
                step["depth_1"] = os.path.join(dir, step["depth_1"])
                step["ins_emb_path"] = os.path.join(dir, step["ins_emb_path"])
                step_info.append(step)
            step_infos += [step for step in step_info]
            # step_infos += [step for step in step_info if int(step["episode"])%50<20]
        
        # start prepare input
        for idx, _ in enumerate(step_infos):
            # episode: {"idx":train_steps, "episode": traj_id, "frame": episode_steps, "path": f'episode{traj_id:07}/frame{episode_steps:04}.npy', "lable": traj_id}
            cond_traj_idx = step_infos[idx]["episode"]
            if idx+skip_step >= len(step_infos):
                break
            pred_traj_idx = step_infos[idx+skip_step]["episode"]
            
            # if idx+step>traj length, just use last frame 
            if cond_traj_idx == pred_traj_idx:
                # current frame
                self.cond_rgb_file.append(step_infos[idx]["wrist_1"])
                self.cond_depth_file.append(step_infos[idx]["depth_1"]) if args.use_depth else None
                self.cond_action.append(step_infos[idx]['state']) if args.action_steps>0 else None
                features = []
                depths = []
                actions = []

                # future frames
                for i in range(args.predict_horizon):
                    pre_idx = idx + i*skip_step
                    if pre_idx>=len(step_infos) or step_infos[pre_idx]["episode"] != cond_traj_idx:
                        features.append(features[-1])
                        depths.append(depths[-1]) if args.use_depth else None
                        actions.append(actions[-1]) if args.action_steps>0 else None
                    else:
                        features.append(step_infos[pre_idx]["wrist_1"])
                        depths.append(step_infos[pre_idx]["depth_1"]) if args.use_depth else None
                        actions.append(step_infos[pre_idx]['state']) if args.action_steps>0 else None

                self.rgb_file.append(features) # [[x,x,x],[x,x,x],[x,x,x]]
                self.depth_file.append(depths)
                self.action.append(actions)
                self.ins_emb_file.append(step_infos[idx]["ins_emb_path"])
        print("length of dataset", len(self.cond_rgb_file))

    def __len__(self):
        return len(self.rgb_file)

    def filter(self, depth):
        depth = cv2.resize(depth, (32,32), interpolation=cv2.INTER_NEAREST)
        return depth
    
    def filter2(self, depth):
        depth = np.clip(depth,1000,5000)/5000
        depth = np.array(depth*256,dtype=np.uint8)
        depth = cv2.medianBlur(depth, 15)
        depth = cv2.resize(depth,(32,32),interpolation=cv2.INTER_NEAREST)/256
        return depth

    def __getitem__(self, idx):
        # rgb image
        condition_file = self.cond_rgb_file[idx]
        rgb_cond = np.load(condition_file)
        feature_file = self.rgb_file[idx]
        rgb = []
        for i in range(len(feature_file)):
            rgb.append(np.load(feature_file[i]))
        rgb = np.concatenate(rgb,axis=1)

        # text info
        if self.args.text_cond:
            text_file = self.ins_emb_file[idx]
            labels = np.load(text_file)
        else:
            labels = np.array([self.labels[idx]],dtype=np.int32)
        
        # depth image
        if self.args.use_depth:
            cond_depth_file = self.cond_depth_file[idx]
            cond_depth = np.load(cond_depth_file)
            cond_depth = self.filter(cond_depth) if not self.args.depth_filter else self.filter2(cond_depth)
            cond_depth = cond_depth[np.newaxis]

            depth_file = self.depth_files[idx]
            depths = []
            for i in range(len(depth_file)):
                d = np.load(depth_file[i])
                d = self.filter(d) if not self.args.depth_filter else self.filter2(d)
                depths.append(d)
            depths = np.stack(depths)
        else:
            cond_depth = np.array([0])
            depths = np.array([0])

        # actions
        if self.args.action_steps>0:
            if self.args.absolute_action:
                action = np.array(self.action[idx])
            else:
                action = np.array(self.action[idx])-np.array(self.cond_action[idx])
            action = action[:self.args.action_steps,:]
            action = action*self.args.action_scale
            cond_action = np.array(self.cond_action[idx]).reshape(1,-1)*self.args.action_scale

            # whether condition on current pose
            if self.args.action_condition:
                action = action.reshape(1,-1)
                assert action.shape[-1] == self.args.action_dim*self.args.action_steps
                assert cond_action.shape[-1] == self.args.action_dim
            else:
                assert action.shape[-1] == self.args.action_dim
        else:
            action = np.array([0])
            cond_action = np.array([0])

        return torch.from_numpy(rgb_cond), torch.from_numpy(rgb), torch.from_numpy(cond_depth).float(), torch.from_numpy(depths).float(), torch.from_numpy(cond_action).float(), torch.from_numpy(action).float(), torch.from_numpy(labels).float()

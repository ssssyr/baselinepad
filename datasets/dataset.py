
"""
Dataset classes and helper functions for training scripts.
Compatible with Meta-World PAD pipeline.

Key fixes:
- cond_action decoupled from use_depth (prevents IndexError)
- stable padding across episode boundary for features/depth/actions
- consistent depth fallback shapes
- labels always available when text_cond=False
- return order matches train loop: x_cond, x, depth_cond, depth, action_cond, action, force_cond, y
"""

import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image



def resolve_force_stats(args, feature_path):
    if not getattr(args, "use_force", False):
        return None, None

    force_mean = getattr(args, "force_mean", None)
    force_std = getattr(args, "force_std", None)
    if (force_mean is not None and force_std is not None
            and not isinstance(force_mean, bool) and not isinstance(force_std, bool)):
        return np.array(force_mean, dtype=np.float32), np.array(force_std, dtype=np.float32)

    stats_path = getattr(args, "force_stats_path", None)
    if stats_path is None and feature_path:
        stats_root = feature_path.split("+")[0]
        stats_path = os.path.join(stats_root, "force_stats.json")
    if stats_path and os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            payload = json.load(f)
        mean = np.array(payload.get("mean", [0, 0, 0, 0, 0, 0]), dtype=np.float32)
        std = np.array(payload.get("std", [1, 1, 1, 1, 1, 1]), dtype=np.float32)
        return mean, std

    print("⚠️ force_stats.json not found; defaulting to mean=0,std=1")
    return np.zeros(6, dtype=np.float32), np.ones(6, dtype=np.float32)


def normalize_force(force, mean, std):
    """
    归一化力数据（带截断处理离群值）

    Args:
        force: 原始力数据 [fx, fy, fz, tx, ty, tz]
        mean: 全局均值
        std: 全局标准差

    Returns:
        归一化后的力数据
    """
    if mean is None or std is None:
        return force

    force = np.array(force, dtype=np.float32)

    
    
    force[:3] = np.clip(force[:3], -100, 100)   
    force[3:] = np.clip(force[3:], -10, 10)     

    
    std = np.where(std < 1e-6, 1.0, std)
    normalized = (force - mean) / std

    return normalized



def center_crop_arr(pil_image, image_size):
    """
    Center cropping from ADM.
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
        self.args = args
        self.force_mean, self.force_std = resolve_force_stats(args, features_dir)
        (self.condition_files, self.features_files,
         self.cond_depth_files, self.depth_files,
         self.labels, self.ins_emb_files,
         self.cond_action, self.action_list,
         self.cond_force) = self.process_dataset(
            features_dir, skip_step=args.skip_step, video_only=False
        )

    def process_dataset(self, features_dir, skip_step=4, video_only=False):
        condition_file, features_file = [], []
        cond_depth_file, depth_file = [], []
        labels, ins_emb_file = [], []
        cond_action, action_list = [], []
        cond_force = []

        features_dirs = features_dir.split("+")
        episode_info = []
        for dir_ in features_dirs:
            step_info = []
            
            json_path = os.path.join(dir_, "dataset_rgb_s_d.json")
            if not os.path.exists(json_path):
                json_path = os.path.join(dir_, "dataset_info_traj.json")
            with open(json_path, "r") as f:
                info_json = json.load(f)

            if video_only:
                
                episode_info_f = []
                for ii, traj in enumerate(info_json):
                    for step in traj[str(ii)]:
                        episode_info_f.append(step)
            else:
                episode_info_f = info_json

            for step in episode_info_f:
                
                if 'wrist_1' in step:
                    step["wrist_1"] = os.path.join(dir_, step["wrist_1"])
                elif 'path' in step:
                    step['wrist_1'] = os.path.join(dir_, step['path'])

                
                if 'state' not in step and 'action' in step:
                    step['state'] = step['action']

                
                step["depth_1"] = os.path.join(dir_, step["depth_1"]) if 'depth_1' in step else None
                if 'ins_emb_path' in step:
                    step["ins_emb_path"] = os.path.join(dir_, step["ins_emb_path"])

                step_info.append(step)

            
            episode_info += [s for s in step_info]

        
        for idx in range(len(episode_info)):
            cond_traj_idx = episode_info[idx]["episode"]
            if idx + skip_step >= len(episode_info):
                break
            pred_traj_idx = episode_info[idx + skip_step]["episode"]

            if cond_traj_idx == pred_traj_idx:
                
                condition_file.append(episode_info[idx]["wrist_1"])
                if self.args.use_depth:
                    cond_depth_file.append(episode_info[idx].get("depth_1", None))
                if self.args.action_steps > 0:
                    cond_action.append(episode_info[idx]["state"])
                else:
                    cond_action.append(None)
                if getattr(self.args, "use_force", False):
                    cond_force.append(episode_info[idx].get("force", None))
                else:
                    cond_force.append(None)

                
                feats, depths, acts = [], [], []
                last_depth = episode_info[idx].get("depth_1", None)
                last_action = episode_info[idx]["state"] if self.args.action_steps > 0 else None
                cur = idx
                for _ in range(self.args.predict_horizon):
                    nxt = cur + skip_step
                    same_ep = (nxt < len(episode_info)) and (episode_info[nxt]["episode"] == cond_traj_idx)
                    if same_ep:
                        feats.append(episode_info[nxt]["wrist_1"])
                        if self.args.use_depth:
                            last_depth = episode_info[nxt].get("depth_1", last_depth)
                            depths.append(last_depth)
                        if self.args.action_steps > 0:
                            last_action = episode_info[nxt]["state"]
                            acts.append(last_action)
                        cur = nxt
                    else:
                        feats.append(feats[-1] if len(feats) > 0 else episode_info[idx]["wrist_1"])
                        if self.args.use_depth:
                            depths.append(last_depth)
                        if self.args.action_steps > 0:
                            acts.append(last_action)

                features_file.append(feats)
                depth_file.append(depths)
                action_list.append(acts)
                labels.append(int(cond_traj_idx))
                ins_emb_file.append(episode_info[idx].get("ins_emb_path", None))

        print("length of dataset", len(condition_file))
        return (condition_file, features_file, cond_depth_file, depth_file,
                labels, ins_emb_file, cond_action, action_list, cond_force)

    def __len__(self):
        assert len(self.features_files) == len(self.labels), \
            "Number of feature files and label files should be same"
        return len(self.features_files)

    def filter(self, depth):
        return cv2.resize(depth, (32, 32), interpolation=cv2.INTER_NEAREST)

    def filter2(self, depth):
        depth = np.clip(depth, 1000, 5000) / 5000
        depth = np.array(depth * 256, dtype=np.uint8)
        depth = cv2.medianBlur(depth, 15)
        return cv2.resize(depth, (32, 32), interpolation=cv2.INTER_NEAREST) / 256

    def __getitem__(self, idx):
        
        cond_files = self.condition_files
        feat_files = self.features_files
        ins_files = self.ins_emb_files
        cdepth_files = self.cond_depth_files
        depth_files = self.depth_files
        act_list = self.action_list
        cact_list = self.cond_action

        
        x_cond = np.load(cond_files[idx])                              
        feats = [np.load(p) for p in feat_files[idx]]                  
        x = np.concatenate(feats, axis=1)                              

        
        if getattr(self.args, "text_cond", False) and ins_files[idx] is not None:
            y = np.load(ins_files[idx])                                
        else:
            y = np.array([self.labels[idx]], dtype=np.int32)

        
        if getattr(self.args, "use_depth", False):
            if cdepth_files[idx] is not None:
                dcond = np.load(cdepth_files[idx])
                dcond = self.filter(dcond) if not getattr(self.args, "depth_filter", False) else self.filter2(dcond)
                dcond = dcond[np.newaxis]                              
            else:
                dcond = np.zeros((1, 32, 32), dtype=np.float32)
            dseq = []
            for p in depth_files[idx]:
                if p is None:
                    dseq.append(np.zeros((32, 32), dtype=np.float32))
                else:
                    d = np.load(p)
                    d = self.filter(d) if not getattr(self.args, "depth_filter", False) else self.filter2(d)
                    dseq.append(d)
            depth = np.stack(dseq) if len(dseq) > 0 else np.zeros((self.args.predict_horizon, 32, 32), dtype=np.float32)
        else:
            dcond = np.zeros((1, 32, 32), dtype=np.float32)
            depth = np.zeros((self.args.predict_horizon, 32, 32), dtype=np.float32)

        
        if getattr(self.args, "action_steps", 0) > 0:
            act = np.array(act_list[idx], dtype=np.float32)                    
            cact = np.array(cact_list[idx], dtype=np.float32).reshape(1, -1)   
            if not getattr(self.args, "absolute_action", True):
                act = act - cact
            act = act[:self.args.action_steps, :]                               
            act = act * self.args.action_scale
            cact = cact * self.args.action_scale

            if getattr(self.args, "action_condition", True):
                act = act.reshape(1, -1)                                        
                assert act.shape[-1] == self.args.action_dim * self.args.action_steps
                assert cact.shape[-1] == self.args.action_dim
            else:
                
                act = act[0:1, :]
        else:
            action_size = max(1, self.args.action_dim * self.args.action_steps)
            act = np.zeros((1, action_size), dtype=np.float32)
            cact = np.zeros((1, max(1, self.args.action_dim)), dtype=np.float32)

        
        if getattr(self.args, "use_force", False):
            force_val = self.cond_force[idx]
            if force_val is None:
                force = np.zeros((1, 6), dtype=np.float32)
            else:
                force = np.array(force_val, dtype=np.float32).reshape(1, 6)
            if self.force_mean is not None and self.force_std is not None:
                force = normalize_force(force, self.force_mean, self.force_std)
        else:
            force = np.zeros((1, 6), dtype=np.float32)

        return (torch.from_numpy(x_cond), torch.from_numpy(x),
                torch.from_numpy(dcond).float(), torch.from_numpy(depth).float(),
                torch.from_numpy(cact).float(), torch.from_numpy(act).float(),
                torch.from_numpy(force).float(),
                torch.from_numpy(y).float())




class RobotDataset(Dataset):
    def __init__(self, features_dir, args):
        """
        Default expected structure under each features_dir:
          dataset_rgb_s_d.json
          episode0000000/
            text_clip.npy
            color_wrist_1_0000.npy
            ...
          episode0000001/
            ...
        Each JSON record (one per frame) includes:
          {
            "episode": <int>, "frame": <int>,
            "wrist_1": "episodeXXXX/...",            
            "ins_emb_path": "episodeXXXX/text_clip.npy",
            "state": [x,y,z,grip],                   
            (optional) "depth_1": "..."              
          }
        """
        self.features_dir = features_dir
        self.args = args

        
        self.cond_rgb_file, self.rgb_file = [], []          
        self.cond_depth_file, self.depth_file = [], []      
        self.cond_action, self.action = [], []              
        self.cond_force = []                                
        self.ins_emb_file, self.labels = [], []             
        self.force_mean, self.force_std = resolve_force_stats(args, features_dir)

        
        step_infos = []
        for d in features_dir.split("+"):
            json_path = os.path.join(d, "dataset_rgb_s_d.json")
            with open(json_path, "r") as f:
                steps = json.load(f)
            for s in steps:
                s = dict(s)  
                s["wrist_1"] = os.path.join(d, s["wrist_1"])
                if getattr(args, "use_depth", False) and ("depth_1" in s):
                    s["depth_1"] = os.path.join(d, s["depth_1"])
                if "ins_emb_path" in s:
                    s["ins_emb_path"] = os.path.join(d, s["ins_emb_path"])
                step_infos.append(s)

        
        episodes = {}
        for s in step_infos:
            ep = int(s["episode"])
            episodes.setdefault(ep, []).append(s)

        H = args.predict_horizon
        S = args.skip_step

        for ep, frames in episodes.items():
            
            if "frame" in frames[0]:
                frames = sorted(frames, key=lambda x: int(x["frame"]))
            L = len(frames)
            
            
            
            max_t = L - 1

            for t in range(0, max_t + 1):
                cond = frames[t]

                
                if getattr(args, "use_depth", False) and ("depth_1" not in cond):
                    continue
                if getattr(args, "action_steps", 0) > 0 and ("state" not in cond):
                    continue

                
                future_rgb, future_depth, future_action = [], [], []
                valid = True
                
                for i in range(1, H + 1):
                    
                    raw_idx = t + i * S
                    
                    idx_f = min(raw_idx, L - 1)
                    
                    step_f = frames[idx_f]
                    
                    
                    if getattr(args, "use_depth", False) and ("depth_1" not in step_f):
                        valid = False
                        break
                    if getattr(args, "action_steps", 0) > 0 and ("state" not in step_f):
                        valid = False
                        break

                    future_rgb.append(step_f["wrist_1"])
                    if getattr(args, "use_depth", False):
                        future_depth.append(step_f["depth_1"])
                    if getattr(args, "action_steps", 0) > 0:
                        future_action.append(step_f["state"])

                if not valid:
                    continue  

                
                
                self.cond_rgb_file.append(cond["wrist_1"])
                if getattr(args, "use_depth", False):
                    self.cond_depth_file.append(cond["depth_1"])
                if getattr(args, "action_steps", 0) > 0:
                    self.cond_action.append(cond["state"])
                if getattr(args, "use_force", False):
                    self.cond_force.append(cond.get("force", None))
                
                self.rgb_file.append(future_rgb)                     
                if getattr(args, "use_depth", False):
                    self.depth_file.append(future_depth)             
                else:
                    self.depth_file.append([])                       
                if getattr(args, "action_steps", 0) > 0:
                    self.action.append(future_action)                
                
                self.ins_emb_file.append(cond.get("ins_emb_path", None))
                self.labels.append(int(ep))

        
        assert len(self.rgb_file) == len(self.cond_rgb_file) == len(self.ins_emb_file) == len(self.labels), \
            "rgb/cond_rgb/ins_emb/labels length mismatch after filtering"
        if getattr(self.args, "action_steps", 0) > 0:
            assert len(self.action) == len(self.cond_action) == len(self.rgb_file), \
                "action/cond_action/rgb length mismatch after filtering"

        print("length of dataset", len(self.cond_rgb_file))

    def __len__(self):
        return len(self.rgb_file)

    @staticmethod
    def filter(depth):
        return cv2.resize(depth, (32, 32), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def filter2(depth):
        depth = np.clip(depth, 1000, 5000) / 5000
        depth = np.array(depth * 256, dtype=np.uint8)
        depth = cv2.medianBlur(depth, 15)
        return cv2.resize(depth, (32, 32), interpolation=cv2.INTER_NEAREST) / 256

    def __getitem__(self, idx):
        
        x_cond = np.load(self.cond_rgb_file[idx])                 
        rgbs = [np.load(p) for p in self.rgb_file[idx]]           
        x = np.concatenate(rgbs, axis=1)                          

        
        if getattr(self.args, "text_cond", False) and (self.ins_emb_file[idx] is not None):
            y = np.load(self.ins_emb_file[idx])                   
        else:
            y = np.array([self.labels[idx]], dtype=np.int32)

        
        if getattr(self.args, "use_depth", False):
            dcond = np.load(self.cond_depth_file[idx])
            dcond = self.filter(dcond) if not getattr(self.args, "depth_filter", False) else self.filter2(dcond)
            dcond = dcond[np.newaxis]                             

            dseq = []
            for p in self.depth_file[idx]:
                d = np.load(p)
                d = self.filter(d) if not getattr(self.args, "depth_filter", False) else self.filter2(d)
                dseq.append(d)
            depth = np.stack(dseq)                                
        else:
            dcond = np.zeros((1, 32, 32), dtype=np.float32)
            depth = np.zeros((self.args.predict_horizon, 32, 32), dtype=np.float32)

        
        if getattr(self.args, "action_steps", 0) > 0:
            act_seq = np.array(self.action[idx], dtype=np.float32)                    
            base = np.array(self.cond_action[idx], dtype=np.float32).reshape(1, -1)   
            if not getattr(self.args, "absolute_action", True):
                act_seq = act_seq - base
            act_seq = act_seq[:self.args.action_steps, :]                              
            act_seq = act_seq * self.args.action_scale
            cact = base * self.args.action_scale                                       

            if getattr(self.args, "action_condition", False):
                
                action = act_seq.reshape(1, -1)                                        
            else:
                
                action = act_seq[0:1, :]                                               
        else:
            action = np.zeros((1, self.args.action_dim * self.args.action_steps), dtype=np.float32)
            cact = np.zeros((1, self.args.action_dim), dtype=np.float32)

        
        if getattr(self.args, "use_force", False):
            force_val = self.cond_force[idx] if idx < len(self.cond_force) else None
            if force_val is None:
                force = np.zeros((1, 6), dtype=np.float32)
            else:
                force = np.array(force_val, dtype=np.float32).reshape(1, 6)
            if self.force_mean is not None and self.force_std is not None:
                force = normalize_force(force, self.force_mean, self.force_std)
        else:
            force = np.zeros((1, 6), dtype=np.float32)

        return (
            torch.from_numpy(x_cond),
            torch.from_numpy(x),
            torch.from_numpy(dcond).float(),
            torch.from_numpy(depth).float(),
            torch.from_numpy(cact).float(),
            torch.from_numpy(action).float(),
            torch.from_numpy(force).float(),
            torch.from_numpy(y).float(),
        )

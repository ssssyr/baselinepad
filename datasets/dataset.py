# -*- coding: utf-8 -*-
"""
Dataset classes and helper functions for training scripts.
Compatible with Meta-World PAD pipeline.

Key fixes:
- cond_action decoupled from use_depth (prevents IndexError)
- stable padding across episode boundary for features/depth/actions
- consistent depth fallback shapes
- labels always available when text_cond=False
- return order matches train loop: x_cond, x, depth_cond, depth, action_cond, action, y
"""

import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image


# -------- optional helper (unused by default, kept for completeness) --------
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


# =============================== CustomDataset2 ===============================
# 兼容多源数据（bridge/metaworld），保留但建议在本项目优先使用 RobotDataset
class CustomDataset2(Dataset):
    def __init__(self, features_dir, args):
        self.features_dir = features_dir
        self.args = args
        (self.condition_files, self.features_files,
         self.cond_depth_files, self.depth_files,
         self.labels, self.ins_emb_files,
         self.cond_action, self.action_list) = self.process_dataset(
            features_dir, skip_step=args.skip_step, video_only=False
        )

    def process_dataset(self, features_dir, skip_step=4, video_only=False):
        condition_file, features_file = [], []
        cond_depth_file, depth_file = [], []
        labels, ins_emb_file = [], []
        cond_action, action_list = [], []

        features_dirs = features_dir.split("+")
        episode_info = []
        for dir_ in features_dirs:
            step_info = []
            # 支持两种索引文件名
            json_path = os.path.join(dir_, "dataset_rgb_s_d.json")
            if not os.path.exists(json_path):
                json_path = os.path.join(dir_, "dataset_info_traj.json")
            with open(json_path, "r") as f:
                info_json = json.load(f)

            if video_only:
                # 兼容某些旧结构：[{ "0":[...], "1":[...]}, ...]
                episode_info_f = []
                for ii, traj in enumerate(info_json):
                    for step in traj[str(ii)]:
                        episode_info_f.append(step)
            else:
                episode_info_f = info_json

            for step in episode_info_f:
                # path 归一化到 wrist_1
                if 'wrist_1' in step:
                    step["wrist_1"] = os.path.join(dir_, step["wrist_1"])
                elif 'path' in step:
                    step['wrist_1'] = os.path.join(dir_, step['path'])

                # 桥接：如果没有 state，但有 action，则用 action 代替（保持 shape）
                if 'state' not in step and 'action' in step:
                    step['state'] = step['action']

                # 可选深度/文本
                step["depth_1"] = os.path.join(dir_, step["depth_1"]) if 'depth_1' in step else None
                if 'ins_emb_path' in step:
                    step["ins_emb_path"] = os.path.join(dir_, step["ins_emb_path"])

                step_info.append(step)

            # 简单样本筛选（这里不丢弃，以最大化样本）
            episode_info += [s for s in step_info]

        # 组样本
        for idx in range(len(episode_info)):
            cond_traj_idx = episode_info[idx]["episode"]
            if idx + skip_step >= len(episode_info):
                break
            pred_traj_idx = episode_info[idx + skip_step]["episode"]

            if cond_traj_idx == pred_traj_idx:
                # 当前步
                condition_file.append(episode_info[idx]["wrist_1"])
                if self.args.use_depth:
                    cond_depth_file.append(episode_info[idx].get("depth_1", None))
                if self.args.action_steps > 0:
                    cond_action.append(episode_info[idx]["state"])
                else:
                    cond_action.append(None)

                # 未来帧（稳定填充）
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
                labels, ins_emb_file, cond_action, action_list)

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
        # aliases
        cond_files = self.condition_files
        feat_files = self.features_files
        ins_files = self.ins_emb_files
        cdepth_files = self.cond_depth_files
        depth_files = self.depth_files
        act_list = self.action_list
        cact_list = self.cond_action

        # RGB latent
        x_cond = np.load(cond_files[idx])                              # (1,4,32,32)
        feats = [np.load(p) for p in feat_files[idx]]                  # list of (1,4,32,32)
        x = np.concatenate(feats, axis=1)                              # (1,4*H,32,32)

        # Text
        if getattr(self.args, "text_cond", False) and ins_files[idx] is not None:
            y = np.load(ins_files[idx])                                # e.g. (512,)
        else:
            y = np.array([self.labels[idx]], dtype=np.int32)

        # Depth
        if getattr(self.args, "use_depth", False):
            if cdepth_files[idx] is not None:
                dcond = np.load(cdepth_files[idx])
                dcond = self.filter(dcond) if not getattr(self.args, "depth_filter", False) else self.filter2(dcond)
                dcond = dcond[np.newaxis]                              # (1,32,32)
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

        # Actions (actually pose/state)
        if getattr(self.args, "action_steps", 0) > 0:
            act = np.array(act_list[idx], dtype=np.float32)                    # (H,4)
            cact = np.array(cact_list[idx], dtype=np.float32).reshape(1, -1)   # (1,4)
            if not getattr(self.args, "absolute_action", True):
                act = act - cact
            act = act[:self.args.action_steps, :]                               # (S,4)
            act = act * self.args.action_scale
            cact = cact * self.args.action_scale

            if getattr(self.args, "action_condition", True):
                act = act.reshape(1, -1)                                        # (1, 4*S)
                assert act.shape[-1] == self.args.action_dim * self.args.action_steps
                assert cact.shape[-1] == self.args.action_dim
            else:
                # if not conditioning on action, often use one step
                act = act[0:1, :]
        else:
            action_size = max(1, self.args.action_dim * self.args.action_steps)
            act = np.zeros((1, action_size), dtype=np.float32)
            cact = np.zeros((1, max(1, self.args.action_dim)), dtype=np.float32)

        return (torch.from_numpy(x_cond), torch.from_numpy(x),
                torch.from_numpy(dcond).float(), torch.from_numpy(depth).float(),
                torch.from_numpy(cact).float(), torch.from_numpy(act).float(),
                torch.from_numpy(y).float())


# ================================ RobotDataset ================================
# 推荐在本项目使用：更贴合当前 Meta-World 数据结构
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
          { "episode": <int>, "frame": <int>, "wrist_1": "episodeXXXX/...", "ins_emb_path": "episodeXXXX/text_clip.npy", "state": [x,y,z,grip], (optional) "depth_1": "..." }
        """
        self.features_dir = features_dir
        self.args = args

        # holders
        self.cond_rgb_file, self.rgb_file = [], []
        self.cond_depth_file, self.depth_file = [], []
        self.cond_action, self.action = [], []
        self.ins_emb_file, self.labels = [], []

        skip_step = args.skip_step

        # load all steps
        step_infos = []
        for d in features_dir.split("+"):
            json_path = os.path.join(d, "dataset_rgb_s_d.json")
            with open(json_path, "r") as f:
                steps = json.load(f)
            for s in steps:
                # normalize paths to absolute
                s["wrist_1"] = os.path.join(d, s["wrist_1"])
                if getattr(args, "use_depth", False) and "depth_1" in s:
                    s["depth_1"] = os.path.join(d, s["depth_1"])
                if "ins_emb_path" in s:
                    s["ins_emb_path"] = os.path.join(d, s["ins_emb_path"])
                step_infos.append(s)

        # build samples
        for idx in range(len(step_infos)):
            cond_traj = step_infos[idx]["episode"]
            if idx + skip_step >= len(step_infos):
                break
            pred_traj = step_infos[idx + skip_step]["episode"]
            if cond_traj != pred_traj:
                continue

            # current
            self.cond_rgb_file.append(step_infos[idx]["wrist_1"])
            if getattr(args, "use_depth", False) and ("depth_1" in step_infos[idx]):
                self.cond_depth_file.append(step_infos[idx]["depth_1"])
            if getattr(args, "action_steps", 0) > 0:
                self.cond_action.append(step_infos[idx]["state"])

            # future with stable padding
            feats, depths, acts = [], [], []
            last_depth = step_infos[idx].get("depth_1", None)
            last_action = step_infos[idx]["state"] if getattr(args, "action_steps", 0) > 0 else None

            cur = idx
            for _ in range(args.predict_horizon):
                nxt = cur + skip_step
                same_ep = (nxt < len(step_infos)) and (step_infos[nxt]["episode"] == cond_traj)
                if same_ep:
                    feats.append(step_infos[nxt]["wrist_1"])
                    if getattr(args, "use_depth", False) and ("depth_1" in step_infos[nxt]):
                        last_depth = step_infos[nxt]["depth_1"]
                        depths.append(last_depth)
                    if getattr(args, "action_steps", 0) > 0:
                        last_action = step_infos[nxt]["state"]
                        acts.append(last_action)
                    cur = nxt
                else:
                    feats.append(feats[-1] if len(feats) > 0 else step_infos[idx]["wrist_1"])
                    if getattr(args, "use_depth", False):
                        depths.append(last_depth)
                    if getattr(args, "action_steps", 0) > 0:
                        acts.append(last_action)

            self.rgb_file.append(feats)
            self.depth_file.append(depths)
            if getattr(args, "action_steps", 0) > 0:
                self.action.append(acts)

            self.ins_emb_file.append(step_infos[idx].get("ins_emb_path", None))
            self.labels.append(int(cond_traj))

        # sanity checks
        assert len(self.rgb_file) == len(self.cond_rgb_file) == len(self.ins_emb_file) == len(self.labels), \
            "rgb/cond_rgb/ins_emb/labels length mismatch"
        if getattr(args, "action_steps", 0) > 0:
            assert len(self.action) == len(self.cond_action) == len(self.rgb_file), \
                "action/cond_action/rgb length mismatch"

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
        # ----- RGB -----
        x_cond = np.load(self.cond_rgb_file[idx])                 # (1,4,32,32)
        rgbs = [np.load(p) for p in self.rgb_file[idx]]           # list of (1,4,32,32)
        x = np.concatenate(rgbs, axis=1)                          # (1,4*H,32,32)

        # ----- Text -----
        if getattr(self.args, "text_cond", False) and (self.ins_emb_file[idx] is not None):
            y = np.load(self.ins_emb_file[idx])                   # (512,) typically
        else:
            y = np.array([self.labels[idx]], dtype=np.int32)

        # ----- Depth -----
        if getattr(self.args, "use_depth", False):
            if idx < len(self.cond_depth_file) and self.cond_depth_file[idx] is not None:
                dcond = np.load(self.cond_depth_file[idx])
                dcond = self.filter(dcond) if not getattr(self.args, "depth_filter", False) else self.filter2(dcond)
                dcond = dcond[np.newaxis]                        # (1,32,32)
            else:
                dcond = np.zeros((1, 32, 32), dtype=np.float32)

            dseq = []
            for p in self.depth_file[idx]:
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

        # ----- Actions (state/pose) -----
        if getattr(self.args, "action_steps", 0) > 0:
            act_seq = np.array(self.action[idx], dtype=np.float32)                    # (H,4)
            base = np.array(self.cond_action[idx], dtype=np.float32).reshape(1, -1)   # (1,4)
            if not getattr(self.args, "absolute_action", True):
                act_seq = act_seq - base
            act_seq = act_seq[:self.args.action_steps, :]                              # (S,4)
            act_seq = act_seq * self.args.action_scale
            cact = base * self.args.action_scale                                       # (1,4)

            if getattr(self.args, "action_condition", True):
                action = act_seq.reshape(1, -1)                                        # (1, 4*S)
                assert action.shape[-1] == self.args.action_dim * self.args.action_steps
                assert cact.shape[-1] == self.args.action_dim
            else:
                action = act_seq[0:1, :]                                               # (1,4)
        else:
            action_size = max(1, self.args.action_dim * self.args.action_steps)
            action = np.zeros((1, action_size), dtype=np.float32)
            cact = np.zeros((1, max(1, self.args.action_dim)), dtype=np.float32)

        return (
            torch.from_numpy(x_cond),
            torch.from_numpy(x),
            torch.from_numpy(dcond).float(),
            torch.from_numpy(depth).float(),
            torch.from_numpy(cact).float(),
            torch.from_numpy(action).float(),
            torch.from_numpy(y).float(),
        )


"""ur10_config.py

Centralized configuration for real-world UR10 deployment.
Configuration aligned with training config: configs/metaworld_4d.yaml
"""

from pathlib import Path
import numpy as np

# -------------------------------
#    MODEL CONFIGURATION
# -------------------------------

CONFIG = {
    "model": {
        "ckpt_path": "/mnt/sda/syr/checkpoint/checkpoint0120/0230000.pt",
        "vae_path": "/home/syr/code/models/sd-vae-ft-mse",
        "clip_path": "/home/syr/code/models/clip-vit-base-patch32",
        "denoise_steps": "ddim16",
        "gpu_id": 0,
        "use_fp16": False,
    },

    # ---------------------------
    #    CAMERA CONFIGURATION
    # ---------------------------
    "camera": {
        "serial_number": "",
        "width": 1280,
        "height": 720,
        "fps": 30,
    },

    # ---------------------------
    #    ROBOT CONFIGURATION
    # ---------------------------
    "robot": {
        "ip": "192.168.1.50",
        "gripper_ip": "192.168.1.1",
        "control_freq": 10,
        "initial_pose": [0.145, -0.746, 0.400, 1.312, 1.183, -1.233],
    },

    # ---------------------------
    #    GAMEPAD CONFIGURATION
    # ---------------------------
    "gamepad": {
        "deadzone": 0.15,
        "linear_speed_max": 0.12,
        "angular_speed_max": 0.25,
        "axis_map": {
            "LEFT_STICK_X": 0,
            "LEFT_STICK_Y": 1,
            "RIGHT_STICK_X": 3,
            "RIGHT_STICK_Y": 4,
            "LT": 2,
            "RT": 5,
        },
        "button_map": {
            "A": 0,
            "B": 1,
            "X": 2,
            "Y": 3,
            "LB": 4,
            "RB": 5,
            "BACK": 6,
            "START": 7,
            "RESET_POSE": 3, # Y button
            "DELETE_LAST": 1 # B button
        }
    },

    # ---------------------------
    #    DATA COLLECTION CONFIG
    # ---------------------------
    "data_collection": {
        "high_freq_hz": 100,
        "low_freq_hz": 10,
        "default_output_dir": "/mnt/sda/datasets/real_data/4",
    },

    "calibration": {
        "camera_to_base": np.eye(4).tolist(),
    },
    "task": {
        "max_steps": 200,
        "task_instruction": "Dispense a suitable amount of hand sanitizer onto one hand from the press bottle.",
        "gripper_threshold": 0.75,
    },
    "force_stats": {
        "mean": None,
        "std": None,
    },
}

#!/usr/bin/env python3
"""
多相机使用示例
"""

import time
import cv2
import numpy as np
from realsense_interface import MultiRealsense


def main():
    print("初始化多相机系统...")
    
    with MultiRealsense(
        resolution=(1280, 720),
        capture_fps=30,
        enable_color=True,
        enable_depth=False,
        verbose=True
    ) as cameras:
        print(f"成功初始化 {cameras.n_cameras} 个相机")
        
        # 设置所有相机参数
        cameras.set_exposure(exposure=120, gain=0)
        cameras.set_white_balance(white_balance=5900)
        
        # 获取所有相机内参
        intrinsics = cameras.get_intrinsics()
        print(f"相机内参形状: {intrinsics.shape}")
        for i, intr in enumerate(intrinsics):
            print(f"相机 {i} 内参:\n{intr}")
        
        # 开始录制 (每个相机一个文件)
        # cameras.start_recording("multi_camera_output", start_time=time.time() + 1)
        
        print("开始获取图像数据，按 'q' 退出...")
        
        # 创建显示窗口
        for i in range(cameras.n_cameras):
            cv2.namedWindow(f'Camera {i}', cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f'Camera {i}', 640, 360)
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            try:
                # 获取所有相机数据
                data = cameras.get()
                frame_count += 1
                
                # 显示每个相机的图像
                for cam_idx, cam_data in data.items():
                    if 'color' in cam_data:
                        img = cam_data['color']
                        cv2.imshow(f'Camera {cam_idx}', img)
                
                # 计算FPS
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    print(f"FPS: {fps:.1f}")
                
                # 检查退出条件
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                    
            except KeyboardInterrupt:
                break
        
        # 停止录制
        # cameras.stop_recording()
        
        cv2.destroyAllWindows()
        print(f"总共处理了 {frame_count} 帧")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
单相机使用示例
"""

import time
import cv2
import numpy as np
from multiprocessing.managers import SharedMemoryManager
from realsense_interface import SingleRealsense


def main():
    # 获取连接的相机序列号
    serials = SingleRealsense.get_connected_devices_serial()
    if not serials:
        print("未发现RealSense相机")
        return
    
    print(f"发现相机: {serials}")
    
    with SharedMemoryManager() as shm_manager:
        with SingleRealsense(
            shm_manager=shm_manager,
            serial_number=serials[0],
            capture_fps=30,
            enable_color=True,
            enable_depth=False,
            verbose=True
        ) as camera:
            print("相机初始化完成")
            
            # 设置相机参数
            camera.set_exposure(exposure=120, gain=0)
            camera.set_white_balance(white_balance=5900)
            
            # 获取相机内参
            intrinsics = camera.get_intrinsics()
            print(f"相机内参矩阵:\n{intrinsics}")
            
            # 开始录制视频 (可选)
            # camera.start_recording("output.mp4", start_time=time.time() + 1)
            
            print("开始获取图像数据，按 'q' 退出...")
            
            # 获取图像数据
            frame_count = 0
            start_time = time.time()
            
            while True:
                try:
                    data = camera.get()
                    frame_count += 1
                    
                    # 显示图像
                    if 'color' in data:
                        # RealSense输出BGR格式
                        img = data['color']
                        cv2.imshow('RealSense Camera', img)
                        
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
            
            # 停止录制 (如果开始了录制)
            # camera.stop_recording()
            
            cv2.destroyAllWindows()
            print(f"总共处理了 {frame_count} 帧")


if __name__ == "__main__":
    main()
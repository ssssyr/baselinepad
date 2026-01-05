#!/usr/bin/env python3
"""
相机测试工具
"""

import argparse
import time
import json
from multiprocessing.managers import SharedMemoryManager
from realsense_interface import SingleRealsense


def test_camera_detection():
    """测试相机检测"""
    print("检测RealSense相机...")
    serials = SingleRealsense.get_connected_devices_serial()
    
    if not serials:
        print("❌ 未发现RealSense相机")
        return False
    
    print(f"✅ 发现 {len(serials)} 个相机:")
    for i, serial in enumerate(serials):
        print(f"  {i}: {serial}")
    
    return True


def test_camera_basic(serial_number, config_file=None):
    """测试基本相机功能"""
    print(f"测试相机 {serial_number}...")
    
    # 加载配置文件
    config = None
    if config_file:
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            print(f"✅ 加载配置文件: {config_file}")
        except Exception as e:
            print(f"❌ 加载配置文件失败: {e}")
            return False
    
    with SharedMemoryManager() as shm_manager:
        try:
            with SingleRealsense(
                shm_manager=shm_manager,
                serial_number=serial_number,
                resolution=(640, 480),
                capture_fps=30,
                enable_color=True,
                enable_depth=False,
                advanced_mode_config=config,
                verbose=True
            ) as camera:
                print("✅ 相机初始化成功")
                
                # 测试参数设置
                camera.set_exposure(exposure=120, gain=0)
                camera.set_white_balance(white_balance=5900)
                print("✅ 相机参数设置成功")
                
                # 获取内参
                intrinsics = camera.get_intrinsics()
                print(f"✅ 相机内参: fx={intrinsics[0,0]:.1f}, fy={intrinsics[1,1]:.1f}")
                
                # 测试数据获取
                print("测试数据获取...")
                for i in range(10):
                    data = camera.get()
                    if 'color' in data:
                        shape = data['color'].shape
                        print(f"  帧 {i}: {shape}")
                    time.sleep(0.1)
                
                print("✅ 数据获取测试成功")
                return True
                
        except Exception as e:
            print(f"❌ 相机测试失败: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description='RealSense相机测试工具')
    parser.add_argument('--serial', '-s', help='指定相机序列号')
    parser.add_argument('--config', '-c', help='配置文件路径')
    parser.add_argument('--list-only', '-l', action='store_true', help='仅列出相机')
    
    args = parser.parse_args()
    
    # 检测相机
    if not test_camera_detection():
        return 1
    
    if args.list_only:
        return 0
    
    # 获取要测试的相机
    serials = SingleRealsense.get_connected_devices_serial()
    
    if args.serial:
        if args.serial not in serials:
            print(f"❌ 指定的相机序列号 {args.serial} 未找到")
            return 1
        test_serial = args.serial
    else:
        test_serial = serials[0]
        print(f"使用第一个相机: {test_serial}")
    
    # 测试相机
    if test_camera_basic(test_serial, args.config):
        print("🎉 所有测试通过!")
        return 0
    else:
        print("❌ 测试失败")
        return 1


if __name__ == "__main__":
    exit(main())
#!/usr/bin/env python3
"""
读取当前机械臂位置
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "real" / "third_party"))

from hardware.ur10_manager import UR10Manager

def main():
    print("正在连接机械臂...")
    config = {
        "ip": "192.168.1.50",
        "gripper_ip": "192.168.1.1",
        "control_freq": 10,
    }

    try:
        robot = UR10Manager(
            robot_ip=config["ip"],
            gripper_ip=config["gripper_ip"],
            control_freq=config["control_freq"]
        )

        pose, gripper = robot.get_tcp_pose()
        print(f"\n当前机械臂位置 (x, y, z, rx, ry, rz):")
        print(f"[{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}, {pose[3]:.3f}, {pose[4]:.3f}, {pose[5]:.3f}]")
        print(f"\n当前夹爪状态: {gripper:.3f}")

        print("\n\n请将上述位置复制到 configs/ur10_config.py 中的 initial_pose")

        robot.disconnect()

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

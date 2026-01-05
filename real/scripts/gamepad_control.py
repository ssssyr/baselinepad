#!/usr/bin/env python3
"""
手柄控制机械臂和夹爪主程序

使用方法:
python gamepad_control.py --robot_ip 192.168.1.50
python gamepad_control.py --robot_ip 192.168.1.50 --gripper_ip 192.168.1.1
"""

import time
import json
import click
import cv2
import numpy as np
import sys
from pathlib import Path

# 添加父目录到 Python 路径，以便导入 hardware 模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from hardware.robot_controller import RobotController
from hardware.gripper_controller import GripperController
from scripts.gamepad_handler import GamepadHandler


def load_config(config_file="config.json"):
    """加载配置文件"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"配置文件 {config_file} 未找到，使用默认配置")
        return {}


def create_display_image(robot_controller, gripper_controller, gamepad_handler, config):
    """创建显示图像"""
    # 创建一个黑色背景图像
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    
    # 获取当前状态
    current_pose = robot_controller.get_current_pose()
    robot_force = robot_controller.get_force_feedback()
    
    gripper_force = None
    if gripper_controller:
        gripper_force = gripper_controller.get_force_feedback()
    
    # 显示文本信息
    text_lines = [
        "手柄控制机械臂系统",
        "",
        f"机械臂位姿:",
        f"  X: {current_pose[0]:.3f}  Y: {current_pose[1]:.3f}  Z: {current_pose[2]:.3f}",
        f"  RX: {current_pose[3]:.3f}  RY: {current_pose[4]:.3f}  RZ: {current_pose[5]:.3f}",
        "",
    ]
    
    # 夹爪信息
    if gripper_controller and gripper_force:
        text_lines.extend([
            f"夹爪状态:",
            f"  宽度: {gripper_force['gripper_width']:.1f} mm",
            f"  抓取检测: {'是' if gripper_force['grip_detected'] else '否'}",
            "",
        ])
    
    # 力反馈信息
    if robot_force['available'] or (gripper_force and gripper_force['available']):
        text_lines.append("力反馈:")
        
        if gripper_force and gripper_force['available']:
            text_lines.extend([
                f"  夹爪力: {gripper_force['force_magnitude']:.2f} N",
                f"  夹爪力矩: {gripper_force['torque_magnitude']:.2f} Nm",
            ])
        
        if robot_force['available']:
            text_lines.extend([
                f"  机械臂力: {robot_force['force_magnitude']:.2f} N",
                f"  机械臂力矩: {robot_force['torque_magnitude']:.2f} Nm",
            ])
        
        text_lines.append("")
    
    # 控制说明
    text_lines.extend([
        "控制说明:",
        "  左摇杆: X/Y轴移动",
        "  十字键: Z轴移动",
        "  右摇杆: 工具旋转",
        "  LB/RB: 夹爪开合",
        "  LT/RT: 减速/加速",
        "  X键: 重置位置",
        "  Start键: 退出",
    ])
    
    # 绘制文本
    y_offset = 30
    for line in text_lines:
        cv2.putText(
            img, line, (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )
        y_offset += 25
    
    # 绘制力反馈条形图
    if gripper_force and gripper_force['available']:
        # 夹爪力反馈条
        bar_x = 600
        bar_y = 100
        bar_width = 150
        bar_height = 20
        
        max_force = 100.0
        force_ratio = min(gripper_force['force_magnitude'] / max_force, 1.0)
        
        # 背景
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        # 力条
        if force_ratio > 0:
            fill_width = int(bar_width * force_ratio)
            color_r = int(255 * force_ratio)
            color_g = int(255 * (1 - force_ratio))
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), (0, color_g, color_r), -1)
        
        # 边框
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
        
        # 标签
        cv2.putText(img, f"Force: {gripper_force['force_magnitude']:.1f}N", 
                   (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img


@click.command()
@click.option('--robot_ip', '-r', required=True, help="机械臂IP地址")
@click.option('--gripper_ip', '-g', default=None, help="夹爪IP地址（可选）")
@click.option('--config', '-c', default="config.json", help="配置文件路径")
@click.option('--frequency', '-f', default=None, type=float, help="控制频率")
@click.option('--sensitivity', '-s', default=None, type=float, help="控制灵敏度")
def main(robot_ip, gripper_ip, config, frequency, sensitivity):
    """手柄控制机械臂和夹爪主程序"""
    
    print("=" * 60)
    print("手柄控制机械臂系统")
    print("=" * 60)
    
    # 加载配置
    config_data = load_config(config)
    robot_config = config_data.get('robot', {})
    gripper_config = config_data.get('gripper', {})
    gamepad_config = config_data.get('gamepad', {})
    
    # 使用命令行参数覆盖配置
    if frequency is not None:
        robot_config['frequency'] = frequency
    if sensitivity is not None:
        robot_config['sensitivity'] = sensitivity
    
    # 设置默认值
    frequency = robot_config.get('frequency', 10.0)
    sensitivity = robot_config.get('sensitivity', 0.01)
    
    dt = 1.0 / frequency
    
    # 初始化组件
    gripper = None  # 确保 finally 里始终可用
    try:
        # 初始化手柄
        gamepad = GamepadHandler(
            deadzone=robot_config.get('deadzone', 0.1),
            trigger_speed_mult=gamepad_config.get('trigger_speed_mult', 2.0),
            trigger_slow_mult=gamepad_config.get('trigger_slow_mult', 0.1)
        )
        
        # 初始化机械臂
        robot = RobotController(
            robot_ip=robot_ip,
            frequency=frequency,
            initial_pose=robot_config.get('initial_pose')
        )
        
        # 初始化夹爪（如果提供IP）
        gripper = None
        if gripper_ip:
            try:
                gripper = GripperController(
                    gripper_ip=gripper_ip,
                    unit=gripper_config.get('unit', 65),
                    port=gripper_config.get('port', 502),
                    max_width=gripper_config.get('max_width', 110.0),
                    max_force=gripper_config.get('max_force', 50.0)
                )
            except Exception as e:
                print(f"夹爪初始化失败: {e}")
                gripper = None
        
        print("\n系统初始化完成！")
        print("按Start键退出程序")
        print("=" * 60)
        
        # 创建可视化窗口（显式创建可提高在不同桌面环境下的稳定性）
        win_name = '手柄控制机械臂'
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 800, 600)
        
        # 主控制循环
        t_start = time.monotonic()
        iter_idx = 0
        
        while True:
            t_cycle_end = t_start + (iter_idx + 1) * dt
            
            # 更新手柄状态
            gamepad.update()
            
            # 检查控制按钮
            control_buttons = gamepad.get_control_buttons()
            if control_buttons['exit']:
                print("退出程序")
                break
            
            if control_buttons['reset']:
                robot.reset_to_initial()
                print("重置到初始位置")
            
            # 获取移动输入
            delta_pos, current_sensitivity = gamepad.get_movement_input(sensitivity)
            if np.any(delta_pos != 0):
                robot.move_by_delta(delta_pos)
            
            # 获取旋转输入
            rotation_speed = robot_config.get('rotation_speed', 0.02)
            delta_rotation = gamepad.get_rotation_input(rotation_speed)
            if abs(delta_rotation) > 0:
                robot.rotate_tool_z(delta_rotation)
            
            # 获取夹爪输入
            if gripper:
                gripper_input = gamepad.get_gripper_input()
                if gripper_input is not None:
                    gripper.set_position(gripper_input)
                
                # 更新夹爪状态
                gripper.read_status()
            
            # 创建并显示图像
            display_img = create_display_image(robot, gripper, gamepad, config_data)
            cv2.imshow(win_name, display_img)
            
            # 检查窗口关闭
            if cv2.waitKey(1) & 0xFF == 27:  # ESC键
                break
            
            # 等待下一个周期
            current_time = time.monotonic()
            if current_time < t_cycle_end:
                time.sleep(t_cycle_end - current_time)
            
            iter_idx += 1
    
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序运行错误: {e}")
    finally:
        # 清理资源
        try:
            gamepad.cleanup()
        except:
            pass
        
        try:
            robot.cleanup()
        except:
            pass
        
        if gripper:
            try:
                gripper.cleanup()
            except:
                pass
        
        cv2.destroyAllWindows()
        print("程序已退出")


if __name__ == '__main__':
    main()
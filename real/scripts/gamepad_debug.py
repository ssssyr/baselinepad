#!/usr/bin/env python3
"""详细的手柄调试工具，显示所有输入状态"""
import pygame
import sys

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("未检测到手柄！")
    sys.exit(1)

joy = pygame.joystick.Joystick(0)
joy.init()

print(f"\n=== 手柄信息 ===")
print(f"名称: {joy.get_name()}")
print(f"按键数量: {joy.get_numbuttons()}")
print(f"摇杆数量: {joy.get_numaxes()}")
print(f"方向键数量: {joy.get_numhats()}")
print(f"\n请按动手柄/摇杆，观察数值变化（按 Ctrl+C 退出）\n")

try:
    while True:
        pygame.event.pump()

        print("\033[H\033[J", end="")  # 清屏

        # 显示摇杆状态
        print("=== 摇杆 ===")
        axis_names = ["LX", "LY", "LT", "RX", "RY", "RT"]
        for i in range(joy.get_numaxes()):
            val = joy.get_axis(i)
            name = axis_names[i] if i < len(axis_names) else f"A{i}"
            bar = "█" * int((val + 1) * 10) if val >= 0 else "█" * int((1 + val) * 10)
            print(f"  {name}: {val:+7.3f} [{bar:20s}]")

        # 显示按键状态
        print("\n=== 按键 ===")
        button_names = ["A", "B", "X", "Y", "LB", "RB", "BACK", "START", "L3", "R3", "GUIDE"]
        for i in range(min(joy.get_numbuttons(), len(button_names))):
            state = "●" if joy.get_button(i) else "○"
            print(f"  {button_names[i]:6s}: {state}")

        # 显示方向键
        print("\n=== 方向键 ===")
        hat = joy.get_hat(0)
        hat_arrows = ""
        if hat[1] == 1: hat_arrows += "↑"
        if hat[1] == -1: hat_arrows += "↓"
        if hat[0] == -1: hat_arrows += "←"
        if hat[0] == 1: hat_arrows += "→"
        if not hat_arrows: hat_arrows = "●"
        print(f"  DPAD: {hat_arrows} ({hat})")

        # 显示计算后的速度命令
        print("\n=== 计算的速度命令 ===")
        deadzone = 0.15
        lx = joy.get_axis(0)
        ly = -joy.get_axis(1)
        rx = -joy.get_axis(4)
        ry = joy.get_axis(3)

        lin_x = lx * 0.25 if abs(lx) > deadzone else 0
        lin_y = ly * 0.25 if abs(ly) > deadzone else 0
        lin_z = hat[1] * 0.25

        ang_x = rx * 0.5 if abs(rx) > deadzone else 0
        ang_y = ry * 0.5 if abs(ry) > deadzone else 0

        lt = (joy.get_axis(2) + 1) / 2
        rt = (joy.get_axis(5) + 1) / 2
        ang_z = (rt - lt) * 0.5

        print(f"  线速度: [{lin_x:+.3f}, {lin_y:+.3f}, {lin_z:+.3f}]")
        print(f"  角速度: [{ang_x:+.3f}, {ang_y:+.3f}, {ang_z:+.3f}]")
        print(f"  夹爪: {'OPEN' if joy.get_button(5) else ('CLOSED' if joy.get_button(4) else 'HOLD')}")

except KeyboardInterrupt:
    print("\n\n测试结束")

pygame.quit()

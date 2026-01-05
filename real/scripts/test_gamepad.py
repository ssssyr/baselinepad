#!/usr/bin/env python3
"""测试 pygame 是否能检测到手柄"""
import pygame

pygame.init()
pygame.joystick.init()

count = pygame.joystick.get_count()
print(f"检测到手柄数量: {count}")

if count == 0:
    print("未检测到手柄！")
    print("请检查:")
    print("  1. 手柄是否已连接")
    print("  2. 用户是否在 input 组中: sudo usermod -a -G input $USER")
    print("  3. 运行 newgrp input 使更改生效")
else:
    joy = pygame.joystick.Joystick(0)
    joy.init()
    print(f"手柄名称: {joy.get_name()}")
    print(f"按键数量: {joy.get_numbuttons()}")
    print(f"摇杆数量: {joy.get_numaxes()}")
    print(f"方向键数量: {joy.get_numhats()}")

    print("\n实时按键测试（按 Ctrl+C 退出）:")
    try:
        while True:
            for event in pygame.event.get():
                pass

            # 显示摇杆状态
            axes = []
            for i in range(joy.get_numaxes()):
                axes.append(joy.get_axis(i))
            print(f"摇杆: {axes}", end='\r')

            # 显示按键状态
            buttons = []
            for i in range(min(10, joy.get_numbuttons())):
                buttons.append(int(joy.get_button(i)))
            print(f" 按键: {buttons}      ", end='\r')
    except KeyboardInterrupt:
        print("\n测试结束")

pygame.quit()

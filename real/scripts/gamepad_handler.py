"""
手柄输入处理模块
"""
import os
import pygame
import numpy as np

class GamepadHandler:
    """手柄输入处理类，专为速度控制的遥操作设计"""
    
    def __init__(self, config):
        """初始化手柄处理器，从配置字典中读取参数"""
        self.config = config
        self.deadzone = self.config['deadzone']
        self.linear_speed_max = self.config['linear_speed_max']
        self.angular_speed_max = self.config['angular_speed_max']
        self.axis_map = self.config['axis_map']
        self.button_map = self.config['button_map']
        
        self.button_states = {}
        self.prev_button_states = {}
        self.hat_states = {}
        
        # Set a dummy video driver BEFORE pygame.init() to prevent it from
        # interacting with the display system, which can cause blocking issues.
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        pygame.init()
        pygame.joystick.init()
        
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("未检测到手柄设备")
        
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        
        print(f"检测到手柄: {self.joystick.get_name()}")
    
    def update(self):
        """更新手柄状态，应在每个控制循环开始时调用"""
        # Using a non-blocking event loop is safer than pump() in some environments.
        for event in pygame.event.get():
            pass
        
        self.prev_button_states = self.button_states.copy()
        num_buttons = self.joystick.get_numbuttons()
        for i in range(num_buttons):
            self.button_states[i] = self.joystick.get_button(i)
        
        num_hats = self.joystick.get_numhats()
        for i in range(num_hats):
            self.hat_states[i] = self.joystick.get_hat(i)

    def get_teleop_velocity(self):
        """获取用于 speedL 的 6D 速度向量 [vx, vy, vz, rx, ry, rz]"""
        lin_vel = np.zeros(3)
        ang_vel = np.zeros(3)

        vx = -self.joystick.get_axis(self.axis_map['LEFT_STICK_X'])  # 水平反向以实现镜像操作
        vy = self.joystick.get_axis(self.axis_map['LEFT_STICK_Y']) # Y-axis inverted for mirrored operation
        if np.linalg.norm([vx, vy]) > self.deadzone:
            lin_vel[0] = vx * self.linear_speed_max
            lin_vel[1] = vy * self.linear_speed_max

        _, hat_y = self.hat_states.get(0, (0, 0))
        lin_vel[2] = hat_y * self.linear_speed_max

        rx = -self.joystick.get_axis(self.axis_map['RIGHT_STICK_Y'])
        ry = self.joystick.get_axis(self.axis_map['RIGHT_STICK_X'])
        if np.linalg.norm([rx, ry]) > self.deadzone:
            ang_vel[0] = rx * self.angular_speed_max
            ang_vel[1] = ry * self.angular_speed_max

        lt = (self.joystick.get_axis(self.axis_map['LT']) + 1) / 2
        rt = (self.joystick.get_axis(self.axis_map['RT']) + 1) / 2
        ang_vel[2] = (rt - lt) * self.angular_speed_max

        return lin_vel, ang_vel

    def get_gripper_input(self):
        """获取夹爪输入 (0.0=闭合, 1.0=张开)"""
        lb_pressed = self.button_states.get(self.button_map['LB'], 0)
        rb_pressed = self.button_states.get(self.button_map['RB'], 0)
        
        if rb_pressed:
            return 1.0
        elif lb_pressed:
            return 0.0
        return None

    def get_control_buttons(self):
        """获取控制按钮的按下事件"""
        return {
            'toggle_recording': self._is_button_just_pressed(self.button_map['START']),
            'save_episode': self._is_button_just_pressed(self.button_map['A']),
            'exit': self._is_button_just_pressed(self.button_map['BACK']),
            'reset_pose': self._is_button_just_pressed(self.button_map['RESET_POSE']),
            'delete_last': self._is_button_just_pressed(self.button_map.get('DELETE_LAST')),
        }

    def _is_button_just_pressed(self, button_id):
        """检测按钮是否在这一帧刚被按下"""
        return self.button_states.get(button_id) and not self.prev_button_states.get(button_id)

    def rumble(self, low_frequency=0.5, high_frequency=0.5, duration_ms=200):
        """
        触发手柄振动

        Args:
            low_frequency: 低频电机强度 (0.0-1.0)
            high_frequency: 高频电机强度 (0.0-1.0)
            duration_ms: 振动持续时间（毫秒）
        """
        try:
            # pygame 2.0+ 支持振动
            if hasattr(self.joystick, 'rumble'):
                self.joystick.rumble(low_frequency, high_frequency, duration_ms)
                return True
            else:
                print("当前 pygame 版本不支持振动功能")
                return False
        except Exception as e:
            print(f"振动触发失败: {e}")
            return False

    def cleanup(self):
        """清理 pygame 资源"""
        pygame.quit()

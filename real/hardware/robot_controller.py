"""
机械臂控制模块
"""
import time
import numpy as np
try:
    import rtde_control
    import rtde_receive
    RTDE_AVAILABLE = True
except ImportError:
    RTDE_AVAILABLE = False
    print("警告: RTDE库未安装，将使用模拟模式")

try:
    from .ft_sensor import FTSensor
    FT_SENSOR_AVAILABLE = True
except ImportError:
    FT_SENSOR_AVAILABLE = False
    print("警告: 力传感器模块未安装")


class RobotController:
    """机械臂控制类"""

    def __init__(self, robot_ip, frequency=10.0, initial_pose=None,
                 move_velocity=0.5, move_acceleration=0.3, ft_sensor_ip=None):
        """
        Initializes the RobotController.

        Args:
            robot_ip (str): IP address of the robot.
            frequency (float): Control frequency in Hz.
            initial_pose (list): Initial pose for simulation mode.
            move_velocity (float): Movement velocity in m/s (default: 0.5).
            move_acceleration (float): Movement acceleration in m/s^2 (default: 0.3).
            ft_sensor_ip (str): IP address of the force/torque sensor (default: None).
        """
        self.robot_ip = robot_ip
        self.frequency = frequency
        self.dt = 1.0 / frequency
        self.move_velocity = move_velocity
        self.move_acceleration = move_acceleration

        # 初始化力传感器
        self.ft_sensor = None
        self.use_ft_sensor = False
        if ft_sensor_ip and FT_SENSOR_AVAILABLE:
            try:
                self.ft_sensor = FTSensor(ip=ft_sensor_ip)
                self.use_ft_sensor = self.ft_sensor.connected
                if self.use_ft_sensor:
                    print(f"[RobotController] 使用真实力传感器: {ft_sensor_ip}")
                else:
                    print(f"[RobotController] 力传感器连接失败，使用UR估算力")
            except Exception as e:
                print(f"[RobotController] 力传感器初始化失败: {e}")

        if RTDE_AVAILABLE:
            try:
                # 初始化RTDE连接
                self.rtde_c = rtde_control.RTDEControlInterface(robot_ip)
                self.rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
                
                # 获取当前位姿
                current_pose = self.rtde_r.getActualTCPPose()
                self.target_pose = np.array(current_pose)
                
                if initial_pose is not None:
                    self.initial_pose = np.array(initial_pose)
                else:
                    self.initial_pose = self.target_pose.copy()
                
                self.connected = True
                print(f"机械臂连接成功: {robot_ip}")
                print(f"当前位姿: {self.target_pose}")
                
            except Exception as e:
                print(f"机械臂连接失败: {e}")
                self.connected = False
                self._init_simulation_mode(initial_pose)
        else:
            self.connected = False
            self._init_simulation_mode(initial_pose)
    
    def _init_simulation_mode(self, initial_pose):
        """初始化模拟模式"""
        if initial_pose is not None:
            self.target_pose = np.array(initial_pose)
            self.initial_pose = np.array(initial_pose)
        else:
            # 默认位姿
            self.target_pose = np.array([-0.074, 0.661, 0.002, -2.211, -2.170, -0.014])
            self.initial_pose = self.target_pose.copy()
        
        print("使用模拟模式")
        print(f"模拟位姿: {self.target_pose}")
    
    def get_current_pose(self):
        """获取当前位姿"""
        if self.connected:
            try:
                return np.array(self.rtde_r.getActualTCPPose())
            except:
                return self.target_pose
        else:
            return self.target_pose
    
    def move_by_delta(self, delta_pos):
        """按增量移动"""
        self.target_pose[:3] += delta_pos
        return self._execute_move()
    
    def rotate_tool_z(self, delta_rotation):
        """绕工具坐标系Z轴旋转"""
        if self.connected:
            try:
                # 使用RTDE的工具旋转功能
                current_pose = self.rtde_r.getActualTCPPose()

                # 计算新的旋转
                # 这里简化处理，实际应该使用旋转矩阵
                new_pose = current_pose.copy()
                new_pose[5] += delta_rotation  # 绕Z轴旋转

                # 执行移动，使用配置的速度和加速度
                self.rtde_c.moveL(new_pose, self.move_velocity, self.move_acceleration, False)
                self.target_pose = np.array(new_pose)
                return self.target_pose
            except Exception as e:
                print(f"旋转执行失败: {e}")
                return None
        else:
            # 模拟模式
            self.target_pose[5] += delta_rotation
            return self.target_pose
    
    def reset_to_initial(self):
        """重置到初始位姿"""
        self.target_pose = self.initial_pose.copy()
        return self._execute_move()
    
    def _execute_move(self):
        """执行移动命令"""
        if self.connected:
            try:
                # 使用moveL进行线性移动，使用配置的速度和加速度
                self.rtde_c.moveL(
                    self.target_pose.tolist(),
                    self.move_velocity,
                    self.move_acceleration,
                    False
                )
                return True
            except Exception as e:
                print(f"移动执行失败: {e}")
                return False
        else:
            # 模拟模式，直接返回成功
            return True
    
    def set_tcp_speed(self, velocity_vector, acceleration=0.5, time_val=0.2):
        """
        Sets the TCP speed using speedL for real-time control. This is non-blocking.

        Args:
            velocity_vector (list or np.ndarray): 6D velocity vector [vx, vy, vz, rx, ry, rz].
            acceleration (float): Tool acceleration [m/s^2].
            time_val (float): Time [s] before the function returns. Should be > controller's dt.
        """
        if self.connected:
            try:
                self.rtde_c.speedL(list(velocity_vector), acceleration, time_val)
                return True
            except Exception as e:
                # This can happen if the robot is stopped; avoid flooding the console.
                # print(f"Failed to set TCP speed: {e}")
                return False
        else:
            # In simulation, approximate the movement
            self.target_pose += np.array(velocity_vector) * self.dt
            return True

    def get_force_feedback(self):
        """获取力反馈 - 优先使用真实力传感器"""
        # 优先使用真实力传感器
        if self.use_ft_sensor and self.ft_sensor:
            try:
                ft_data = self.ft_sensor.read()
                force_magnitude = np.linalg.norm(ft_data[:3])
                torque_magnitude = np.linalg.norm(ft_data[3:])
                return {
                    'force': ft_data[:3],
                    'torque': ft_data[3:],
                    'force_magnitude': force_magnitude,
                    'torque_magnitude': torque_magnitude,
                    'available': True,
                    'source': 'ft_sensor'  # 标记数据来源
                }
            except Exception as e:
                print(f"[RobotController] 力传感器读取失败: {e}")

        # 回退到UR估算的力
        if self.connected:
            try:
                tcp_force = self.rtde_r.getActualTCPForce()
                if tcp_force is not None:
                    tcp_force = np.array(tcp_force)
                    force_magnitude = np.linalg.norm(tcp_force[:3])
                    torque_magnitude = np.linalg.norm(tcp_force[3:])
                    return {
                        'force': tcp_force[:3],
                        'torque': tcp_force[3:],
                        'force_magnitude': force_magnitude,
                        'torque_magnitude': torque_magnitude,
                        'available': True
                    }
            except:
                pass
        
        # 返回空的力反馈
        return {
            'force': np.array([0.0, 0.0, 0.0]),
            'torque': np.array([0.0, 0.0, 0.0]),
            'force_magnitude': 0.0,
            'torque_magnitude': 0.0,
            'available': False
        }
    
    def cleanup(self):
        """清理连接"""
        # 断开力传感器连接
        if self.ft_sensor:
            try:
                self.ft_sensor.disconnect()
            except:
                pass

        if self.connected:
            try:
                # 停止控制脚本
                if hasattr(self, 'rtde_c') and self.rtde_c is not None:
                    self.rtde_c.stopScript()
                    self.rtde_c.disconnect()

                # 断开接收连接
                if hasattr(self, 'rtde_r') and self.rtde_r is not None:
                    self.rtde_r.disconnect()

                print("机械臂连接已完全关闭")
            except Exception as e:
                print(f"清理连接时出错: {e}")
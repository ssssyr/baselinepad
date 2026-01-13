"""
ur10_manager.py

Encapsulates the control logic for the UR10 robot arm and gripper, providing a 
high-level API by wrapping the existing RobotController and GripperController.
"""

import time
import numpy as np

# Import the actual controller implementations
from .robot_controller import RobotController
from .gripper_controller import GripperController

class UR10Manager:
    """
    A high-level manager for the UR10 robot and its gripper.
    It uses RobotController and GripperController for the actual hardware communication.
    """
    def __init__(self, robot_ip, gripper_ip=None, control_freq=10.0, gripper_max_width=110.0, default_gripper_open=1.0):
        """
        Initializes the UR10Manager.

        Args:
            robot_ip (str): The IP address of the UR10 robot controller.
            gripper_ip (str, optional): The IP address of the gripper. Defaults to None.
            control_freq (float): The control frequency for the robot controller.
            gripper_max_width (float): The maximum opening width of the gripper in mm.
            default_gripper_open (float): Default gripper state when no gripper is connected
                                         (0.0=closed, 1.0=open). Defaults to 1.0.
        """
        if not robot_ip:
            raise ValueError("Robot IP address cannot be empty.")

        self.robot_ip = robot_ip
        self.gripper_ip = gripper_ip
        self.gripper_max_width = gripper_max_width
        self.default_gripper_open = default_gripper_open

        # Instantiate the low-level controllers, passing the control frequency
        # 如果有gripper_ip，也作为力传感器IP传递（OnRobot夹爪内置力传感器）
        self.robot_controller = RobotController(robot_ip=self.robot_ip, frequency=control_freq, ft_sensor_ip=gripper_ip)

        # Verify robot connection
        if not self.robot_controller.connected:
            raise ConnectionError(
                f"Failed to connect to UR10 robot at {robot_ip}. "
                f"Please check:\n"
                f"  1. Robot is powered on\n"
                f"  2. IP address is correct\n"
                f"  3. Network connection is working\n"
                f"  4. RTDE is enabled on the robot"
            )

        # Initialize gripper (optional)
        self.gripper_controller = None
        if self.gripper_ip:
            try:
                self.gripper_controller = GripperController(
                    gripper_ip=self.gripper_ip,
                    max_width=self.gripper_max_width
                )
                if not self.gripper_controller.connected:
                    print(f"警告：夹爪未连接，将使用默认夹爪状态")
                    self.gripper_controller = None
            except Exception as e:
                print(f"夹爪连接失败: {e}")
                print("将继续运行，但夹爪功能不可用")
                self.gripper_controller = None

    @property
    def is_connected(self):
        """Returns True if robot is connected."""
        return self.robot_controller.connected

    def connect(self):
        """Verifies connection status (already done in __init__)."""
        if not self.robot_controller.connected:
            raise ConnectionError(f"Robot at {self.robot_ip} is not connected.")
        if self.gripper_ip and self.gripper_controller is None:
            print(f"Warning: Gripper at {self.gripper_ip} is not connected.")
        print("UR10Manager is connected and ready.")

    def disconnect(self):
        """Cleans up connections for both robot and gripper."""
        print("Disconnecting hardware...")
        if self.robot_controller:
            self.robot_controller.cleanup()
        if self.gripper_controller:
            self.gripper_controller.cleanup()
        print("Hardware disconnected.")

    def get_tcp_pose(self):
        """
        Gets the current TCP pose and normalized gripper state.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: The pose as [x, y, z, rx, ry, rz].
                - float: The normalized gripper state (0.0 for closed, 1.0 for open).
        """
        if not self.robot_controller.connected:
            raise RuntimeError("Robot is not connected.")

        # Get robot pose
        robot_pose = self.robot_controller.get_current_pose()

        # Get gripper state
        if self.gripper_controller and self.gripper_controller.connected:
            # Read gripper status to get the latest width
            self.gripper_controller.read_status()
            gripper_info = self.gripper_controller.get_force_feedback()
            current_width_mm = gripper_info.get('gripper_width', 0.0)
            gripper_state_normalized = np.clip(current_width_mm / self.gripper_max_width, 0.0, 1.0)
        else:
            # Use default gripper state when no gripper is connected
            gripper_state_normalized = self.default_gripper_open

        return robot_pose, gripper_state_normalized

    def get_tcp_pose_with_ts(self):
        """
        Gets the current TCP pose, gripper state, AND timestamp.

        This is the recommended method for data collection, as it provides
        timing information needed for proper multi-modal alignment.

        Returns:
            tuple: (robot_pose, gripper_state, timestamp)
                - robot_pose: np.ndarray, [x, y, z, rx, ry, rz]
                - gripper_state: float, 0.0=closed, 1.0=open
                - timestamp: float, time.monotonic() when data was read
        """
        ts = time.monotonic()
        pose, gripper = self.get_tcp_pose()
        return pose, gripper, ts

    def get_force_torque(self):
        """
        Gets the force/torque data from the gripper's FT sensors.

        Returns:
            np.ndarray: An array of 6 values [Fx, Fy, Fz, Tx, Ty, Tz].
        """
        if not self.robot_controller.connected:
            raise RuntimeError("Robot is not connected.")

        # 使用夹爪的FT传感器数据（左右取平均）
        if self.gripper_controller and self.gripper_controller.connected:
            self.gripper_controller.read_status()
            ft_data = self.gripper_controller.get_force_feedback()
            if ft_data and ft_data['available']:
                return np.concatenate([ft_data['force'], ft_data['torque']])

        # 如果夹爪未连接，使用机器人的估算数据
        ft_data = self.robot_controller.get_force_feedback()
        if ft_data and ft_data['available']:
            return np.concatenate([ft_data['force'], ft_data['torque']])
        else:
            # Return zeros if force sensor data is not available
            return np.zeros(6)

    def get_force_torque_with_ts(self):
        """
        Gets the force/torque data AND timestamp.

        Returns:
            tuple: (force_torque, timestamp)
                - force_torque: np.ndarray, [Fx, Fy, Fz, Tx, Ty, Tz]
                - timestamp: float, time.monotonic() when data was read
        """
        ts = time.monotonic()
        ft = self.get_force_torque()
        return ft, ts

    def move_to_pose_sync(self, target_pose, timeout=5.0):
        """
        Moves the robot's TCP to a target pose and waits for completion.

        Args:
            target_pose (list or np.ndarray): The target pose [x, y, z, rx, ry, rz].
            timeout (float): Maximum seconds to wait for movement completion. Default: 5.0.

        Returns:
            bool: True if movement succeeded, False otherwise.
        """
        if not self.robot_controller.connected:
            raise RuntimeError("Robot is not connected.")

        print(f"Moving to pose: {np.round(target_pose, 3)}...")

        try:
            # 启动异步运动 (async=True 立即返回，避免无限阻塞)
            self.robot_controller.rtde_c.moveL(
                np.array(target_pose).tolist(),
                0.5,  # velocity (m/s)
                0.3,  # acceleration (m/s²)
                True  # async (立即返回，然后手动等待)
            )

            # 手动等待运动完成，带超时保护
            import time
            start_time = time.time()
            while time.time() - start_time < timeout:
                if self.robot_controller.rtde_c.isSteady():
                    print("Movement complete.")
                    return True
                time.sleep(0.05)  # 50ms 检查一次

            print(f"Warning: Movement timeout after {timeout}s (机器人可能无法到达目标位置)")
            return False

        except Exception as e:
            print(f"Warning: Movement failed to execute: {e}")
            return False

    def set_tcp_speed(self, velocity_vector, acceleration=0.5):
        """
        Sends a real-time speed command to the robot's TCP.

        Args:
            velocity_vector (list or np.ndarray): 6D velocity vector [vx, vy, vz, rx, ry, rz].
            acceleration (float): Tool acceleration [m/s^2].
        """
        if self.robot_controller and self.robot_controller.connected:
            # The time_val should be slightly larger than the control loop's dt
            # to ensure smooth command sending without buffer overruns.
            time_val = 1.5 / self.robot_controller.frequency # e.g., 1.5 * (1/100) = 0.015
            self.robot_controller.set_tcp_speed(velocity_vector, acceleration, time_val)

    def set_gripper(self, value):
        """
        Sets the state of the gripper.

        Args:
            value (float): The target gripper state (0.0 for closed, 1.0 for open).

        Returns:
            bool: True if command was sent, False if no gripper is connected.
        """
        if self.gripper_controller and self.gripper_controller.connected:
            # The gripper controller expects a value from 0.0 to 1.0
            print(f"Setting gripper to: {value:.2f}")
            result = self.gripper_controller.set_position(value)
            if result:
                print("Gripper command sent.")
            else:
                print("Gripper command failed.")
            return result
        else:
            if self.gripper_ip:
                print(f"Warning: Gripper at {self.gripper_ip} is not connected. Command ignored.")
            return False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

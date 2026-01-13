"""
夹爪控制模块
"""
import time
import numpy as np
try:
    from pymodbus.client import ModbusTcpClient as ModbusClient
    import struct
    MODBUS_AVAILABLE = True
except ImportError:
    MODBUS_AVAILABLE = False
    print("警告: pymodbus库未安装，夹爪功能将不可用")


class GripperController:
    """RG2-FT夹爪控制类"""
    
    def __init__(self, gripper_ip, unit=65, port=502, max_width=110.0, max_force=50.0):
        self.gripper_ip = gripper_ip
        self.unit = unit
        self.port = port
        self.max_width = max_width
        self.max_force = max_force
        
        # 状态变量
        self.current_width = 0.0
        self.gripper_busy = 0
        self.grip_detected = 0
        self.left_ft_sensor = np.zeros(6)
        self.right_ft_sensor = np.zeros(6)
        
        # 控制变量
        self.last_command_time = 0.0
        self.last_read_time = 0.0
        self.last_sent_width = 0.0
        
        if MODBUS_AVAILABLE:
            try:
                print(f"正在连接夹爪: {gripper_ip}:{port}...")
                self.modbus = ModbusClient(host=gripper_ip, port=port, timeout=2.0)
                connect_result = self.modbus.connect()
                
                # 检查连接状态（pymodbus 3.x 可能需要检查 socket 状态）
                is_connected = connect_result
                if hasattr(self.modbus, 'is_socket_open'):
                    is_connected = is_connected and self.modbus.is_socket_open()
                elif hasattr(self.modbus, 'socket'):
                    is_connected = is_connected and self.modbus.socket is not None
                
                if is_connected:
                    print(f"夹爪连接成功: {gripper_ip}:{port}")
                    # 先设置连接状态
                    self.connected = True
                    # 测试读取一个寄存器以验证连接
                    test_result = self.modbus.read_holding_registers(280, count=1, device_id=self.unit)
                    if test_result.isError():
                        print(f"警告: 连接成功但读取测试失败: {test_result}")
                        raise ConnectionError(f"无法读取夹爪数据: {test_result}")
                    # 读取初始状态
                    self.read_status()
                else:
                    raise ConnectionError(f"无法连接到夹爪: connect()={connect_result}, socket状态={is_connected}")
                    
            except Exception as e:
                print(f"夹爪连接失败: {e}")
                import traceback
                traceback.print_exc()
                self.connected = False
                self._init_simulation_mode()
        else:
            print("警告: pymodbus库未安装，夹爪功能将不可用")
            self.connected = False
            self._init_simulation_mode()
    
    def _init_simulation_mode(self):
        """初始化模拟模式"""
        self.current_width = self.max_width / 2  # 初始半开状态
        print("夹爪使用模拟模式")
    
    def read_status(self):
        """读取夹爪状态"""
        if not self.connected:
            return
        
        current_time = time.monotonic()
        if (current_time - self.last_read_time) < 0.2:  # 限制读取频率
            return
        
        try:
            # 读取夹爪宽度
            result = self.modbus.read_holding_registers(280, count=1, device_id=self.unit)
            if not result.isError():
                self.current_width = self._validate_int16(result) / 10
            
            # 读取忙碌状态
            result = self.modbus.read_holding_registers(281, count=1, device_id=self.unit)
            if not result.isError():
                self.gripper_busy = result.registers[0] if result.registers else 0
            
            # 读取抓取检测
            result = self.modbus.read_holding_registers(282, count=1, device_id=self.unit)
            if not result.isError():
                self.grip_detected = result.registers[0] if result.registers else 0
            
            # 读取左力传感器
            for i in range(6):
                result = self.modbus.read_holding_registers(259+i, count=1, device_id=self.unit)
                if not result.isError():
                    if i < 3:
                        self.left_ft_sensor[i] = self._validate_int16(result) / 10
                    else:
                        self.left_ft_sensor[i] = self._validate_int16(result) / 100
            
            # 读取右力传感器
            for i in range(6):
                result = self.modbus.read_holding_registers(268+i, count=1, device_id=self.unit)
                if not result.isError():
                    if i < 3:
                        self.right_ft_sensor[i] = self._validate_int16(result) / 10
                    else:
                        self.right_ft_sensor[i] = self._validate_int16(result) / 100
            
            self.last_read_time = current_time
            
        except Exception as e:
            # 静默处理错误
            pass
    
    def set_position(self, position):
        """设置夹爪位置 (0.0-1.0)"""
        target_width = position * self.max_width
        return self.set_width(target_width)
    
    def set_width(self, target_width_mm):
        """设置夹爪宽度 (mm)"""
        target_width_mm = np.clip(target_width_mm, 0.0, self.max_width)
        
        if not self.connected:
            # 模拟模式
            self.current_width = target_width_mm
            return True
        
        current_time = time.monotonic()
        
        # 检查是否需要发送新命令
        if abs(target_width_mm - self.last_sent_width) < 0.5:  # 0.5mm阈值
            return True
        
        if (current_time - self.last_command_time) < 0.1:  # 限制命令频率
            return False
        
        if self.gripper_busy:
            return False
        
        try:
            # 发送控制命令
            result1 = self.modbus.write_register(3, int(target_width_mm * 10), device_id=self.unit)
            if result1.isError():
                print(f"发送夹爪宽度命令失败: {result1}")
                return False
            
            result2 = self.modbus.write_register(2, int(self.max_force * 10), device_id=self.unit)
            if result2.isError():
                print(f"发送夹爪力度命令失败: {result2}")
                return False
            
            result3 = self.modbus.write_register(4, 1, device_id=self.unit)  # 激活运动
            if result3.isError():
                print(f"激活夹爪运动失败: {result3}")
                return False
            
            self.last_sent_width = target_width_mm
            self.last_command_time = current_time
            return True
            
        except Exception as e:
            print(f"发送夹爪命令失败: {e}")
            return False
    
    def stop(self):
        """停止夹爪运动"""
        if self.connected:
            try:
                result = self.modbus.write_register(4, 0, device_id=self.unit)
                if result.isError():
                    print(f"停止夹爪失败: {result}")
                    return False
                return True
            except:
                return False
        return True
    
    def get_force_feedback(self):
        """获取力反馈信息"""
        if not self.connected:
            return {
                'force': np.array([0.0, 0.0, 0.0]),
                'torque': np.array([0.0, 0.0, 0.0]),
                'force_magnitude': 0.0,
                'torque_magnitude': 0.0,
                'gripper_width': self.current_width,
                'grip_detected': 0,
                'available': False
            }

        # 左右传感器数据取平均
        left_force = self.left_ft_sensor[:3]
        right_force = self.right_ft_sensor[:3]
        total_force = (left_force + right_force) / 2

        left_torque = self.left_ft_sensor[3:]
        right_torque = self.right_ft_sensor[3:]
        total_torque = (left_torque + right_torque) / 2

        force_magnitude = np.linalg.norm(total_force)
        torque_magnitude = np.linalg.norm(total_torque)

        return {
            'force': total_force,
            'torque': total_torque,
            'force_magnitude': force_magnitude,
            'torque_magnitude': torque_magnitude,
            'left_force': left_force,
            'right_force': right_force,
            'gripper_width': self.current_width,
            'grip_detected': self.grip_detected,
            'available': True
        }
    
    def _validate_int16(self, instance):
        """验证并解码16位整数（兼容pymodbus 3.x）"""
        if instance is None or instance.isError() or not instance.registers:
            return 0.0

        try:
            # pymodbus 3.x: 直接获取寄存器值并转换为有符号16位整数
            value = instance.registers[0]
            # 转换为有符号16位整数（处理负值）
            if value > 32767:
                value = value - 65536
            return float(value)
        except:
            return 0.0
    
    def cleanup(self):
        """清理连接"""
        if self.connected:
            try:
                self.stop()
                self.modbus.close()
                print("夹爪连接已关闭")
            except:
                pass
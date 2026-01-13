"""
ft_sensor.py

OnRobot Gripper Force/Torque Sensor interface via Modbus TCP.
兼容 pymodbus 3.x
"""
import time
import numpy as np

try:
    from pymodbus.client import ModbusTcpClient as ModbusClient
except ImportError:
    from pymodbus.client.sync import ModbusTcpClient as ModbusClient


class FTSensor:
    """OnRobot夹爪力/力矩传感器"""

    def __init__(self, ip="192.168.1.1", unit=65, port=502):
        """
        初始化力传感器

        Args:
            ip: 夹爪IP地址
            unit: Modbus单元ID
            port: Modbus端口
        """
        self.ip = ip
        self.unit = unit
        self.port = port
        self.modbus = None
        self.connected = False
        self.ft_data = np.zeros(6)  # [fx, fy, fz, tx, ty, tz]

        self._connect()

    def _connect(self):
        """连接到夹爪"""
        try:
            self.modbus = ModbusClient(self.ip, port=self.port)
            result = self.modbus.connect()
            self.connected = result
            if self.connected:
                print(f"[FTSensor] Connected to {self.ip}:{self.port}")
            else:
                print(f"[FTSensor] Failed to connect to {self.ip}:{self.port}")
        except Exception as e:
            print(f"[FTSensor] Connection error: {e}")
            self.connected = False

    def read(self):
        """
        读取力/力载数据 - 取左右夹爪传感器的平均值

        Returns:
            np.ndarray: [fx, fy, fz, tx, ty, tz] 单位: N, Nm
        """
        if not self.connected or not self.modbus:
            return np.zeros(6)

        try:
            # 左夹爪传感器: 寄存器 259-264
            # 右夹爪传感器: 寄存器 268-273
            left_ft = np.zeros(6)
            right_ft = np.zeros(6)

            # 读取左夹爪传感器
            for i in range(3):
                force_val = self._read_register(259 + i)
                if force_val is not None:
                    left_ft[i] = force_val / 10.0  # N
                torque_val = self._read_register(259 + 3 + i)
                if torque_val is not None:
                    left_ft[i + 3] = torque_val / 100.0  # Nm

            # 读取右夹爪传感器
            for i in range(3):
                force_val = self._read_register(268 + i)
                if force_val is not None:
                    right_ft[i] = force_val / 10.0  # N
                torque_val = self._read_register(268 + 3 + i)
                if torque_val is not None:
                    right_ft[i + 3] = torque_val / 100.0  # Nm

            # 取平均值
            self.ft_data = (left_ft + right_ft) / 2.0
            return self.ft_data.copy()

        except Exception as e:
            print(f"[FTSensor] Read error: {e}")
            return np.zeros(6)

    def _read_register(self, address):
        """读取单个寄存器并解析16位有符号整数"""
        try:
            # pymodbus 3.x: 只有 address 是位置参数，count 和 device_id 是关键字参数
            result = self.modbus.read_holding_registers(address, count=1, device_id=self.unit)
            if not result.isError():
                # 寄存器值是16位有符号整数，大端序
                value = result.registers[0]
                # 转换为有符号整数
                if value > 32767:
                    value = value - 65536
                return value
        except Exception:
            pass
        return None

    def disconnect(self):
        """断开连接"""
        if self.modbus:
            self.modbus.close()
            self.connected = False
            print("[FTSensor] Disconnected")


def diagnose_registers(sensor, start_addr=250, count=30):
    """诊断寄存器，显示所有寄存器的原始值"""
    print(f"\n{'='*60}")
    print(f"寄存器诊断 (地址 {start_addr}-{start_addr+count-1})")
    print(f"{'='*60}")
    print(f"{'Addr':<6} {'Raw':<10} {'Signed':<10} {'Interpretation'}")
    print(f"{'-'*60}")

    for i in range(count):
        addr = start_addr + i
        try:
            result = sensor.modbus.read_holding_registers(addr, count=1, device_id=sensor.unit)
            if not result.isError():
                raw = result.registers[0]
                # 转换为有符号整数
                signed = raw if raw <= 32767 else raw - 65536

                # 推测可能的含义
                interp = ""
                if 259 <= addr <= 264:
                    interp = " <- 左力传感器"
                elif 268 <= addr <= 273:
                    interp = " <- 右力传感器"

                print(f"{addr:<6} {raw:<10} {signed:<10} {interp}")
        except Exception as e:
            print(f"{addr:<6} Error: {e}")

    print(f"{'='*60}")

    # 显示左右传感器和平均值
    print("\n力传感器数据对比:")
    print(f"{'轴':<6} {'左传感器':<12} {'右传感器':<12} {'平均值':<12}")
    print(f"{'-'*42}")
    ft = sensor.read()
    labels = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
    for i in range(6):
        left_val = sensor._read_register(259 + i) if i < 3 else sensor._read_register(259 + 3 + i)
        right_val = sensor._read_register(268 + i) if i < 3 else sensor._read_register(268 + 3 + i)

        left_converted = left_val / 10.0 if i < 3 and left_val is not None else (left_val / 100.0 if i >= 3 and left_val is not None else 0.0)
        right_converted = right_val / 10.0 if i < 3 and right_val is not None else (right_val / 100.0 if i >= 3 and right_val is not None else 0.0)

        unit = 'N' if i < 3 else 'Nm'
        print(f"{labels[i]:<6} {left_converted:10.2f} {right_converted:10.2f} {ft[i]:10.2f} {unit}")
    print()


if __name__ == "__main__":
    # 测试代码
    sensor = FTSensor()

    if sensor.connected:
        # 首先诊断寄存器
        diagnose_registers(sensor)

        print("读取力传感器数据（按Ctrl+C停止）...")
        try:
            i = 0
            while True:
                ft = sensor.read()
                force_mag = np.linalg.norm(ft[:3])
                torque_mag = np.linalg.norm(ft[3:])

                print(f"[{i}] Force: [{ft[0]:7.2f}, {ft[1]:7.2f}, {ft[2]:7.2f}] N (mag={force_mag:.2f}), "
                      f"Torque: [{ft[3]:7.3f}, {ft[4]:7.3f}, {ft[5]:7.3f}] Nm (mag={torque_mag:.3f})")
                i += 1
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n停止读取")
        sensor.disconnect()
    else:
        print("传感器未连接")
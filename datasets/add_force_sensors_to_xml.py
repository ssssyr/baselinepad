#!/usr/bin/env python3
"""
批量为 MetaWorld v2 任务 XML 文件添加力传感器

这个脚本会：
1. 扫描 metaworld/envs/assets_v2/sawyer_xyz/ 下所有 .xml 文件
2. 检查是否已有 <sensor> 标签
3. 如果没有，在 <actuator> 后添加 <sensor> 标签
"""

import os
import re
from pathlib import Path

# 要添加的传感器 XML
SENSOR_XML = '''    <sensor>
        <force name="ee_force" site="endEffector"/>
        <torque name="ee_torque" site="endEffector"/>
    </sensor>
'''

# XML 目录
XML_DIR = Path("/home/syr/code/prediction_with_action/metaworld/metaworld/envs/assets_v2/sawyer_xyz")


def add_sensor_to_xml(xml_path: Path) -> bool:
    """
    为 XML 文件添加传感器

    Returns:
        bool: 是否进行了修改
    """
    with open(xml_path, 'r') as f:
        content = f.read()

    # 检查是否已有 sensor 标签
    if '<sensor>' in content or '<force' in content:
        return False

    # 在 <actuator> 后添加 <sensor>
    # 查找 </actuator> 的位置
    actuator_end = content.find('</actuator>')
    if actuator_end == -1:
        print(f"  ⚠️  {xml_path.name}: 没有 </actuator> 标签")
        return False

    # 在 </actuator> 后插入 <sensor>
    insert_pos = actuator_end + len('</actuator>')
    new_content = content[:insert_pos] + '\n' + SENSOR_XML + content[insert_pos:]

    # 写回文件
    with open(xml_path, 'w') as f:
        f.write(new_content)

    return True


def main():
    print("=== 为 MetaWorld v2 XML 文件添加力传感器 ===")
    print(f"目录: {XML_DIR}\n")

    # 获取所有 XML 文件
    xml_files = sorted(XML_DIR.glob("*.xml"))

    print(f"找到 {len(xml_files)} 个 XML 文件\n")

    modified_count = 0
    skipped_count = 0

    for xml_path in xml_files:
        modified = add_sensor_to_xml(xml_path)
        if modified:
            print(f"✅ {xml_path.name}: 已添加传感器")
            modified_count += 1
        else:
            print(f"⊘  {xml_path.name}: 已有传感器或无 actuator 标签，跳过")
            skipped_count += 1

    print(f"\n=== 完成 ===")
    print(f"修改: {modified_count} 个文件")
    print(f"跳过: {skipped_count} 个文件")


if __name__ == "__main__":
    main()

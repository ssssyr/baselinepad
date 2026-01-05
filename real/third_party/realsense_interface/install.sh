#!/bin/bash

# RealSense Interface 安装脚本

set -e

echo "=== RealSense Interface 安装脚本 ==="

# 检查Python版本
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "检测到Python版本: $python_version"

if [[ $(echo "$python_version < 3.7" | bc -l) -eq 1 ]]; then
    echo "❌ 需要Python 3.7或更高版本"
    exit 1
fi

# 检查是否在虚拟环境中
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  建议在虚拟环境中安装"
    read -p "是否继续? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 安装系统依赖 (Ubuntu/Debian)
if command -v apt-get &> /dev/null; then
    echo "检测到apt包管理器，安装系统依赖..."
    
    # 添加Intel RealSense仓库
    if ! grep -q "librealsense.intel.com" /etc/apt/sources.list.d/* 2>/dev/null; then
        echo "添加Intel RealSense仓库..."
        sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || \
        sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
        
        sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u
    fi
    
    # 安装RealSense SDK
    sudo apt-get update
    sudo apt-get install -y librealsense2-dkms librealsense2-utils librealsense2-dev
    
    # 安装其他系统依赖
    sudo apt-get install -y python3-dev python3-pip
    
    echo "✅ 系统依赖安装完成"
else
    echo "⚠️  未检测到apt包管理器，请手动安装Intel RealSense SDK"
    echo "参考: https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md"
fi

# 安装Python依赖
echo "安装Python依赖..."
pip3 install -r requirements.txt

# 安装可选依赖
read -p "是否安装开发依赖? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip3 install pytest pytest-cov black flake8
    echo "✅ 开发依赖安装完成"
fi

# 安装包
echo "安装RealSense Interface..."
pip3 install -e .

# 验证安装
echo "验证安装..."
if python3 -c "import realsense_interface; print('✅ 导入成功')" 2>/dev/null; then
    echo "🎉 安装成功!"
    
    # 运行相机检测
    echo "检测RealSense相机..."
    python3 -m realsense_interface.examples.test_camera --list-only
    
else
    echo "❌ 安装失败"
    exit 1
fi

echo ""
echo "=== 安装完成 ==="
echo "使用示例:"
echo "  python3 -m realsense_interface.examples.single_camera_example"
echo "  python3 -m realsense_interface.examples.multi_camera_example"
echo "  realsense-test --help"
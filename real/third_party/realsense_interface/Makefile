# RealSense Interface Makefile

.PHONY: help install test clean lint format check-cameras

help:
	@echo "RealSense Interface 项目管理"
	@echo ""
	@echo "可用命令:"
	@echo "  install        - 安装项目和依赖"
	@echo "  test          - 运行测试"
	@echo "  lint          - 代码检查"
	@echo "  format        - 代码格式化"
	@echo "  check-cameras - 检测RealSense相机"
	@echo "  clean         - 清理临时文件"
	@echo "  example-single - 运行单相机示例"
	@echo "  example-multi  - 运行多相机示例"

install:
	@echo "安装RealSense Interface..."
	pip install -r requirements.txt
	pip install -e .
	@echo "✅ 安装完成"

test:
	@echo "运行测试..."
	python -m pytest tests/ -v
	@echo "✅ 测试完成"

lint:
	@echo "代码检查..."
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format:
	@echo "代码格式化..."
	black . --line-length 100
	@echo "✅ 格式化完成"

check-cameras:
	@echo "检测RealSense相机..."
	python -m realsense_interface.examples.test_camera --list-only

clean:
	@echo "清理临时文件..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	@echo "✅ 清理完成"

example-single:
	@echo "运行单相机示例..."
	python -m realsense_interface.examples.single_camera_example

example-multi:
	@echo "运行多相机示例..."
	python -m realsense_interface.examples.multi_camera_example

# 开发相关命令
dev-install:
	pip install -r requirements.txt
	pip install -e ".[dev]"

dev-test:
	python -m pytest tests/ -v --cov=realsense_interface --cov-report=html

# 打包相关
build:
	python setup.py sdist bdist_wheel

upload-test:
	twine upload --repository testpypi dist/*

upload:
	twine upload dist/*
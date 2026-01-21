"""
LPIPS (Learned Perceptual Image Patch Similarity) 计算脚本
用于评估生成图像与真实图像之间的感知相似度
"""

import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path


# ========== 配置区域 - 在这里修改路径 ==========

# 方法1: 单对图像比较
IMAGE_1_PATH = "/path/to/image1.png"  # 真实图像/参考图像
IMAGE_2_PATH = "/path/to/image2.png"  # 生成图像/预测图像

# 方法2: 批量比较（两个文件夹）- 从 run_metaworld 生成的配对图片
# 默认路径: ./images/rollout_metaworld/lpips_pairs/{task_name}/actual
#         和 ./images/rollout_metaworld/lpips_pairs/{task_name}/predicted
IMAGE_DIR_1 = "/home/syr/code/prediction_with_action/images/rollout_metaworld/lpips_pairs/basketball-v2/actual"  # 真实图像文件夹
IMAGE_DIR_2 = "/home/syr/code/prediction_with_action/images/rollout_metaworld/lpips_pairs/basketball-v2/predicted"  # 预测图像文件夹

# 设备选择
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================


def load_image(image_path):
    """加载图像并转换为tensor"""
    img = Image.open(image_path).convert('RGB')
    # 调整大小到至少256x256 (LPIPS推荐)
    if min(img.size) < 256:
        img = img.resize((max(img.size), max(img.size)), Image.LANCZOS)
    return img


def calculate_lpips(img1_path, img2_path, model=None):
    """计算两张图像之间的LPIPS距离"""
    import lpips

    if model is None:
        # 初始化LPIPS模型 (使用VGG特征提取器)
        model = lpips.LPIPS(net='vgg').to(DEVICE)

    # 加载图像
    img1 = load_image(img1_path)
    img2 = load_image(img2_path)

    # 转换为tensor并归一化到[-1, 1]
    from torchvision.transforms import ToTensor, Normalize

    transform = torch.nn.Sequential(
        ToTensor(),  # 转换到 [0, 1]
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 转换到 [-1, 1]
    )

    tensor1 = transform(img1).unsqueeze(0).to(DEVICE)  # [1, 3, H, W]
    tensor2 = transform(img2).unsqueeze(0).to(DEVICE)

    # 计算LPIPS
    with torch.no_grad():
        distance = model(tensor1, tensor2)

    return distance.item()


def batch_calculate_lpips(dir1, dir2):
    """批量计算两个文件夹中图像的LPIPS距离"""
    import lpips
    from glob import glob

    model = lpips.LPIPS(net='vgg').to(DEVICE)

    # 获取图像列表
    images1 = sorted(glob(os.path.join(dir1, "*.png")) +
                     glob(os.path.join(dir1, "*.jpg")))
    images2 = sorted(glob(os.path.join(dir2, "*.png")) +
                     glob(os.path.join(dir2, "*.jpg")))

    if len(images1) != len(images2):
        print(f"警告: 文件夹1有{len(images1)}张图像, 文件夹2有{len(images2)}张图像")

    results = []
    for i, (img1_path, img2_path) in enumerate(zip(images1, images2)):
        try:
            distance = calculate_lpips(img1_path, img2_path, model)
            results.append({
                'pair': i,
                'image1': os.path.basename(img1_path),
                'image2': os.path.basename(img2_path),
                'lpips': distance
            })
            print(f"[{i+1}/{len(images1)}] {os.path.basename(img1_path)} vs {os.path.basename(img2_path)}: {distance:.4f}")
        except Exception as e:
            print(f"错误处理 {img1_path} vs {img2_path}: {e}")

    return results


def print_statistics(results):
    """打印统计信息"""
    if not results:
        print("没有结果!")
        return

    distances = [r['lpips'] for r in results]

    print("\n" + "="*50)
    print("LPIPS 统计结果")
    print("="*50)
    print(f"样本数量: {len(distances)}")
    print(f"平均值:   {np.mean(distances):.4f}")
    print(f"中位数:   {np.median(distances):.4f}")
    print(f"标准差:   {np.std(distances):.4f}")
    print(f"最小值:   {np.min(distances):.4f}")
    print(f"最大值:   {np.max(distances):.4f}")
    print("="*50)

    # LPIPS解释:
    # 0.0   - 完全相同
    # 0.1-0.2 - 非常相似
    # 0.2-0.4 - 相似但有差异
    # 0.4-0.6 - 有明显差异
    # >0.6   - 差异很大


# ========== 主程序 ==========

if __name__ == "__main__":

    print("LPIPS 计算脚本")
    print("="*50)

    # 尝试安装lpips（如果没有安装）
    try:
        import lpips
    except ImportError:
        print("正在安装 lpips...")
        os.system("pip install lpips")
        import lpips

    # 选择模式
    print("\n请选择模式:")
    print("1. 单对图像比较")
    print("2. 批量文件夹比较")

    mode = input("请输入模式 (1 或 2): ").strip()

    if mode == "1":
        # ========== 单对图像比较 ==========
        print("\n=== 单对图像比较模式 ===")
        print(f"当前配置:")
        print(f"  图像1: {IMAGE_1_PATH}")
        print(f"  图像2: {IMAGE_2_PATH}")
        print()

        use_current = input("使用当前配置? (y/n): ").strip().lower()

        if use_current != 'y':
            IMAGE_1_PATH = input("请输入图像1路径: ").strip()
            IMAGE_2_PATH = input("请输入图像2路径: ").strip()

        # 检查文件是否存在
        if not os.path.exists(IMAGE_1_PATH):
            print(f"错误: 图像1不存在: {IMAGE_1_PATH}")
            exit(1)
        if not os.path.exists(IMAGE_2_PATH):
            print(f"错误: 图像2不存在: {IMAGE_2_PATH}")
            exit(1)

        # 计算LPIPS
        distance = calculate_lpips(IMAGE_1_PATH, IMAGE_2_PATH)

        print("\n" + "="*50)
        print(f"LPIPS 距离: {distance:.4f}")
        print("="*50)

        if distance < 0.1:
            print("解释: 图像非常相似")
        elif distance < 0.3:
            print("解释: 图像相似但有差异")
        elif distance < 0.5:
            print("解释: 图像有明显差异")
        else:
            print("解释: 图像差异很大")

    elif mode == "2":
        # ========== 批量文件夹比较 ==========
        print("\n=== 批量文件夹比较模式 ===")
        print(f"当前配置:")
        print(f"  文件夹1: {IMAGE_DIR_1}")
        print(f"  文件夹2: {IMAGE_DIR_2}")
        print()

        use_current = input("使用当前配置? (y/n): ").strip().lower()

        if use_current != 'y':
            IMAGE_DIR_1 = input("请输入文件夹1路径: ").strip()
            IMAGE_DIR_2 = input("请输入文件夹2路径: ").strip()

        # 检查文件夹是否存在
        if not os.path.exists(IMAGE_DIR_1):
            print(f"错误: 文件夹1不存在: {IMAGE_DIR_1}")
            exit(1)
        if not os.path.exists(IMAGE_DIR_2):
            print(f"错误: 文件夹2不存在: {IMAGE_DIR_2}")
            exit(1)

        # 批量计算
        results = batch_calculate_lpips(IMAGE_DIR_1, IMAGE_DIR_2)

        # 打印统计
        print_statistics(results)

        # 保存结果
        save_results = input("\n是否保存结果? (y/n): ").strip().lower()
        if save_results == 'y':
            import csv
            output_path = os.path.join(os.path.dirname(IMAGE_DIR_1), "lpips_results.csv")
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['pair', 'image1', 'image2', 'lpips'])
                writer.writeheader()
                writer.writerows(results)
            print(f"结果已保存到: {output_path}")

    else:
        print("无效的模式选择")

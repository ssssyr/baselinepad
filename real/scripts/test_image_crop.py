#!/usr/bin/env python3
"""
test_image_crop.py - 测试图像裁剪效果
展示从 1280x720 到 256x256 的处理过程
"""
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # 使用非交互后端
import matplotlib.pyplot as plt


def center_crop_arr(pil_image, image_size):
    """
    中心裁剪 PIL 图像到指定大小。

    对于 1280x720 的图像处理流程:
    1. 逐步缩小直到最小边 < 2*image_size
    2. 缩放使最小边 = image_size
    3. 中心裁剪到 image_size x image_size
    """
    print(f"  原始尺寸: {pil_image.size}")

    steps = [(pil_image.size, "原始")]

    while min(pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
        steps.append((pil_image.size, f"缩小到一半"))
        print(f"  缩小后: {pil_image.size}")

    scale = image_size / min(pil_image.size)
    print(f"  缩放比例: {scale:.4f}")
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    steps.append((pil_image.size, f"缩放到最小边={image_size}"))
    print(f"  缩放后: {pil_image.size}")

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    print(f"  裁剪位置: y={crop_y}, x={crop_x}")

    cropped = arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]
    steps.append(((image_size, image_size), f"中心裁剪到 {image_size}x{image_size}"))

    return Image.fromarray(cropped), steps


def main():
    # 读取第一张图片
    import numpy as np

    data_path = "/home/syr/code/baselinepad/real/scripts/robot_data/episode_0000.npz"

    print("加载数据...")
    data = np.load(data_path)

    images = data["image"]
    print(f"数据形状: {images.shape}")
    print(f"图像数量: {len(images)}")

    # 取第一帧
    first_img = images[0]
    print(f"第一帧形状: {first_img.shape}, dtype: {first_img.dtype}")

    # 转换为 PIL Image
    if first_img.max() <= 1.0:
        first_img = (first_img * 255).astype(np.uint8)

    original_pil = Image.fromarray(first_img)
    print(f"\n原始 PIL Image 尺寸: {original_pil.size}")
    print(f"原始 PIL Image 模式: {original_pil.mode}")

    # 处理图像
    print("\n开始处理...")
    cropped_pil, steps = center_crop_arr(original_pil, 256)
    print(f"\n最终尺寸: {cropped_pil.size}")

    # 创建可视化对比图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 原始图像
    axes[0].imshow(original_pil)
    axes[0].set_title(f"原始图像\n{original_pil.size[0]}x{original_pil.size[1]}", fontsize=14)
    axes[0].axis('off')

    # 缩放后的图像（处理中间步骤）
    temp_img = original_pil.copy()
    while min(temp_img.size) >= 2 * 256:
        temp_img = temp_img.resize(tuple(x // 2 for x in temp_img.size), resample=Image.BOX)
    scale = 256 / min(temp_img.size)
    temp_img = temp_img.resize(tuple(round(x * scale) for x in temp_img.size), resample=Image.BICUBIC)
    axes[1].imshow(temp_img)
    axes[1].set_title(f"缩放后（未裁剪）\n{temp_img.size[0]}x{temp_img.size[1]}", fontsize=14)
    axes[1].axis('off')

    # 最终裁剪结果
    axes[2].imshow(cropped_pil)
    axes[2].set_title(f"最终结果\n256x256", fontsize=14)
    axes[2].axis('off')

    plt.tight_layout()

    # 保存结果
    output_path = "/home/syr/code/baselinepad/real/scripts/crop_comparison.png"
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"\n对比图已保存到: {output_path}")

    # 也保存单独的处理结果
    result_path = "/home/syr/code/baselinepad/real/scripts/cropped_sample.png"
    cropped_pil.save(result_path)
    print(f"处理结果已保存到: {result_path}")

    # 打印处理步骤
    print("\n处理步骤:")
    for i, (size, desc) in enumerate(steps):
        print(f"  {i+1}. {desc}: {size}")


if __name__ == "__main__":
    main()

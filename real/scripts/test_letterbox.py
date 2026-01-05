#!/usr/bin/env python3
"""
test_letterbox.py - 测试 Letterbox 图像处理效果
"""
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def center_crop_arr(pil_image, image_size):
    """原方法：中心裁剪"""
    while min(pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def letterbox_resize(pil_image, target_size):
    """新方法：Letterbox 保持比例"""
    original_w, original_h = pil_image.size

    # 计算缩放比例（使长边等于目标尺寸）
    scale = target_size / max(original_w, original_h)

    # 等比例缩放
    new_w = int(round(original_w * scale))
    new_h = int(round(original_h * scale))
    resized = pil_image.resize((new_w, new_h), resample=Image.BICUBIC)

    # 创建黑色背景
    result = Image.new("RGB", (target_size, target_size), (0, 0, 0))

    # 计算粘贴位置（居中）
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2

    # 粘贴缩放后的图像
    result.paste(resized, (paste_x, paste_y))

    return result


def main():
    # 读取第一张图片
    data_path = "/home/syr/code/baselinepad/real/scripts/robot_data/episode_0000.npz"

    print("加载数据...")
    data = np.load(data_path)
    images = data["image"]

    # 取第一帧
    first_img = images[0]
    if first_img.max() <= 1.0:
        first_img = (first_img * 255).astype(np.uint8)

    original_pil = Image.fromarray(first_img)
    print(f"原始图像: {original_pil.size[0]}x{original_pil.size[1]}")

    # 原方法（裁剪）
    print("\n原方法（裁剪）...")
    cropped_pil = center_crop_arr(original_pil.copy(), 256)
    print(f"裁剪结果: {cropped_pil.size}")

    # 新方法（Letterbox）
    print("\n新方法（Letterbox）...")
    letterbox_pil = letterbox_resize(original_pil, 256)
    print(f"Letterbox 结果: {letterbox_pil.size}")

    # 创建对比图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 原始图像
    axes[0].imshow(original_pil)
    axes[0].set_title(f"原始图像\n{original_pil.size[0]}x{original_pil.size[1]}", fontsize=14)
    axes[0].axis('off')

    # 原方法（裁剪）
    axes[1].imshow(cropped_pil)
    axes[1].set_title(f"原方法（裁剪）\n256x256\n丢失边缘信息", fontsize=14, color='red')
    axes[1].axis('off')

    # 新方法（Letterbox）
    axes[2].imshow(letterbox_pil)
    axes[2].set_title(f"新方法（Letterbox）\n256x256\n保留所有信息", fontsize=14, color='green')
    axes[2].axis('off')

    plt.tight_layout()

    # 保存结果
    output_path = "/home/syr/code/baselinepad/real/scripts/letterbox_comparison.png"
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"\n对比图已保存到: {output_path}")

    # 保存单独的 Letterbox 结果
    result_path = "/home/syr/code/baselinepad/real/scripts/letterbox_sample.png"
    letterbox_pil.save(result_path)
    print(f"Letterbox 结果已保存到: {result_path}")


if __name__ == "__main__":
    main()

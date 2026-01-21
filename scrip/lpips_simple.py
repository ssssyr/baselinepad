"""
简化版 LPIPS 计算脚本 - 使用项目中已有的 torch
不需要安装额外的 lpips 包
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from glob import glob


# ========== 配置区域 ==========
IMAGE_DIR_1 = "/home/syr/code/prediction_with_action/images/rollout_metaworld/lpips_pairs/basketball-v2/actual"
IMAGE_DIR_2 = "/home/syr/code/prediction_with_action/images/rollout_metaworld/lpips_pairs/basketball-v2/predicted"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ===================================


class LPIPSModel(nn.Module):
    """简化的LPIPS模型 - 使用VGG特征"""
    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg.features[:16]  # 使用到ReLU3_3层
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        # 冻结参数
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in [3, 8, 15]:  # ReLU1_2, ReLU2_2, ReLU3_3
                features.append(x)
        return features


def load_image_tensor(image_path):
    from torchvision.transforms import ToTensor, Resize
    from torchvision.transforms import functional as F

    img = Image.open(image_path).convert('RGB')
    if min(img.size) < 224:
        img = img.resize((max(img.size), max(img.size)), Image.LANCZOS)

    img = F.resize(img, [224, 224])
    img = F.to_tensor(img)
    return img.unsqueeze(0)


def calculate_lpips_distance(img1_path, img2_path, model):
    tensor1 = load_image_tensor(img1_path).to(DEVICE)
    tensor2 = load_image_tensor(img2_path).to(DEVICE)

    with torch.no_grad():
        features1 = model(tensor1)
        features2 = model(tensor2)

    distances = []
    for f1, f2 in zip(features1, features2):
        f1_flat = f1.view(f1.size(0), -1)
        f2_flat = f2.view(f2.size(0), -1)
        distance = torch.norm(f1_flat - f2_flat, dim=1) / torch.norm(f1_flat, dim=1)
        distances.append(distance.item())

    weights = [0.1, 0.3, 0.6]
    return sum(d * w for d, w in zip(distances, weights))


if __name__ == "__main__":
    print("="*50)
    print("简化版 LPIPS 计算脚本")
    print("="*50)
    print(f"\n实际观测: {IMAGE_DIR_1}")
    print(f"模型预测: {IMAGE_DIR_2}")
    print(f"设备: {DEVICE}\n")

    model = LPIPSModel().to(DEVICE)
    model.eval()

    images1 = sorted(glob(os.path.join(IMAGE_DIR_1, "*.png")))
    images2 = sorted(glob(os.path.join(IMAGE_DIR_2, "*.png")))

    print(f"找到 {len(images1)} 对图像\n")

    results = []
    for i, (img1_path, img2_path) in enumerate(zip(images1, images2)):
        distance = calculate_lpips_distance(img1_path, img2_path, model)
        results.append(distance)
        print(f"[{i+1:2d}/{len(images1)}] {os.path.basename(img1_path):40s} -> LPIPS: {distance:.4f}")

    # 统计
    distances = np.array(results)
    print("\n" + "="*50)
    print("统计结果")
    print("="*50)
    print(f"样本数:   {len(distances)}")
    print(f"平均值:   {np.mean(distances):.4f}")
    print(f"中位数:   {np.median(distances):.4f}")
    print(f"标准差:   {np.std(distances):.4f}")
    print(f"最小值:   {np.min(distances):.4f}")
    print(f"最大值:   {np.max(distances):.4f}")
    print("="*50)

    avg = np.mean(distances)
    if avg < 0.1:
        print("\n解释: 图像非常相似，预测质量很好")
    elif avg < 0.3:
        print("解释: 图像相似但有差异，预测质量较好")
    elif avg < 0.5:
        print("解释: 图像有明显差异，预测质量中等")
    else:
        print("解释: 图像差异很大，预测质量需要改进")

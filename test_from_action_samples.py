import torch
import torch.distributed as dist
from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import os
import json
from glob import glob

class SimpleDiffusionAgent():
    def __init__(self, ckpt_path, vae_path, clip_path, denoise_steps=200):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        args = checkpoint["args"]
        self.args = args

        new_attr = ["action_condition", "action_dim", "action_steps", "d_hidden_size", "use_depth", 'ckpt_wrapper']
        for attr in new_attr:
            if not hasattr(self.args, attr):
                setattr(self.args, attr, False)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("load dit")
        self.latent_size = args.image_size // 8
        self.model = DiT_models[args.model](
            input_size=self.latent_size,
            num_classes=args.num_classes,
            args = args
        )
        state_dict = checkpoint["model"]
        model_dict = self.model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        print("load diffusion")
        self.diffusion = create_diffusion(str(denoise_steps))
        
        print("load vae and clip")
        self.vae = AutoencoderKL.from_pretrained(vae_path).to(self.device)
        from transformers import AutoTokenizer, CLIPTextModelWithProjection
        self.clip_model = CLIPTextModelWithProjection.from_pretrained(clip_path).to(self.device)
        self.clip_tokenizer = AutoTokenizer.from_pretrained(clip_path)
    
    def encode_text(self, text):
        with torch.no_grad():
            inputs = self.clip_tokenizer([text], padding=True, return_tensors="pt").to(self.device)
            outputs = self.clip_model(**inputs)
            text_embeds = outputs.text_embeds
        return text_embeds

    def encode_image(self, image):
        image = np.array(image)
        image = cv2.resize(image, (256,256), interpolation=cv2.INTER_AREA)
        image = torch.tensor(image).float().to(self.device).permute(2,0,1).unsqueeze(0)
        image = torch.clamp((image-128.0)/127.5, -1, 1).to(self.device)
        with torch.no_grad():
            latent = self.vae.encode(image).latent_dist.sample().mul_(0.18215)
        return latent

    def action(self, text, rgb=None, depth=None, state=None):
        y = self.encode_text(text)
        x_cond = self.encode_image(rgb)
        depth_cond = None
        
        sample_fn = self.model.forward
        t = self.args.predict_horizon
        latent_size = self.args.image_size // 8
        z = torch.randn(1, self.model.in_channels*t, latent_size, latent_size).to(self.device)
        
        action_cond = None
        z_a = None
        z_d = None

        model_kwargs = {
            "y": y,
            "x_cond": x_cond,
            "noised_action": z_a,
            "depth_cond": depth_cond,
            "noised_depth": z_d,
            "action_cond": action_cond
        }
        
        samples = self.diffusion.p_sample_loop(
            sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=self.device
        )
        return samples, None, None
    
    def decode(self, x, x_pred, prefix="", save=False):
        x = cv2.resize(x, (256,256), interpolation=cv2.INTER_AREA)
        B,C,H,W = x_pred.shape
        t = self.args.predict_horizon
        x_pred = x_pred.view(B*t,int(C/t),H,W)
        rec_pred = self.vae.decode(x_pred / 0.18215).sample
        rec_pred = torch.clamp(127.5 * rec_pred + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        img = np.concatenate([x, rec_pred[0], rec_pred[1], rec_pred[2]], axis=1)
        if not save:
            return img

if __name__ == "__main__":
    # 配置
    ckpt_path = "/mnt/sda/syr/checkpoint/checkpoint1114/0010000.pt"
    vae_path = "/home/syr/code/models/sd-vae-ft-mse/"
    clip_path = "/home/syr/code/models/clip-vit-base-patch32/"
    sample_dir = "gallery/action_samples"
    output_dir = "output/action_samples_vision_prediction"

    # 构建代理
    agent = SimpleDiffusionAgent(ckpt_path=ckpt_path, vae_path=vae_path, clip_path=clip_path)

    # 查找所有样本
    json_files = glob(os.path.join(sample_dir, "*_actions.json"))

    # 测试所有样本
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        sample_id = data['sample_id']
        instruction = data['instruction']
        image_path = os.path.join(sample_dir, f"{sample_id}.jpg")

        if not os.path.exists(image_path):
            print(f"\n⚠️  找不到图像 {image_path}，跳过样本 {sample_id}")
            continue

        print(f"\n正在测试 {sample_id}: {instruction}")
        
        # 加载图像
        rgb = Image.open(image_path).convert("RGB")
        rgb = np.array(rgb)
        rgb = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
        
        # 执行图像预测
        samples, _, _ = agent.action(instruction, rgb)
        img = agent.decode(rgb, samples, prefix=sample_id, save=False)

        # 保存结果
        uuid = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{sample_id}_{uuid}.jpg")
        Image.fromarray(img).save(output_path)
        print(f"✅ {sample_id} 评估完成！结果保存到: {output_path}")
        time.sleep(1) # 避免文件名冲突

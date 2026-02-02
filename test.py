import argparse
import os
import torch
from PIL import Image
from torchvision import transforms
import torchvision.utils as utils
from utils.utils import img_resize, load_segment
import numpy as np
from torch.cuda.amp import autocast
from models.RevResNet import RevResNet

from models.WaveletAdaIN import WaveletAdaIN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--ckpoint', type=str, default='./checkpoint.pt')
# data
parser.add_argument('--content_dir', type=str, default='./data/content')
parser.add_argument('--style_dir', type=str, default='./data/style')

parser.add_argument('--out_dir', type=str, default="./output")
parser.add_argument('--max_size', type=int, default=512)
parser.add_argument('--alpha_c', type=float, default=None)


args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)
out_dir = args.out_dir

# Reversible Network
RevNetwork = RevResNet(nBlocks=[10, 10, 10], nStrides=[1, 2, 2], nChannels=[16, 64, 256], in_channel=3, mult=4, hidden_dim=64, sp_steps=1)


state_dict = torch.load(args.ckpoint)
RevNetwork.load_state_dict(state_dict['state_dict'])
RevNetwork = RevNetwork.to(device)
RevNetwork.eval()

#  Transfer module
fusion =WaveletAdaIN()
fusion.to(device)



# Get all images from content and style folders
content_images = [os.path.join(args.content_dir, img) for img in os.listdir(args.content_dir) if img.endswith(('.jpg', '.png'))]
style_images = [os.path.join(args.style_dir, img) for img in os.listdir(args.style_dir) if img.endswith(('.jpg', '.png'))]

# Stylization for all content-style pairs
for content_path in content_images:
    for style_path in style_images:
        content = Image.open(content_path).convert('RGB')
        style = Image.open(style_path).convert('RGB')

        ori_csize = content.size
        content = img_resize(content, args.max_size, down_scale=RevNetwork.down_scale)
        style = img_resize(style, args.max_size, down_scale=RevNetwork.down_scale)

        content = transforms.ToTensor()(content).unsqueeze(0).to(device)
        style = transforms.ToTensor()(style).unsqueeze(0).to(device)

        # Stylization
        with torch.no_grad():
            z_c = RevNetwork(content, forward=True)
            z_s = RevNetwork(style, forward=True)

            z_cs = fusion(z_c, z_s)

            stylized = RevNetwork(z_cs, forward=False)

        # Save stylized image
        cn = os.path.basename(content_path)
        sn = os.path.basename(style_path)
        file_name = f"{cn.split('.')[0]}_{sn.split('.')[0]}.png"
        path = os.path.join(out_dir, file_name)

        grid = utils.make_grid(stylized.data, nrow=1, padding=0)
        ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        out_img = Image.fromarray(ndarr)
        out_img.save(path, quality=100)
        print(f"Saved at {path}")

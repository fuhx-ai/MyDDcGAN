#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/4 20:02
# @Author  : Fuhx
# @File    : val.py
# @Brief   :
# @Version : 0.1


import torch
from PIL import Image
from kornia.color import ycbcr_to_rgb
from torch.utils.data import DataLoader
from torchvision import transforms
from kornia import image_to_tensor

from core.dataset import FusionDataset
from core.model import FuseModel
from core.utils import gray_read, img_write, ycbcr_read
from core.utils.config import load_config
import cv2


if __name__ == '__main__':
    ir = gray_read('demo/03994/ir.png')
    trans = transforms.Compose([
        transforms.Resize([768, 1024]),
    ])
    ir = trans(ir)
    vi, cbcr = ycbcr_read('demo/03994/vi.png')
    vi = trans(vi)
    cbcr = trans(cbcr)
    data = {'Vis': vi.unsqueeze(0), 'Inf': ir.unsqueeze(0)}
    config = load_config('config/GAN_G1_D2_color_spl_disc.yaml')
    ganModel = FuseModel(config, val=True)
    ganModel.Generator.load_state_dict(torch.load('weights/GAN_G1_D2_ir/generator/generator_99.pth'))
    ganModel.eval()
    Generator_feats, _, _ = ganModel(data)
    fuse = Generator_feats['Generator_1'][0]

    re = torch.cat([fuse, cbcr], dim=0)
    re = ycbcr_to_rgb(re)
    img_write(re, 'demo/03994/res.png')

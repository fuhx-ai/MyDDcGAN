#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/25 19:59
# @Author  : Fuhx
# @File    : fusionDataset.py
# @Brief   :
# @Version : 0.1

import os
from copy import deepcopy
from typing import Literal

import torch
from PIL import Image
from core.utils import gray_read, ycbcr_read, load_config
from collections import Counter
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


class FusionDataset(Dataset):
    """docstring for Fusion_Datasets"""

    def __init__(self, mode: Literal['train', 'val', 'pred'], config: dict):
        super().__init__()
        self.root_dir = config['root_dir']
        self.sensors = config['sensors']
        self.color = config['color']
        self.mode = mode
        self.config = config
        self.img_list = {i: os.listdir(os.path.join(self.root_dir, i)) for i in self.sensors}
        self.img_path = {i: [os.path.join(self.root_dir, i, j) for j in os.listdir(os.path.join(self.root_dir, i))]
                         for i in self.sensors}
        self.input_size = config['input_size']

    def __getitem__(self, index):
        trans = transforms.Compose([
                transforms.Resize([self.input_size, self.input_size]),
            ]) if self.mode == 'train' else transforms.Compose([
                transforms.Resize([self.input_size, self.input_size]),
            ])
        for i in self.sensors:
            if i == 'Inf':
                ir = gray_read(self.img_path[i][index])
                ir = trans(ir)
            else:
                if self.color:
                    vi, cbcr = ycbcr_read(self.img_path[i][index])
                    vi = trans(vi)
                    cbcr = trans(cbcr)
                else:
                    vi = gray_read(self.img_path[i][index])
                    vi = trans(vi)
        if self.color:
            sample = {
                'Inf': ir,
                'Vis': vi,
                'CBCR': cbcr
            }
        else:
            sample = {
                'Inf': ir,
                'Vis': vi,
            }
        sample_1 = deepcopy(sample)
        return sample_1

    def __len__(self):
        img_num = [len(self.img_list[i]) for i in self.img_list]
        img_counter = Counter(img_num)
        assert len(img_counter) == 1, 'Sensors Has Different len'
        return img_num[0]

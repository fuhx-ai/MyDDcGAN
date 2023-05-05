#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/4 15:38
# @Author  : Fuhx
# @File    : img_utils.py
# @Brief   :
# @Version : 0.1
from pathlib import Path
from typing import Tuple
from kornia import image_to_tensor, tensor_to_image
from kornia.color import rgb_to_ycbcr, bgr_to_rgb, rgb_to_bgr
import cv2
from torch import Tensor
import torch


def gray_read(img_path: str | Path) -> Tensor:
    img_n = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    img_t = image_to_tensor(img_n).float() / 255
    return img_t


def ycbcr_read(img_path: str | Path) -> Tuple[Tensor, Tensor]:
    img_n = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    img_t = image_to_tensor(img_n).float() / 255
    img_t = rgb_to_ycbcr(bgr_to_rgb(img_t))
    y, cbcr = torch.split(img_t, [1, 2], dim=0)
    return y, cbcr


def img_write(img_t: Tensor, img_path: str | Path):
    if img_t.shape[0] == 3:
        img_t = rgb_to_bgr(img_t)
    img_n = tensor_to_image(img_t.squeeze().cpu()) * 255
    cv2.imwrite(str(img_path), img_n)

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from core.model import FuseModel
from core.utils.config import load_config
import math
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # 设置warm up的轮次为10次
    # warm_up_iter = 10
    # T_max = 100  # 周期
    # lr_max = 0.1  # 最大值
    # lr_min = 1e-5  # 最小值
    #
    # lambda0 = lambda cur_iter: cur_iter / warm_up_iter if cur_iter < warm_up_iter else \
    #     (lr_min + 0.5 * (lr_max - lr_min) * (
    #                 1.0 + math.cos((cur_iter - warm_up_iter) / (T_max - warm_up_iter) * math.pi))) / 0.1
    # lr = []
    # for i in range(100):
    #     y = lambda0(i)
    #     lr.append(y)
    # plt.plot(np.arange(100), np.array(lr))
    # plt.show()
    a = torch.tensor([1, 1, 0, 0.])
    b = torch.tensor([0.8, 0.8, 0.2, 0.2])
    c = torch.tensor([0.001, 0.001, 0., 0.])
    l = torch.nn.BCELoss()(c, a)
    print(l)


#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/2 10:30
# @Author  : Fuhx
# @File    : loss.py
# @Brief   :
# @Version : 0.1


import torch
import torch.nn as nn


class L_con(nn.Module):
    """L_con, 包括F范数和TV范数"""

    def __init__(self, eta=1.2):
        super(L_con, self).__init__()
        self.eta = eta

    def forward(self, G, v, i):
        I = torch.pow(torch.pow((G - i), 2).sum(), 0.5) / (G.shape[2])
        r = G - v
        [W, H] = r.shape[2:4]
        tv1 = torch.pow((r[:, :, 1:, :] - r[:, :, :H - 1, :]), 2).mean()
        tv2 = torch.pow((r[:, :, :, 1:] - r[:, :, :, :W - 1]), 2).mean()
        V = tv1 + tv2
        return (I + self.eta * V) / 2


class L_adv_G(nn.Module):
    """docstring for L_adv_G"""

    def __init__(self):
        super(L_adv_G, self).__init__()

    def forward(self, dv, di):
        idx_m = int(len(dv) / 2)  # dv, di 前一半是对红外or可见光图像的判断， 后一半才是对融合图像的判断
        dvi = torch.concat([dv[idx_m:], di[idx_m:]], dim=0)
        conf_vi = torch.zeros_like(dvi)
        return -nn.BCELoss()(dvi, conf_vi)


class L_G(nn.Module):
    """docstring for L_G"""

    def __init__(self, lam=0.5, eta=1.2):
        super(L_G, self).__init__()
        self.lam = lam
        self.eta = eta

        self.L_con = L_con(self.eta)
        self.L_adv_G = L_adv_G()

    def forward(self, v, i, G, dv, di):
        return self.L_adv_G(dv, di) + self.lam * self.L_con(G, v, i)


class L_Dv(nn.Module):
    """docstring for L_Dv"""

    def __init__(self):
        super(L_Dv, self).__init__()

    def forward(self, dv_G, dv_conf):
        return nn.BCELoss()(dv_G, dv_conf)


class L_Di(nn.Module):
    """docstring for L_Di"""

    def __init__(self):
        super(L_Di, self).__init__()

    def forward(self, di_G, di_conf):
        return nn.BCELoss()(di_G, di_conf)


if __name__ == '__main__':
    vis = torch.rand((4, 3, 256, 256))
    ir = torch.rand((4, 3, 256, 256))
    fuse = torch.rand((4, 3, 256, 256))
    dv = torch.tensor([0.8, 0.8, 0.3, 0.3])
    conf_v = torch.tensor([1., 1., 0., 0.])
    di = torch.tensor([0.8, 0.8, 0.4, 0.4])
    conf_i = torch.tensor([1., 1., 0., 0.])

    l_dv = L_Dv()
    loss_dv = l_dv(dv, conf_v)
    l_di = L_Di()
    loss_di = l_di(di, conf_i)

    print(loss_dv, loss_di)

    l_adv_g = L_adv_G()
    loss_adv_g = l_adv_g(dv, di)
    print('adv_g =', loss_adv_g)

    l_con = L_con()
    loss_con_g = l_con(fuse, vis, ir)
    print('con_g =', loss_con_g)

    l_g = L_G()
    loss_g = l_g(vis, ir, fuse, dv, di)
    print('g =', loss_g)


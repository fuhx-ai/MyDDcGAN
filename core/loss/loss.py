#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/2 10:30
# @Author  : Fuhx
# @File    : loss.py
# @Brief   :
# @Version : 0.1


import torch
import torch.nn as nn


class LossCon(nn.Module):
    """LossCon内容损失, 包括F范数(fuse和ir)和TV范数(fuse和vis)"""

    def __init__(self, eta_vis=10, eta_ir=0.12, eta_tv=1.2):
        super(LossCon, self).__init__()
        self.eta_vis = eta_vis
        self.eta_ir = eta_ir
        self.eta_tv = eta_tv

    def forward(self, fuse, vis, ir):
        # I = torch.pow(torch.pow((G - i), 2).sum(), 0.5) / (G.shape[2])
        loss_ir = nn.MSELoss()(fuse, ir)
        loss_vis = nn.L1Loss()(fuse, vis)
        r = fuse - vis
        [W, H] = r.shape[2:4]
        tv1 = torch.pow((r[:, :, 1:, :] - r[:, :, :H - 1, :]), 2).mean()
        tv2 = torch.pow((r[:, :, :, 1:] - r[:, :, :, :W - 1]), 2).mean()
        loss_tv = tv1 + tv2
        return self.eta_vis * loss_vis + self.eta_ir * loss_ir + self.eta_tv * loss_tv


class LossAdvG(nn.Module):
    """generator对抗损失"""

    def __init__(self):
        super(LossAdvG, self).__init__()

    def forward(self, dv, di):
        idx_m = int(len(dv) / 2)  # dv, di 前一半是对红外or可见光图像的判断， 后一半才是对融合图像的判断
        dvi = torch.concat([dv[idx_m:], di[idx_m:]], dim=0)
        conf_vi = torch.ones_like(dvi)
        return nn.MSELoss()(dvi, conf_vi)


class LossG(nn.Module):
    """generator总损失，对抗损失+内容损失"""

    def __init__(self, lam=0.5, eta_vis=10, eta_ir=0.12, eta_tv=1.2):
        super(LossG, self).__init__()
        self.lam = lam
        self.loss_con = LossCon(eta_vis, eta_ir, eta_tv)
        self.loss_adv_g = LossAdvG()

    def forward(self, v, i, f, dv, di):
        return self.loss_adv_g(dv, di) + self.lam * self.loss_con(f, v, i)


class LossDv(nn.Module):
    """dv判别器loss"""

    def __init__(self):
        super(LossDv, self).__init__()

    def forward(self, dv_g, dv_conf):
        return nn.BCELoss()(dv_g, dv_conf)


class LossDi(nn.Module):
    """Di判别器loss"""

    def __init__(self):
        super(LossDi, self).__init__()

    def forward(self, di_g, di_conf):
        return nn.BCELoss()(di_g, di_conf)


if __name__ == '__main__':
    vis = torch.rand((4, 3, 256, 256))
    ir = torch.rand((4, 3, 256, 256))
    fuse = torch.rand((4, 3, 256, 256))
    dv = torch.tensor([0.8, 0.8, 0.3, 0.3])
    conf_v = torch.tensor([1., 1., 0., 0.])
    di = torch.tensor([0.8, 0.8, 0.4, 0.4])
    conf_i = torch.tensor([1., 1., 0., 0.])

    l_dv = LossDv()
    loss_dv = l_dv(dv, conf_v)
    l_di = LossDi()
    loss_di = l_di(di, conf_i)

    print(loss_dv, loss_di)

    l_adv_g = LossAdvG()
    loss_adv_g = l_adv_g(dv, di)
    print('adv_g =', loss_adv_g)

    l_con = LossCon()
    loss_con_g = l_con(fuse, vis, ir)
    print('con_g =', loss_con_g)

    l_g = LossG()
    loss_g = l_g(vis, ir, fuse, dv, di)
    print('g =', loss_g)

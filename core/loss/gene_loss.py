#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/25 20:06
# @Author  : Fuhx
# @File    : gene_loss.py
# @Brief   :
# @Version : 0.1


from torch import nn
from core.loss.loss import L_G, L_adv_G, L_con


class GeneratorLoss(nn.Module):
    """计算生成器的loss"""

    def __init__(self, generator_cfg):
        super().__init__()
        self.adv_weight = generator_cfg['Loss_adv_weight']
        self.con_weight = generator_cfg['Loss_Dist_weight']
        self.Dist_Loss = generator_cfg['Dist_Loss']
        self.MSELoss = nn.MSELoss()
        self.l_adv_g = L_adv_G()
        self.l_con_g = L_con()

    def forward(self, inputs, generator, disc, conf):
        # fuse = generator['Generator_1']
        # vi, ir = inputs['Vis'], inputs['Inf']
        # dv, di = disc['Discriminator_1'], disc['Discriminator_2']
        # loss_adv = self.l_adv_g(dv, di)
        # loss_con = self.l_con_g(fuse, vi, ir)
        # loss_g = self.adv_weight * loss_adv + self.con_weight * loss_con

        disc_name = [i for i in disc]
        loss_dist = 0
        for i in self.Dist_Loss:
            loss_dist = loss_dist + self.MSELoss(generator[i[0]], inputs[i[1]])
        loss_adv = 0
        for i in disc_name:
            loss_adv = loss_adv + self.MSELoss(disc[i], conf[i])
        loss_g = self.con_weight * loss_dist + self.adv_weight * loss_adv

        return loss_g

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/25 20:06
# @Author  : Fuhx
# @File    : gene_loss.py
# @Brief   :
# @Version : 0.1


from torch import nn


class GeneratorLoss(nn.Module):
    """计算生成器的loss"""

    def __init__(self, generator_cfg):
        super().__init__()
        self.Loss_adv_weight = generator_cfg['Loss_adv_weight']
        self.Loss_Dist_weight = generator_cfg['Loss_Dist_weight']
        self.Dist_Loss = generator_cfg['Dist_Loss']
        self.MSELoss = nn.MSELoss()

    def forward(self, inputs, generator, disc, conf):
        # Generator_name = [i for i in Generator]
        # batch_size = Generator[Generator_name[0]].shape[0]
        disc_name = [i for i in disc]

        loss_dist = 0
        for i in self.Dist_Loss:
            loss_dist = loss_dist + self.MSELoss(generator[i[0]], inputs[i[1]])

        loss_adv = 0
        for i in disc_name:
            loss_adv = loss_adv + self.MSELoss(disc[i], conf[i])

        return self.Loss_Dist_weight * loss_dist + self.Loss_adv_weight * loss_adv

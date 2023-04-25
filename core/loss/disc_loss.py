#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/25 20:18
# @Author  : Fuhx
# @File    : disc_loss.py
# @Brief   :
# @Version : 0.1


from torch import nn


class DiscriminatorLoss(nn.Module):
    def __init__(self, disc_cfg):
        super().__init__()
        self.MSELoss = nn.MSELoss()

    def forward(self, gene, disc, conf):
        # Generator = Generator['Generator']
        # Generator_name = [i for i in Generator]
        disc_name = [i for i in disc]

        loss_adv = 0
        for i in disc_name:
            loss_adv = loss_adv + self.MSELoss(disc[i], conf[i])
        return loss_adv

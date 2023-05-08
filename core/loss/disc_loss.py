#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/25 20:18
# @Author  : Fuhx
# @File    : disc_loss.py
# @Brief   :
# @Version : 0.1


from torch import nn
from core.loss.loss import L_Di, L_Dv


class DiscriminatorLoss(nn.Module):
    def __init__(self, disc_cfg):
        super().__init__()
        self.MSELoss = nn.MSELoss()
        self.l_dv = L_Dv()
        self.l_di = L_Di()

    def forward(self, gene, disc, conf):
        # disc_name = [i for i in disc]
        # loss_adv = 0
        # for i in disc_name:
        #     loss_adv = loss_adv + self.MSELoss(disc[i], conf[i])
        # return loss_adv

        disc_name = [i for i in disc]
        dv, di = disc[disc_name[0]], disc[disc_name[1]]
        conf_v, conf_i = conf[disc_name[0]], conf[disc_name[1]]
        loss_dv = self.l_dv(dv, conf_v)
        loss_di = self.l_di(di, conf_i)
        loss_d = loss_dv + loss_di
        return loss_d

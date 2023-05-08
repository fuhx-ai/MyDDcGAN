#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/25 20:25
# @Author  : Fuhx
# @File    : fuseModel.py
# @Brief   :
# @Version : 0.1


import torch
import torch.nn as nn

from core.model.baseModel import Model


class FuseModel(nn.Module):
    """docstring for build_model"""

    def __init__(self, config, val=False):
        super().__init__()
        self.val = val  # 是否为验证模式
        generator_config = config['Generator']

        self.Generator_name = generator_config['Generator_Name']
        self.Generator_input = {G: input for G, input in
                                zip(self.Generator_name, generator_config['Input_Datasets'])}
        self.Generator = nn.ModuleDict({i: Model({i: config['Struct'][i]}) for i in self.Generator_name})

        discriminator_config = config['Discriminator']
        self.Discriminator_name = discriminator_config['Discriminator_Name']
        self.Discriminator_input = {D: input for D, input in
                                    zip(self.Discriminator_name, discriminator_config['Input_Datasets'])}
        self.Discriminator = nn.ModuleDict({i: Model({i: config['Struct'][i]}) for i in self.Discriminator_name})
        if config['Train']['Base']['continue'] and not val:
            self.Generator.load_state_dict(torch.load(config['Train']['Base']['gene_path']))
            self.Discriminator.load_state_dict(torch.load(config['Train']['Base']['disc_path']))

    def forward(self, inputs):

        generator_feats = {}
        for G in self.Generator:
            generator_inputs = inputs.copy()
            generator_inputs = {i: generator_inputs[i] for i in self.Generator_input[G]}
            generator_feat = self.Generator[G](generator_inputs)  # 各层输出dict
            generator_feat = generator_feat[[i for i in generator_feat][-1]]  # 最后一层输出value
            generator_feats.update({G: generator_feat})
        if self.val:
            return generator_feats, {}, {}
        discriminator_feats = {}
        confidence = {}
        for D in self.Discriminator:
            discriminator_inputs = inputs.copy()  # vis, ir
            # discriminator_inputs.update({'Generator': generator_feats['Generator']})
            discriminator_inputs.update(generator_feats)  # vis, ir, gen
            discriminator_inputs = {i: discriminator_inputs[i] for i in self.Discriminator_input[D]}  # vis, gen or ir, gen

            confidence.update({D: torch.cat(
                [torch.zeros(discriminator_inputs[i].shape[0]).to(discriminator_inputs[i].device)
                 if i in self.Generator else
                 torch.ones(discriminator_inputs[i].shape[0]).to(discriminator_inputs[i].device)
                 for i in discriminator_inputs], dim=0)})

            discriminator_feat = self.Discriminator[D](discriminator_inputs)
            discriminator_feat = discriminator_feat[[i for i in discriminator_feat][-1]]
            discriminator_feats.update({D: discriminator_feat.squeeze()})

        return generator_feats, discriminator_feats, confidence

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/25 20:05
# @Author  : Fuhx
# @File    : __init__.py.py
# @Brief   :
# @Version : 0.1

from .disc_loss import DiscriminatorLoss
from .gene_loss import GeneratorLoss


__all__ = ['DiscriminatorLoss', 'GeneratorLoss']

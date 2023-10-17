#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import math

import torch
import torch.nn as nn

from utils.events import LOGGER
from utils.scheduler import CosineAnnealingWarmUpRestarts
from torch.optim.lr_scheduler import CyclicLR


def build_optimizer(cfg, model):
    """ Build optimizer from cfg file. """
    if cfg.solver.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.solver.lr, weight_decay=cfg.solver.weight_decay)
    elif cfg.solver.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.solver.lr, weight_decay=cfg.solver.weight_decay,
                                    momentum=cfg.solver.momentum)
    else:
        raise NotImplementedError("Not Implemented {}...".format(cfg.solver.optim))
    return optimizer


def build_lr_scheduler(cfg, optimizer):
    """ Build learning rate scheduler from cfg file. """
    if cfg.solver.scheduler == 'cyclelr':
        scheduler = CyclicLR(optimizer, base_lr=cfg.solver.lr, max_lr=cfg.solver.lr_max, step_size_up=cfg.solver.T_up,
                             gamma=cfg.solver.gamma,
                             step_size_down=cfg.solver.T_down, cycle_momentum=False,
                             mode='triangular2')
    elif cfg.solver.scheduler == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.solver.T_down, gamma=cfg.solver.gamma)
    elif cfg.solver.scheduler == 'cosine':
        lf = lambda x: ((1 - math.cos(x * math.pi / cfg.epochs)) / 2) * (cfg.solver.lrf - 1) + 1
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    else:
        raise LOGGER.error('unknown lr scheduler, use Cosine defaulted')
    return scheduler

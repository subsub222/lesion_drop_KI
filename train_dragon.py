#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import os

import torch
import os.path as osp
from pathlib import Path

from core.engine_dagon import Trainer
from utils.config import Config
from utils.events import LOGGER
from cmd_in import get_args_parser
from utils.general import increment_name


def check_and_init(args):
    save_dir = str(increment_name(osp.join(args.outputdir, args.foldername)))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Load configuration file
    cfg = Config.fromfile(args.conffile)
    setattr(cfg, 'save_dir', save_dir)
    for key, value in args._get_kwargs():
        if not hasattr(cfg, str(key)):
            setattr(cfg, str(key), value)
    #
    print_cfg(cfg)
    return cfg._cfg_dict


def print_cfg(config):
    print('=' * 15, 'Configuration', '=' * 15)
    for key in list(config.keys()):
        if not isinstance(config._cfg_dict[key], dict):
            print("{0:>20} :".format(str(key)) + " {0:<20}".format(config._cfg_dict[key]))
        else:
            print("{0:>20} :".format(str(key)) + " {0:<20}".format(str(config._cfg_dict[key])))
    print('=' * 15, 'Configuration', '=' * 15)


if __name__ == '__main__':
    #
    args = get_args_parser().parse_args()
    # Setup
    cfg = check_and_init(args)
    #
    from setproctitle import *
    setproctitle(cfg.procname)
    # device
    device = torch.device('cuda:{}'.format(cfg.gpuid)) if cfg.gpuid is not None else torch.device('cpu')
    # Get trainer
    trainer = Trainer(cfg, device)
    # Start training
    trainer.train()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import logging
import shutil


def set_logging(name=None):
    rank = int(os.getenv('RANK', -1))
    logging.basicConfig(format="%(message)s", level=logging.INFO if (rank in (-1, 0)) else logging.WARNING)
    return logging.getLogger(name)


LOGGER = set_logging(__name__)
NCOLS = min(100, shutil.get_terminal_size().columns)


def write_tbloss(tblogger, losses, step):
    tblogger.add_scalar("train/loss/total", losses[0], step + 1)
    tblogger.add_scalar("train/loss/regression", losses[1], step + 1)
    tblogger.add_scalar("train/loss/standard", losses[2], step + 1)


def write_tbacc(tblogger, acc, epoch, task):
    tblogger.add_scalar("acc/{}".format(task), acc, epoch + 1)


def write_tbloss_val(tblogger, losses, step):
    tblogger.add_scalar("training/loss/val", losses[0], step + 1)


def write_tbacc_one_plot(tblogger, acc, epoch):
    tblogger.add_scalars("acc_multiple/", {'train': acc[0], 'val': acc[1]}, epoch + 1)


def write_tbimg(tblogger, imgs, step, type='train'):
    'Display train_batch and validation predictions to tensorboard.'
    if type == 'train':
        tblogger.add_image(f'train_batch', imgs, step + 1, dataformats='HWC')
    elif type == 'val':
        for idx, img in enumerate(imgs):
            tblogger.add_image(f'val_img_{idx + 1}', img, step + 1, dataformats='HWC')
    else:
        LOGGER.warning('WARNING: Unknown image type to visualize.\n')

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import logging
import shutil

import numpy as np
import cv2
import torch


def set_logging(name=None):
    rank = int(os.getenv('RANK', -1))
    logging.basicConfig(format="%(message)s", level=logging.INFO if (rank in (-1, 0)) else logging.WARNING)
    return logging.getLogger(name)


LOGGER = set_logging(__name__)
NCOLS = min(100, shutil.get_terminal_size().columns)


def write_tbloss(tblogger, losses, name, step):
    tblogger.add_scalar(f"train/loss/{name}", losses, step + 1)


def write_tbacc(tblogger, acc, step, task):
    tblogger.add_scalar("acc/{}".format(task), acc, step + 1)


def write_tblr(tblogger, lr, step):
    tblogger.add_scalar("train/lr", lr, step + 1)


def write_tbacc_seg(tblogger, val_score, step):
    tblogger.add_scalar(f'val/Overall_Acc', val_score['Overall Acc'], step)
    tblogger.add_scalar(f'val/Mean_Acc', val_score['Mean Acc'], step)
    tblogger.add_scalar(f'val/FreqW_Acc', val_score['FreqW Acc'], step)
    tblogger.add_scalar(f'val/Mean_IoU', val_score['Mean IoU'], step)
    #
    import matplotlib.pyplot as plt

    def addlabels(x, y):
        for i in range(len(x)):
            plt.text(i, y[i], y[i], ha='center')

    class_iou = val_score['Class IoU']
    fig = plt.figure()
    x = np.arange(len(class_iou))
    y = np.array(list(class_iou.values()))
    addlabels(x, y.round(3))
    plt.bar(x, y, width=0.5)
    plt.xticks(x, x)
    plt.ylim([0, 1.1])
    tblogger.add_figure('val/Class_IoU', fig, step)
    plt.close()


def write_segimg(tblogger, img, target, pred, index, step, type='val'):
    assert len(img.shape) == 3, f'the number of img should be 1...'
    if img.shape[2] == 3:
        result = np.concatenate([img, target, pred], axis=1)
    else:
        result = np.concatenate([np.concatenate([img, img, img], axis=-1), target, pred], axis=1)
    tblogger.add_image(f'{type}/img/{index}', result, step, dataformats='HWC')


def write_tbimg(tblogger, imgs, step, type='train'):
    """Display train_batch and validation predictions to tensorboard."""
    if type == 'train':
        tblogger.add_image(f'train_batch', imgs, step + 1, dataformats='HWC')
    elif type == 'val':
        for idx, img in enumerate(imgs):
            tblogger.add_image(f'val_img_{idx + 1}', img, step + 1, dataformats='HWC')
    else:
        LOGGER.warning('WARNING: Unknown image type to visualize.\n')


def write_tbcam_rescls(tblogger, imgs_t, cam_t, onehot_t, step, type='train'):
    cam_t = cam_t.cpu().numpy()
    imgs_t = imgs_t.cpu().numpy()
    for idx in range(imgs_t.shape[0]):
        gcam_img = ((cam_t[idx] - np.min(cam_t[idx])) / (np.max(cam_t[idx]) - np.min(cam_t[idx]) + 1e-6))
        gcam_img = (255 * gcam_img).astype(np.uint8)
        gcam_img = cv2.applyColorMap(cv2.cvtColor(np.squeeze(gcam_img, axis=0), cv2.COLOR_GRAY2BGR), cv2.COLORMAP_JET)
        gcam_img_w_rgb = np.concatenate([(imgs_t[idx].transpose(1, 2, 0) * 255).astype(np.uint8),
                                         (gcam_img.astype(np.uint8))], axis=1)
        #
        if onehot_t[idx] == 1:
            tblogger.add_image(f'{type}/cam/{onehot_t[idx]}/{idx + 1}', gcam_img_w_rgb, step + 1, dataformats='HWC')


def write_tbcam(tblogger, imgs_t, img_tis, cam_t, cam_tis, step, type='train', gain_type='total'):
    """Display train_batch and validation predictions to tensorboard."""
    for idx, img in enumerate(imgs_t):
        if gain_type == 'meta':
            gcam_img = ((cam_t[idx] - np.min(cam_t[idx]))
                        / (np.max(cam_t[idx]) - np.min(cam_t[idx])))
            gcam_img[gcam_img <= 0.5] = 0
            gcam_img = (255 * gcam_img).astype(np.uint8)
            gcam_img = cv2.applyColorMap(np.squeeze(gcam_img, axis=0), cv2.COLORMAP_JET)
            gcam_img = cv2.cvtColor(gcam_img, cv2.COLOR_BGR2RGB)
            gcam_img_w_rgb = (0.4 * img.transpose(1, 2, 0) * 255 + 0.6 * gcam_img).astype(np.uint8)
            #
            tblogger.add_image(f'{type}/cam_meta/{idx + 1}', gcam_img_w_rgb, step + 1, dataformats='HWC')
            #
            gcam_img_w_rgb = []
            for cls_idx in range(len(cam_tis)):
                gcam_img = ((cam_tis[cls_idx][idx] - np.min(cam_tis[cls_idx][idx]))
                            / (np.max(cam_tis[cls_idx][idx]) - np.min(cam_tis[cls_idx][idx])))
                gcam_img[gcam_img <= 0.5] = 0
                gcam_img = (255 * gcam_img).astype(np.uint8)
                gcam_img = cv2.applyColorMap(np.squeeze(gcam_img, axis=0), cv2.COLORMAP_JET)
                gcam_img = cv2.cvtColor(gcam_img, cv2.COLOR_BGR2RGB)
                gcam_img_w_rgb.append((0.4 * img_tis[idx].transpose(1, 2, 0) * 255 + 0.6 * gcam_img).astype(np.uint8))
            img_cam = np.concatenate(gcam_img_w_rgb, axis=1)
            tblogger.add_image(f'{type}/cam_tis/{idx + 1}', img_cam, step + 1, dataformats='HWC')
            #
        else:
            raise ValueError


#
# def write_tbcam(tblogger, imgs, cam, onehot_t, onehot_tis, step, type='train', gain_type='total'):
#     """Display train_batch and validation predictions to tensorboard."""
#     for batch_idx, img in enumerate(imgs):
#         if gain_type == 'total':
#             gcam_img = ((cam[batch_idx].detach().cpu() - torch.min(cam[batch_idx].detach().cpu()))
#                         / (torch.max(cam[batch_idx].detach().cpu()) - torch.min(cam[batch_idx].detach().cpu())))
#             gcam_img[gcam_img <= 0.5] = 0
#             gcam_img = (255 * gcam_img).type(torch.uint8)
#             gcam_img = cv2.applyColorMap(gcam_img.squeeze(dim=0).numpy(), cv2.COLORMAP_JET)
#             gcam_img = cv2.cvtColor(gcam_img, cv2.COLOR_BGR2RGB)
#             gcam_img_w_rgb = (0.3 * img.transpose(1, 2, 0) * 255 + 0.7 * gcam_img).astype(np.uint8)
#             #
#             img_cam = (
#                 np.concatenate([(img.transpose(1, 2, 0) * 255).astype(np.uint8), gcam_img_w_rgb], axis=1)).astype(
#                 np.uint8)
#             tblogger.add_image(f'{type}/cam_tis/{batch_idx + 1}', img_cam, step + 1, dataformats='HWC')
#         elif gain_type == 'per':
#             gcam_img_w_rgb = []
#             for cls_idx in range(len(cam)):
#                 gcam_img = ((cam[cls_idx][batch_idx].detach().cpu() - torch.min(cam[cls_idx][batch_idx].detach().cpu()))
#                             / (torch.max(cam[cls_idx][batch_idx].detach().cpu()) - torch.min(
#                             cam[cls_idx][batch_idx].detach().cpu())))
#                 gcam_img[gcam_img <= 0.7] = 0
#                 gcam_img = (255 * gcam_img).type(torch.uint8)
#                 gcam_img = cv2.applyColorMap(gcam_img.squeeze(dim=0).numpy(), cv2.COLORMAP_JET)
#                 gcam_img = cv2.cvtColor(gcam_img, cv2.COLOR_BGR2RGB)
#                 gcam_img_w_rgb.append((0.3 * img.transpose(1, 2, 0) * 255 + 0.7 * gcam_img).astype(np.uint8))
#             img_cam = np.concatenate(gcam_img_w_rgb, axis=1)
#             tblogger.add_image(f'{type}/cam_tis/{batch_idx + 1}', img_cam, step + 1, dataformats='HWC')
#         elif gain_type == 'meta':
#             #
#             # if onehot_tis[batch_idx].mean().item() != -1:
#             #     cam_tis = cam[0]
#             #     gcam_img_w_rgb = []
#             #     for cls_idx in range(len(cam_tis)):
#             #         gcam_img = ((cam_tis[cls_idx][batch_idx].detach().cpu() - torch.min(
#             #             cam_tis[cls_idx][batch_idx].detach().cpu()))
#             #                     / (torch.max(cam_tis[cls_idx][batch_idx].detach().cpu()) - torch.min(
#             #                     cam_tis[cls_idx][batch_idx].detach().cpu())))
#             #         gcam_img[gcam_img <= 0.5] = 0
#             #         gcam_img = (255 * gcam_img).type(torch.uint8)
#             #         gcam_img = cv2.applyColorMap(gcam_img.squeeze(dim=0).numpy(), cv2.COLORMAP_JET)
#             #         gcam_img = cv2.cvtColor(gcam_img, cv2.COLOR_BGR2RGB)
#             #         gcam_img_w_rgb.append((0.3 * img.transpose(1, 2, 0) * 255 + 0.7 * gcam_img).astype(np.uint8))
#             #     img_cam = np.concatenate(gcam_img_w_rgb, axis=1)
#             #     tblogger.add_image(f'{type}/cam_tis/{batch_idx + 1}', img_cam, step + 1, dataformats='HWC')
#             #     #
#             #     del cam_tis
#             #     del gcam_img
#             #     del img_cam
#             #     del gcam_img_w_rgb
#             #
#             if onehot_t[batch_idx][1] == 1:
#                 cam_meta = cam[1]
#                 gcam_img = ((cam_meta[1][batch_idx].detach().cpu() - torch.min(cam_meta[1][batch_idx].detach().cpu()))
#                             / (torch.max(cam_meta[1][batch_idx].detach().cpu()) - torch.min(
#                             cam_meta[1][batch_idx].detach().cpu())))
#                 gcam_img[gcam_img <= 0.7] = 0
#                 gcam_img = (255 * gcam_img).type(torch.uint8)
#                 gcam_img = cv2.applyColorMap(gcam_img.squeeze(dim=0).numpy(), cv2.COLORMAP_JET)
#                 gcam_img = cv2.cvtColor(gcam_img, cv2.COLOR_BGR2RGB)
#                 gcam_img_w_rgb = (0.3 * img.transpose(1, 2, 0) * 255 + 0.7 * gcam_img).astype(np.uint8)
#                 #
#                 tblogger.add_image(f'{type}/cam_meta/{batch_idx + 1}', gcam_img_w_rgb, step + 1, dataformats='HWC')
#         else:
#             raise ValueError


def write_tbval(tblogger, P, R, step, task='train'):
    tblogger.add_scalars(f"{task}/precision", {
        'Invasive Cancer': P[1],
        'Tumor': P[2],
        'DCIS, LCIS': P[3],
    }, step + 1)
    tblogger.add_scalars(f"{task}/recall", {
        'Invasive Cancer': R[1],
        'Tumor': R[2],
        'DCIS, LCIS': R[3],
    }, step + 1)

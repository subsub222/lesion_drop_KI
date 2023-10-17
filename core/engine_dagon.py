from tqdm import tqdm

import cv2
import numpy as np
import math
import time
import torch
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
from utils.events import LOGGER, NCOLS, write_tbacc, write_tbloss, write_tblr

from models.dragonnet.dragonnet import dragonnet
from datasets.lesion import lesionDataset
from torch.utils.data.dataloader import DataLoader

from utils.ema_single import ModelEMA
from solver.build import build_optimizer, build_lr_scheduler

import os
from copy import deepcopy

torch.set_float32_matmul_precision('high')


class Trainer():
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = torch.device('cuda:{}'.format(cfg.gpuid) if cfg.gpuid is not None else 'cpu')
        self.save_dir = cfg.save_dir
        # ===== Model =====
        self.model = self.get_model(cfg, device)
        # ===== Optimizer =====
        self.optimizer = self.get_optimizer(cfg, self.model)
        self.scaler = amp.GradScaler(enabled=self.device != 'cpu')
        # ===== Scheduler =====
        self.scheduler = self.get_lr_scheduler(cfg, self.optimizer)
        # ===== EMA =====
        self.ema = ModelEMA(self.model)
        # ===== tensorboard =====
        self.tblogger = SummaryWriter(self.save_dir)
        # ===== DataLoader =====
        self.train_loader, self.val_loader = self.get_data_loader(self.cfg)
        # ===== Loss =====
        self.compute_loss = self.set_criterion()
        # ===== resume =====
        if cfg.model_dragon.resume:
            self.ckpt = torch.load(cfg.model_dragon.resume, map_location=self.device)
        if hasattr(self, "ckpt"):
            resume_state_dict = self.ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
            self.model.load_state_dict(resume_state_dict, strict=True)  # load
            self.optimizer.load_state_dict(self.ckpt['optimizer'])
            self.ema.ema.load_state_dict(self.ckpt['ema'].float().state_dict())
        #
        self.start_epoch = 0
        self.max_epoch = cfg.epochs
        self.batch_size = cfg.batchsize
        self.max_stepnum = len(self.train_loader)
        #
        self.last_opt_step = 0
        self.accumulate = max(1, round(64 / cfg.batchsize))
        self.best_acc = 0.0

    def train(self):
        start_time = time.time()
        try:
            LOGGER.info(f'Training start...')
            for epoch in range(self.start_epoch, self.max_epoch):
                self.train_one_epoch(epoch)
            LOGGER.info(f'\nTraining completed in {(time.time() - start_time) / 3600:.3f} hours.')
            # strip optimizers for saved pt model
            LOGGER.info(f'Strip optimizer from the saved pt model...')
        except Exception as _:
            LOGGER.error('ERROR in training loop or eval/save model.')
            raise
        finally:
            if self.device != 'cpu':
                torch.cuda.empty_cache()

    def train_one_epoch(self, epoch):
        mean_loss = 0.0
        cnt_correct = 0.0
        cnt_total = 0.0
        #
        pbar = tqdm(enumerate(self.train_loader),
                    total=self.max_stepnum, ncols=NCOLS, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        self.model.train()
        for step, batch_data in pbar:
            x, y, t = batch_data[2], batch_data[3], batch_data[4]
            yt = torch.cat([y, t], dim=1).to(self.device)

            preds = self.model(x.to(self.device))
            loss = self.compute_loss._calc(yt, preds, reduction='sum')

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.update_optimizer(step + epoch * self.max_stepnum)
            #
            mean_loss = (mean_loss * step + loss) / (step + 1)
            #
            result = self.calc_acc(yt, preds)
            cnt_correct += result[0]
            cnt_total += result[1]
            #
            reset_acc = self.print_verbose(epoch, step, mean_loss, cnt_correct / cnt_total, pbar)
            if reset_acc:
                cnt_correct = 0.0
                cnt_total = 0.0
        # Update the learning rate of scheduler
        self.scheduler.step()
        write_tblr(self.tblogger, self.scheduler.get_lr()[0], epoch)
        try:
            # Save the model
            self.save(epoch, 'last')
        except Exception as _:
            LOGGER.error('ERROR in evaluate and save model.')
            raise

    def validation(self, epoch, global_step):
        self.model.eval()
        #
        cnt_correct = 0.0
        cnt_total = 0.0
        for step, batch_data in enumerate(self.val_loader):
            x, y, t = batch_data[2], batch_data[3], batch_data[4]
            yt = torch.cat([y, t], dim=1).to(self.device)
            #
            preds = self.model(x.to(self.device))
            #
            result = self.calc_acc(yt, preds)
            cnt_correct += result[0]
            cnt_total += result[1]
        #
        write_tbacc(self.tblogger, acc=cnt_correct / cnt_total, step=(epoch * self.max_stepnum) + global_step,
                    task='val')
        #
        self.model.train()
        return cnt_correct / cnt_total

    @staticmethod
    def calc_acc(concat_true, concat_pred):
        t_true = concat_true[:, 2:]
        t_pred = torch.sigmoid(concat_pred[:, 4:5])
        t_pred[t_pred > 0.5] = 1
        t_pred[t_pred <= 0.5] = 0
        cnt_correct = (t_pred == t_true).sum()
        cnt_total = t_pred.shape[0]
        return cnt_correct.float().item(), float(cnt_total)

    def save(self, epoch, prefix, val_acc=None):
        # Check save directory
        save_ckpt_dir = os.path.join(self.save_dir, 'weights')
        if not os.path.exists(save_ckpt_dir):
            os.makedirs(save_ckpt_dir)

        # Make checkpoint dictionary
        ckpt = {
            'model': deepcopy(self.model).half(),
            'ema': deepcopy(self.ema.ema).half(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'val_acc': val_acc,
        }

        # Save last checkpoint
        # In the folder, run/train/exp/weights, last and best ckpts are saved
        if prefix == 'best':
            torch.save(ckpt, os.path.join(save_ckpt_dir, f'{prefix}_ckpt_{val_acc}.pth'))
        else:
            torch.save(ckpt, os.path.join(save_ckpt_dir, f'{prefix}_ckpt.pth'))

    def print_verbose(self, epoch, step, mean_loss, acc, pbar):
        reset = False
        if (epoch != 0 or (epoch == 0 and step != 0)) and (
                (epoch * self.max_stepnum + step) % self.cfg.training_params.print_freq == 0):
            pbar.set_description(('%10s' + '%10.4g') % (f'{epoch + 1}/{self.max_epoch}', mean_loss))
            write_tbloss(self.tblogger, losses=mean_loss, name='tarreg', step=(epoch * self.max_stepnum) + step)
        if (epoch != 0 or (epoch == 0 and step != 0)) and (
                (epoch * self.max_stepnum + step) % self.cfg.training_params.acc_freq == 0):
            write_tbacc(self.tblogger, acc=acc, step=(epoch * self.max_stepnum) + step, task='train')
            reset = True
        if (epoch != 0 or (epoch == 0 and step != 0)) and (
                (epoch * self.max_stepnum + step) % self.cfg.training_params.eval_freq == 0):
            val_acc = self.validation(epoch, step)
            if self.best_acc < val_acc:
                self.best_acc = val_acc
                self.save(epoch, 'best', val_acc=val_acc)
        return reset

    def update_optimizer(self, curr_step):
        if curr_step - self.last_opt_step >= self.accumulate:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.ema:
                self.ema.update(self.model)
            self.last_opt_step = curr_step

    @staticmethod
    def set_criterion():
        from models.dragonnet.loss import TarReg_Loss
        loss = TarReg_Loss(alpha=3., beta=3.)
        return loss

    @staticmethod
    def get_data_loader(cfg):
        object = lesionDataset(cfg, None, mode='dragon', task='train')
        train_loader = DataLoader(
            object, batch_size=cfg.batchsize, shuffle=True,
            num_workers=cfg.workers, collate_fn=lesionDataset.collate_fn,
            pin_memory=True
        )
        object = lesionDataset(cfg, None, mode='dragon', task='test')
        test_loader = DataLoader(
            object, batch_size=cfg.batchsize, shuffle=True,
            num_workers=cfg.workers, collate_fn=lesionDataset.collate_fn,
            pin_memory=True
        )
        return train_loader, test_loader

    @staticmethod
    def get_model(cfg, device):
        model = dragonnet(in_dim=cfg.model_dragon.input_dim, rep_dim=cfg.model_dragon.rep_dim,
                          hypo_dim=cfg.model_dragon.hypo_dim,
                          sec_dim=cfg.model_dragon.sec_dim,
                          thir_dim=cfg.model_dragon.thir_dim).to(device)
        return model

    @staticmethod
    def get_optimizer(cfg, model):
        accumulate = max(1, round(64 / cfg.batchsize))
        # If args.batch_size < 64, (args.batch_size * accumulate / 64) ~= 1.
        # If args.batch_size > 64, (args.batch_size * accumulate / 64) ~= (args.batch_size / 64)
        # Hence, If batch_size is larger than 64, weight_decay becomes larger.
        cfg.solver.weight_decay *= cfg.batchsize * accumulate / 64
        # cfg.solver.lr0 *= args.batch_size / (self.world_size * args.bs_per_gpu)
        optimizer = build_optimizer(cfg, model)
        return optimizer

    @staticmethod
    def get_lr_scheduler(cfg, optimizer):
        lr_scheduler = build_lr_scheduler(cfg, optimizer)
        return lr_scheduler

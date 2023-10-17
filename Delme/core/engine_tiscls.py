from tqdm import tqdm

import os
import time
import torch
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
from utils.events import LOGGER, NCOLS, write_tbloss, write_tblr, write_tbval, write_tbcam

from datasets.lesion import lesionDataset
from torch.utils.data.dataloader import DataLoader

from utils.ema_single import ModelEMA
from solver.build import build_optimizer, build_lr_scheduler

from torchvision.transforms import v2

torch.set_float32_matmul_precision('high')
import numpy as np

from utils.train_val import CELoss, VarifocalLoss_binary
from torchvision.ops.focal_loss import sigmoid_focal_loss
from copy import deepcopy


class Trainer():
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = torch.device('cuda:{}'.format(cfg.gpuid) if cfg.gpuid is not None else 'cpu')
        self.save_dir = cfg.save_dir
        # ===== Model =====
        self.num_tiscls_classes = cfg.model_tiscls.num_tis_classes
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
        self.compute_loss = self.set_criterion(cfg)
        # ===== resume =====
        self.check_resume(cfg)
        #
        self.start_epoch = 0
        self.max_epoch = cfg.epochs
        self.batch_size = cfg.batchsize
        self.max_stepnum = len(self.train_loader)
        #
        self.last_opt_step = 0
        self.accumulate = max(1, round(cfg.solver.accumulate_criterion / cfg.batchsize))
        self.best_score = 0.0
        self.loss_lambda = 1.0

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
        mean_loss = dict()
        #
        pbar = tqdm(enumerate(self.train_loader),
                    total=self.max_stepnum, ncols=NCOLS, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                    mininterval=2, maxinterval=10,
                    ascii=True, leave=True)
        #
        TP1 = torch.zeros(self.num_tiscls_classes)
        FP1 = torch.zeros(self.num_tiscls_classes)
        FN1 = torch.zeros(self.num_tiscls_classes)
        #
        self.model.train()
        for step, batch_data in pbar:
            imgs, mask, label = batch_data[0], batch_data[1], batch_data[6]
            imgs = imgs.to(self.device, dtype=torch.float32)
            label = label.to(self.device, dtype=torch.float32)
            with amp.autocast(enabled=self.device != 'cpu'):
                outputs = self.model(imgs, label, mask, self.cfg.gain)
                loss, loss_dict = self.calc_loss(outputs, label)
                #
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.update_optimizer(step)
                #
                for key in list(loss_dict.keys()):
                    if key not in list(mean_loss.keys()):
                        mean_loss.setdefault(key, loss_dict[key])
                    else:
                        mean_loss[key] = (mean_loss[key] * step + loss_dict[key]) / (step + 1)
                #
                preds = torch.where(torch.sigmoid(outputs) > 0.5, 1, 0).cpu()
                TP1 += torch.count_nonzero(label.cpu() * preds, dim=0)
                FP1 += torch.count_nonzero(torch.where((preds - label.cpu()) > 0, 1, 0), dim=0)
                FN1 += torch.count_nonzero(torch.where((label.cpu() - preds) > 0, 1, 0), dim=0)
                #
                self.print_verbose(epoch, step, mean_loss, pbar)
        # Update the learning rate of scheduler
        self.scheduler.step()
        write_tblr(self.tblogger, self.scheduler.get_last_lr()[0], epoch)
        write_tbval(self.tblogger, TP1 / (TP1 + FP1 + 1e-6), TP1 / (TP1 + FN1 + 1e-6), epoch + 1, 'train')
        if self.cfg.gain:
            write_tbcam(self.tblogger, imgs.cpu().numpy(), self.model.get_gcam(),
                        (epoch * self.max_stepnum + step + 1),  type='train', gain_type=self.cfg.gaintype)
        try:
            # Save the model
            self.save('last')
        except Exception as _:
            LOGGER.error('ERROR in evaluate and save model.')
            raise

    def calc_loss(self, outputs, targets):
        loss = 0.0
        loss_dict = dict()
        for key in list(self.compute_loss.keys()):
            if key == 'ce':
                temp_loss = self.loss_lambda * self.compute_loss[key](outputs, targets)
                loss_dict.setdefault('ce_loss', deepcopy(temp_loss.detach().cpu().numpy()))
                loss += temp_loss
            elif key == 'focal':
                temp_loss = self.loss_lambda * self.compute_loss[key](outputs, targets, reduction='sum')
                loss_dict.setdefault('focal_loss', deepcopy(temp_loss.detach().cpu().numpy()))
                loss += temp_loss
            else:
                raise NotImplementedError('Invalid loss function name...')
        #
        if self.cfg.gain:
            loss_dict.setdefault('am_loss', deepcopy(self.model.get_loss_am()))
            loss += self.model.get_loss_am()
        return loss, loss_dict

    def update_optimizer(self, curr_step):
        if curr_step - self.last_opt_step >= self.accumulate:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.last_opt_step = curr_step

    def check_resume(self, cfg):
        if cfg.model_tiscls.resume:
            try:
                ckpt = torch.load(cfg.model_tiscls.resume, map_location=self.device)
                resume_state_dict = ckpt['model_state']  # checkpoint state_dict as FP32
                self.model.load_state_dict(resume_state_dict, strict=True)  # load
                # self.optimizer.load_state_dict(ckpt['optimizer_state'])
                # self.scheduler.load_state_dict(ckpt['scheduler_state'])
                print("Training state restored from %s" % cfg.model_tiscls.resume)
                del ckpt
            except:
                raise LOGGER.error('CHECKPOINT DOESN\'T EXIST')

    def set_criterion(self, cfg):
        loss_names = cfg.loss.split('+')
        criterion = dict()
        for loss_name in loss_names:
            if loss_name == 'ce':
                criterion.setdefault(loss_name, torch.nn.BCEWithLogitsLoss(reduction=cfg.lossredu).to(self.device))
            elif loss_name == 'focal':
                criterion.setdefault(loss_name, sigmoid_focal_loss)
                # criterion.setdefault(loss_name, VarifocalLoss_binary().to(self.device))
            else:
                raise NotImplementedError(f"Not Implemented ... {loss_name}")
        return criterion

    def get_model(self, cfg, device):
        if (cfg.model_tiscls.model in
                ['resnet4', 'resnet6', 'resnet10', 'resnet18', 'resnet24', 'resnet34', 'resnet50']):
            if cfg.gaintype == 'total':
                from models.resnet.resnet_gain import resnet as net
            elif cfg.gaintype == 'per':
                from models.resnet.resnet_gain_per_class import resnet as net
            else:
                raise NotImplementedError
            model = net(cfg, cfg.model_tiscls.model, self.num_tiscls_classes)
            model.init_weights()
        else:
            raise NotImplementedError(f'{cfg.model_tiscls.model} Not Implemented...!')
        return model.to(device)

    @staticmethod
    def get_data_loader(cfg):
        transform = v2.Compose([
            v2.RandomCrop(size=(cfg.dataset.input_size, cfg.dataset.input_size)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
        ])
        #
        assert cfg.mode in ['tiscls'], f'Invalid mode....'
        #
        object = lesionDataset(cfg, mode=cfg.mode, task='train', transform=transform)
        train_loader = DataLoader(
            object, batch_size=cfg.batchsize, shuffle=True,
            num_workers=cfg.workers, collate_fn=lesionDataset.collate_fn,
            pin_memory=True
        )
        #
        object = lesionDataset(cfg, mode=cfg.mode, task='test', transform=transform)
        test_loader = DataLoader(
            object, batch_size=cfg.batchsize, shuffle=False,
            num_workers=cfg.workers, collate_fn=lesionDataset.collate_fn,
            pin_memory=True
        )
        return train_loader, test_loader

    @staticmethod
    def get_optimizer(cfg, model):
        accumulate = max(1, round(cfg.solver.accumulate_criterion / cfg.batchsize))
        # If args.batch_size < 64, (args.batch_size * accumulate / 64) ~= 1.
        # If args.batch_size > 64, (args.batch_size * accumulate / 64) ~= (args.batch_size / 64)
        # Hence, If batch_size is larger than 64, weight_decay becomes larger.
        cfg.solver.weight_decay *= cfg.batchsize * accumulate / cfg.solver.accumulate_criterion
        # cfg.solver.lr0 *= args.batch_size / (self.world_size * args.bs_per_gpu)
        optimizer = build_optimizer(cfg, model)
        return optimizer

    @staticmethod
    def get_lr_scheduler(cfg, optimizer):
        lr_scheduler = build_lr_scheduler(cfg, optimizer)
        return lr_scheduler

    def save(self, prefix, val_score=None):
        save_ckpt_dir = os.path.join(self.save_dir, 'weights')
        if not os.path.exists(save_ckpt_dir):
            os.makedirs(save_ckpt_dir)
        #
        if prefix == 'last':
            torch.save({
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
            }, os.path.join(save_ckpt_dir, '{}_ckpt.pth'.format(prefix)))
        else:
            torch.save({
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "val_score": val_score
            }, os.path.join(save_ckpt_dir, f'{prefix}_ckpt_{val_score}.pth'))
        #
        print("Model saved as %s" % save_ckpt_dir)

    def print_verbose(self, epoch, step, loss_dict, pbar):
        if (epoch != 0 or (epoch == 0 and step != 0)) and (
                (epoch * self.max_stepnum + step + 1) % self.cfg.training_params.print_freq == 0):
            state = f'{epoch + 1}/{self.max_epoch}'
            for key in list(loss_dict.keys()):
                state += ' %10.4g' % loss_dict[key]
            pbar.set_description(state)
            for key in list(loss_dict.keys()):
                write_tbloss(self.tblogger, losses=[loss_dict[key]], name=key,
                             step=(epoch * self.max_stepnum + step + 1))
        if (epoch != 0 or (epoch == 0 and step != 0)) and (
                (epoch * self.max_stepnum + step + 1) % self.cfg.training_params.eval_freq == 0):
            self.model.eval()
            #
            acc = self.validate(model=self.model, loader=self.val_loader, device=self.device, epoch=epoch)
            #
            if self.best_score < acc:
                self.best_score = acc
                self.save('best', val_score=acc)
            self.model.train()

    def validate(self, model, loader, device, epoch):
        """Do validation and return specified samples"""
        TP1 = torch.zeros(self.num_tiscls_classes)
        FP1 = torch.zeros(self.num_tiscls_classes)
        FN1 = torch.zeros(self.num_tiscls_classes)
        with torch.no_grad():
            for batch_id, batch_data in tqdm(enumerate(loader), total=len(loader),
                                             ncols=NCOLS,
                                             mininterval=2, maxinterval=10,
                                             ascii=True, leave=True):
                imgs, labels = batch_data[0], batch_data[6]
                imgs = imgs.to(device, dtype=torch.float32)
                #
                outputs = model(imgs, labels, None)
                #
                preds = torch.where(torch.sigmoid(outputs) > 0.5, 1, 0).cpu()
                TP1 += torch.count_nonzero(labels * preds, dim=0)
                FP1 += torch.count_nonzero(torch.where((preds - labels) > 0, 1, 0), dim=0)
                FN1 += torch.count_nonzero(torch.where((labels - preds) > 0, 1, 0), dim=0)

        write_tbval(self.tblogger, TP1 / (TP1 + FP1 + 1e-6), TP1 / (TP1 + FN1 + 1e-6), epoch + 1, 'val')
        return round(torch.mean(TP1 / (TP1 + FP1 + 1e-6)).numpy() * torch.mean(TP1 / (TP1 + FN1 + 1e-6)).numpy(), 3)


from tqdm import tqdm

import os
import time
import torch
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
from utils.events import LOGGER, NCOLS, write_tbloss, write_tblr, write_segimg, write_tbacc_seg

from models.deeplabv3 import utils
from datasets.lesion import lesionDataset
from torch.utils.data.dataloader import DataLoader

from utils.ema_single import ModelEMA
from solver.build import build_optimizer, build_lr_scheduler

from models.deeplabv3.metrics import StreamSegMetrics
from torchvision.transforms import v2

torch.set_float32_matmul_precision('high')
import numpy as np

from models.unet.train_val import CELoss, RMSELoss, DiceLoss, FocalLoss


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
        self.accumulate = max(1, round(64 / cfg.batchsize))
        self.best_score = 0.0
        #
        self.metrics = StreamSegMetrics(cfg.model_seg.num_classes)

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
        #
        pbar = tqdm(enumerate(self.train_loader),
                    total=self.max_stepnum, ncols=NCOLS, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        self.model.train()
        for step, batch_data in pbar:
            imgs, segs = batch_data[0], batch_data[1]
            imgs = imgs.to(self.device, dtype=torch.float32)
            segs = segs.to(self.device, dtype=torch.long)
            with amp.autocast(enabled=self.device != 'cpu'):
                #
                outputs = self.model(imgs)
                #
                loss = self.calc_loss(outputs, segs)
            #
            self.optimizer.zero_grad()
            loss.backward()
            self.update_optimizer(step)
            #
            mean_loss = (mean_loss * step + loss.detach().cpu().numpy()) / (step + 1)
            #
            self.print_verbose(epoch, step, mean_loss, pbar)

        # Update the learning rate of scheduler
        self.scheduler.step()
        write_tblr(self.tblogger, self.scheduler.get_lr()[0], epoch)
        try:
            # Save the model
            self.save('last')
        except Exception as _:
            LOGGER.error('ERROR in evaluate and save model.')
            raise

    def calc_loss(self, outputs, targets):
        loss = 0.0
        for key in list(self.compute_loss.keys()):
            if key == 'focal':
                loss += self.compute_loss[key](outputs, targets)
            elif key == 'ce':
                loss += self.compute_loss[key](outputs, targets)
            elif key == 'mae':
                loss += self.compute_loss[key](outputs.max(dim=1)[1].float(), targets.max(dim=1)[1].float())
            elif key == 'dice':
                loss += self.compute_loss[key](outputs.max(dim=1)[1].float(), targets.max(dim=1)[1].float())
            elif key == 'mse':
                loss += self.compute_loss[key](outputs.max(dim=1)[1].float(), targets.max(dim=1)[1].float())
            elif key == 'rmse':
                loss += self.compute_loss[key](outputs.max(dim=1)[1].float(), targets.max(dim=1)[1].float())
            else:
                raise NotImplementedError('Invalid loss function name...')
        return loss

    def update_optimizer(self, curr_step):
        if curr_step - self.last_opt_step >= self.accumulate:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.last_opt_step = curr_step

    def check_resume(self, cfg):
        if cfg.model_seg.resume:
            try:
                ckpt = torch.load(cfg.model_seg.resume, map_location=self.device)
                resume_state_dict = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
                self.model.load_state_dict(resume_state_dict, strict=True)  # load
                self.optimizer.load_state_dict(ckpt['optimizer_state'])
                self.scheduler.load_state_dict(ckpt['scheduler_state'])
                self.ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
                self.ema.updates = ckpt['updates']
                print("Training state restored from %s" % cfg.model_seg.resume.ckpt)
                del ckpt
            except:
                raise LOGGER.error('CHECKPOINT DOESN\'T EXIST')

    def set_criterion(self, cfg):
        loss_names = cfg.loss.split('+')
        criterion = dict()
        for loss_name in loss_names:
            if loss_name == 'focal':
                criterion.setdefault(loss_name, FocalLoss(reduction=cfg.lossredu, batch_average=False))
            elif loss_name == 'ce':
                criterion.setdefault(loss_name, CELoss(reduction=cfg.lossredu))
            elif loss_name == 'dice':
                criterion.setdefault(loss_name, DiceLoss())
            elif loss_name == 'mse':
                criterion.setdefault(loss_name, torch.nn.MSELoss())
            elif loss_name == 'rmse':
                criterion.setdefault(loss_name, RMSELoss())
            elif loss_name == 'mae':
                criterion.setdefault(loss_name, torch.nn.L1Loss())
            else:
                raise NotImplementedError(f"Not Implemented ... {loss_name}")
        return criterion

    @staticmethod
    def get_model(cfg, device):
        if cfg.type == 'deeplab':
            assert cfg.inchannel == 3, f'Invalid in_channel. For {cfg.model_seg.type}, in_channel should be 3..'
            from models.deeplabv3 import network
            model = network.modeling.__dict__[cfg.model_seg.model](num_classes=cfg.model_seg.num_classes,
                                                                   output_stride=cfg.model_seg.output_stride)
            if cfg.model_seg.separable_conv and 'plus' in cfg.model_seg.model:
                network.convert_to_separable_conv(model.classifier)
            utils.set_bn_momentum(model.backbone, momentum=0.01)
        elif cfg.type == 'unet':
            from models.unet.model import UNet
            model = UNet(n_channels=cfg.inchannel, n_classes=cfg.model_seg.num_classes)
        elif cfg.type == 'unet_short':
            from models.unet.model import UNet_short
            model = UNet_short(n_channels=cfg.inchannel, n_classes=cfg.model_seg.num_classes)
        else:
            raise NotImplementedError(f'model {cfg.model_seg.type} is not implemented...')
        return model.to(device)

    @staticmethod
    def get_data_loader(cfg):
        transform = v2.Compose([
            v2.RandomCrop(size=(cfg.dataset.input_size, cfg.dataset.input_size)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
        ])
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

    def print_verbose(self, epoch, step, mean_loss, pbar):
        if (epoch != 0 or (epoch == 0 and step != 0)) and (
                (epoch * self.max_stepnum + step + 1) % self.cfg.training_params.print_freq == 0):
            pbar.set_description(('%10s' + '%10.4g') % (f'{epoch + 1}/{self.max_epoch}', mean_loss))
            write_tbloss(self.tblogger, losses=[mean_loss], name='seg', step=(epoch * self.max_stepnum + step + 1))
        if (epoch != 0 or (epoch == 0 and step != 0)) and (
                (epoch * self.max_stepnum + step + 1) % self.cfg.training_params.eval_freq == 0):
            self.model.eval()
            val_score = self.validate(
                model=self.model, loader=self.val_loader, device=self.device, metrics=self.metrics,
                step=(epoch * self.max_stepnum + step + 1))
            #
            write_tbacc_seg(self.tblogger, val_score, step=(epoch * self.max_stepnum + step + 1))
            #
            if self.best_score < val_score['Overall Acc']:
                self.best_score = val_score['Overall Acc']
                self.save('best', val_score=round(val_score['Overall Acc'], 3))
            self.model.train()

    def validate(self, model, loader, device, metrics, step):
        """Do validation and return specified samples"""
        metrics.reset()
        with torch.no_grad():
            for batch_id, batch_data in tqdm(enumerate(loader)):
                imgs, segs = batch_data[0], batch_data[1]
                imgs = imgs.to(device, dtype=torch.float32)
                segs = segs.to(device, dtype=torch.long)
                #
                outputs = model(imgs)
                preds = outputs.detach().max(dim=1)[1].cpu().numpy()
                targets = segs.detach().max(dim=1)[1].cpu().numpy()
                # targets = segs.cpu().numpy()
                #
                metrics.update(targets, preds)
                #
                if batch_id < 2:
                    for i in range(len(imgs)):
                        image = imgs[i].detach().cpu().numpy() * 255.
                        target = targets[i]
                        pred = preds[i]
                        #
                        image = image.astype(np.uint8)
                        target = loader.dataset.decode_target(target).astype(np.uint8)
                        pred = loader.dataset.decode_target(pred).astype(np.uint8)
                        #
                        write_segimg(self.tblogger, np.ascontiguousarray(image.transpose(1, 2, 0)), target, pred,
                                     (batch_id * len(imgs) + i), step, 'val')

            score = metrics.get_results()
        return score

import time
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

""" 
    Class that defines the Dice Loss function.
"""


class DiceLoss(nn.Module):

    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def dice_coef(self, y_pred, y_true):
        pred_probs = torch.sigmoid(y_pred)
        y_true_f = y_true.view(-1)
        y_pred_f = pred_probs.view(-1)
        intersection = torch.sum(y_true_f * y_pred_f)
        return (2. * intersection + self.smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + self.smooth)

    def forward(self, y_pred, y_true):
        return -self.dice_coef(y_pred, y_true)


""" 
    Class that defines the Root Mean Square Loss function.
"""


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


"""
    Class that defines the Cross Entropy Loss Function
"""


class FocalLoss(nn.Module):
    def __init__(self, reduction='mean', batch_average=True, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.batch_average = batch_average
        # self.criterion = nn.CrossEntropyLoss(reduction=reduction).to(self.device)
        self.criterion = CELoss(reduction)

    def forward(self, logit, target, gamma=2, alpha=0.5):
        logpt = -self.criterion(logit, target.float())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt
        return loss


class VarifocalLoss_binary(nn.Module):
    def __init__(self):
        super(VarifocalLoss_binary, self).__init__()
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred_logits, label, alpha=0.75, gamma=2.0):
        pred_score = torch.sigmoid(pred_logits)
        weight = alpha * pred_score.pow(gamma) * (1 - label) + label
        # Reference article for AMP : https://computing-jhson.tistory.com/37
        with torch.cuda.amp.autocast(enabled=False):
            loss = (self.bce(pred_logits, label) * weight).sum()
            # loss = (F.binary_cross_entropy(pred_logits.float(), label.float(), reduction='none') * weight).sum()
        return loss


class CELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CELoss, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        if self.reduction == 'mean':
            return -torch.mean(torch.sum(y_true * torch.log(F.softmax(y_pred, dim=1)), dim=1))
        elif self.reduction == 'sum':
            return -torch.sum(torch.sum(y_true * torch.log(F.softmax(y_pred, dim=1)), dim=1))


"""
    Class that defines the Cross Entropy Loss Function
"""


class WCELoss(nn.Module):
    def __init__(self):
        super(WCELoss, self).__init__()

    def forward(self, y_pred, y_true, weights):
        y_true = y_true / (y_true.sum(2).sum(2, dtype=torch.float).unsqueeze(-1).unsqueeze(-1))
        y_true[y_true != y_true] = 0.0
        y_true = torch.sum(y_true, dim=1, dtype=torch.float).unsqueeze(1)
        y_true = y_true * weights.to(torch.float)
        old_range = torch.max(y_true) - torch.min(y_true)
        new_range = 100 - 1
        y_true = (((y_true - torch.min(y_true)) * new_range) / old_range) + 1
        return -torch.mean(torch.sum(y_true * torch.log(F.softmax(y_pred, dim=1)), dim=1))


""" 
    Functions that trains a net.
"""


def train_net(net, device, loader, optimizer, criterion, batch_size, isWCE=False):
    net.train()
    train_loss = AverageMeter()
    time_start = time.time()
    for batch_idx, (data, gt, weights) in enumerate(loader):

        # Use GPU or not
        data, gt = data.to(device), gt.to(device)

        # Forward
        predictions = net(data)

        # Loss Calculation
        if not isWCE:
            loss = criterion(predictions, gt)
        else:
            weights = weights.to(device)
            loss = criterion(predictions, gt, weights)

        # Updates the record
        train_loss.update(loss.item(), predictions.size(0))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('[{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(
            batch_idx * len(data), len(loader) * batch_size,
            100. * batch_idx / len(loader), loss.item()))

    time_dif = time.time() - time_start
    print('\nAverage Training Loss: ' + str(train_loss.avg))
    print('Train Time: It tooks %.4fs to finish the epoch.' % (time_dif))

    return train_loss.avg


""" 
    Function that validates the net.
"""


def val_net(net, device, loader, criterion, batch_size):
    net.eval()
    val_loss = AverageMeter()
    time_start = time.time()
    with torch.no_grad():
        for batch_idx, (data, gt) in enumerate(loader):
            # Use GPU or not
            data, gt = data.to(device), gt.to(device)

            # Forward
            predictions = net(data)

            # Loss Calculation
            loss = criterion(predictions, gt)

            # Updates the record
            val_loss.update(loss.item(), predictions.size(0))

            print('[{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(
                batch_idx * len(data), len(loader) * batch_size,
                100. * batch_idx / len(loader), loss.item()))

    time_dif = time.time() - time_start
    print('\nValidation set: Average loss: ' + str(val_loss.avg))
    print('Validation time: It tooks %.4fs to finish the Validation.' % (time_dif))

    return val_loss.avg


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

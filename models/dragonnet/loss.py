import torch
import torch.nn as nn
from torcheval.metrics import BinaryAccuracy


class FocalLoss_binary(nn.Module):
    '''Binary Focal loss implementation'''

    def __init__(self, gamma=2, weight=(1.0, 1.0)):
        super(FocalLoss_binary, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        treat_0 = torch.where(target == 0)[0]
        treat_1 = torch.where(target == 1)[0]

        pt = torch.sigmoid(input)
        pt_0, pt_1 = pt[treat_0], pt[treat_1]

        logpt_0 = -1.0 * pt_0 ** self.gamma * torch.log(1 - pt_0)
        logpt_1 = -1.0 * (1 - pt_1) ** self.gamma * torch.log(pt_1)

        loss = self.weight[0] * logpt_0.sum() + self.weight[1] * logpt_1.sum()
        return loss


class Base_Loss():
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.name = 'standard_loss'
        # self.bce = nn.BCEWithLogitsLoss(reduction='sum')
        self.focal = FocalLoss_binary(gamma=2, weight=(1.0, 2.0))
        self.bac = BinaryAccuracy()

    def split_pred_(self, concat_pred):
        preds = {}
        preds['y0_pred'] = concat_pred[:, :2]
        preds['y1_pred'] = concat_pred[:, 2:4]
        preds['t_pred'] = concat_pred[:, 4]
        preds['epsilon'] = concat_pred[:, 5:6]  # we're moving epsilon into slot three
        preds['phi'] = concat_pred[:, 6:]
        return preds

    def treatment_bce(self, concat_true, concat_pred, reduction='mean'):
        t_true = concat_true[:, 2:]
        p = self.split_pred(concat_pred)
        lossP = self.focal(p['t_pred'], t_true)
        return lossP

    def treatment_acc(self, concat_true, concat_pred, device):
        t_true = concat_true[:, 2:]
        p = self.split_pred(concat_pred)
        metric = BinaryAccuracy().to(device)
        metric.update(torch.sigmoid(p['t_pred'].squeeze(dim=1)), t_true[:, 0])
        return metric.compute(), p['t_pred'], t_true

    def regression_loss(self, concat_true, concat_pred, reduction='mean'):
        y_true = concat_true[:, :2]
        t_true = concat_true[:, 2:]
        p = self.split_pred(concat_pred)
        loss0 = torch.sum((1. - t_true) * torch.square(y_true - p['y0_pred']))
        loss1 = torch.sum(t_true * torch.square(y_true - p['y1_pred']))
        return (loss0 + loss1) / (t_true.shape[0] if reduction == 'mean' else 1.0)

    def standard_loss(self, concat_true, concat_pred, reduction='mean'):
        lossR = self.regression_loss(concat_true, concat_pred, reduction)

        lossP = self.treatment_bce(concat_true, concat_pred)
        return lossR + self.alpha * lossP


class TarReg_Loss(Base_Loss):
    def __init__(self, alpha=1., beta=1.):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.name = 'tarreg_loss'

    def split_pred(self, concat_pred):
        # generic helper to make sure we dont make mistakes
        preds = {}
        preds['y0_pred'] = concat_pred[:, :2]
        preds['y1_pred'] = concat_pred[:, 2:4]
        preds['t_pred'] = concat_pred[:, 4:5]
        preds['epsilon'] = concat_pred[:, 5:6]  # we're moving epsilon into slot three
        preds['phi'] = concat_pred[:, 6:]
        return preds

    def calc_hstar(self, concat_true, concat_pred):
        # step 2 above
        try:
            p = self.split_pred(concat_pred)
            y_true = concat_true[:, :2]  # ys
            t_true = concat_true[:, 2:]  # t

            t_pred = torch.sigmoid(p['t_pred'])
            t_pred = (t_pred + 0.001) / 1.002  # a little numerical stability trick implemented by Shi
            y_pred = t_true * p['y1_pred'] + (1 - t_true) * p['y0_pred']

            # calling it cc for "clever covariate" as in SuperLearner TMLE literature
            cc = t_true / t_pred - (1 - t_true) / (1 - t_pred)
            h_star = y_pred + p['epsilon'] * cc
        except Exception as e:
            print(e)

        return h_star

    def _calc(self, concat_true, concat_pred, reduction='mean'):
        y_true = concat_true[:, :2]

        standard_loss = self.standard_loss(concat_true, concat_pred, reduction)
        h_star = self.calc_hstar(concat_true, concat_pred)
        # step 3 above
        targeted_regularization = (torch.sum(torch.square(y_true - h_star))) / (y_true.shape[0] if reduction == 'mean' else 1.0)

        # final
        loss = standard_loss + self.beta * targeted_regularization
        return loss

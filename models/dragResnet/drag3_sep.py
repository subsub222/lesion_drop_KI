import torch
import torch.nn as nn


def init_weights_(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class EpsilonLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = nn.Parameter(torch.abs(torch.randn([1, 1], requires_grad=True)))

    def forward(self, x):
        return self.epsilon * torch.ones_like(x[:, 0:1])


class drag3d(nn.Module):
    def __init__(self, cfg, inplanes, num_patches):
        super().__init__()
        self.input_channel = cfg.inchannel * num_patches
        self.inplanes = inplanes
        #
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=int(self.inplanes / 2), kernel_size=(1, 3, 3), stride=(1, 2, 2),
                      padding=0),
            nn.BatchNorm3d(int(self.inplanes / 2)),
            nn.SiLU(),
            nn.MaxPool3d((1, 5, 5), (1, 5, 5), 0),
            nn.Conv3d(in_channels=int(self.inplanes / 2), out_channels=self.inplanes, kernel_size=(1, 3, 3),
                      stride=(1, 2, 2), padding=0),
            nn.BatchNorm3d(self.inplanes),
            nn.SiLU(),
        )
        self.avg_pool = nn.AdaptiveMaxPool3d((num_patches, 1, 1))
        #

    def _forward_conv3d(self, x):
        out = self.conv1(x)
        out = self.avg_pool(out)
        return out.view(out.shape[0], -1)

    def forward(self, imgs):
        out_3d = self._forward_conv3d(imgs)
        return out_3d

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class dragonnet(nn.Module):
    def __init__(self, in_dim, rep_dim, hypo_dim, sec_dim, thir_dim):
        super().__init__()
        # Representation
        self.rep_blocks = nn.Sequential(
            nn.Linear(in_dim, rep_dim),
            nn.ELU(),
            nn.Linear(rep_dim, rep_dim),
            nn.ELU(),
            nn.Linear(rep_dim, rep_dim)
        )

        self.y0_hidden = nn.Sequential(
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(rep_dim, hypo_dim),  # Hypothesis
            nn.ELU(),
            nn.Linear(hypo_dim, sec_dim),  # second
            nn.ELU(),
            nn.Linear(sec_dim, thir_dim),  # third
            nn.ELU(),
        )
        self.y1_hidden = nn.Sequential(
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(rep_dim, hypo_dim),  # Hypothesis
            nn.ELU(),
            nn.Linear(hypo_dim, sec_dim),  # second
            nn.ELU(),
            nn.Linear(sec_dim, thir_dim),  # third
            nn.ELU(),
        )

        self.t_hidden = nn.Sequential(
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(rep_dim, thir_dim),
            nn.ELU(),
        )

        self.y0_pred = nn.Linear(2 * thir_dim, 2)  # pred
        self.y1_pred = nn.Linear(2 * thir_dim, 2)  # pred

        # Propensity prediction
        self.propen_pred = nn.Sequential(
            nn.Linear(thir_dim, 1)
        )

        self.epsilons = EpsilonLayer()

    def forward(self, x):
        # Representation
        phi = self.rep_blocks(x)
        #
        out_0 = self.y0_hidden(phi)
        out_1 = self.y1_hidden(phi)
        out_t = self.t_hidden(phi)
        #
        pred_0 = self.y0_pred(torch.cat([out_0, out_t], dim=1))
        pred_1 = self.y1_pred(torch.cat([out_1, out_t], dim=1))
        pred_propen = self.propen_pred(out_t)
        # Epsilon
        epsilons = self.epsilons(pred_propen)

        return torch.cat([pred_0, pred_1, pred_propen, epsilons, phi], dim=1), phi

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class classify_total(nn.Module):
    def __init__(self, inplanes, num_patches):
        super().__init__()
        fc_in = int(inplanes * num_patches)
        #
        self.linear = nn.Sequential(
            nn.Linear(fc_in, int(fc_in / 2)),
            nn.SiLU(),
            nn.Linear(int(fc_in / 2), 2),
        )

    def forward(self, out_drag, out_res):
        return self.linear(torch.cat([out_drag, out_res], dim=1))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class dragcls(nn.Module):
    def __init__(self, cfg, inplanes, num_patches):
        super().__init__()
        self.net_mode = cfg.netmode
        #
        self.dragonnet = dragonnet(in_dim=cfg.model_dragon.input_dim,
                                   rep_dim=cfg.model_dragon.rep_dim,
                                   hypo_dim=cfg.model_dragon.hypo_dim,
                                   sec_dim=cfg.model_dragon.sec_dim,
                                   thir_dim=cfg.model_dragon.thir_dim)

        if cfg.netmode == 'wimg':
            self.conv3d = drag3d(cfg, inplanes, num_patches)
            self.classify_total = classify_total(inplanes, num_patches)
        elif cfg.netmode == 'default':
            self.conv3d = None
            self.classify_total = None
        else:
            raise NotImplementedError

    def forward(self, x_table, x_img=None):
        assert (x_img is None and self.conv3d is None) or (x_img is not None and self.conv3d is not None)

        out_drag, phi = self.dragonnet(x_table)

        if x_img is not None:
            out_conv3d = self.conv3d(x_img)
            out_total = self.classify_total(out_drag, out_conv3d)
            return out_drag, out_total
        else:
            return out_drag, None

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    from utils.config import Config
    cfg = Config.fromfile('/home/hrlee/PycharmProjects/lesion/config/lesion_drag_cls.py')
    setattr(cfg, 'netmode', 'default')
    net = dragcls(cfg, 32, 64)

    from datasets.lesion import lesionDataset
    from torch.utils.data.dataloader import DataLoader
    object = lesionDataset(cfg, None, mode='dragon', task='train')
    train_loader = DataLoader(
        object, batch_size=cfg.batchsize, shuffle=True,
        num_workers=0, collate_fn=lesionDataset.collate_fn,
        pin_memory=True
    )

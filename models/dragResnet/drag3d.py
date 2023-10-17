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
    def __init__(self, cfg, num_patches):
        super().__init__()
        self.input_channel = cfg.inchannel * num_patches
        self.inplanes = 32
        fc_in = int(self.inplanes * num_patches)
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

        # ==== For DragonNet ====
        # Representation
        self.rep_blocks = nn.Sequential(
            nn.Linear(cfg.model_dragon.input_dim, cfg.model_dragon.rep_dim),
            nn.ELU(),
            nn.Linear(cfg.model_dragon.rep_dim, cfg.model_dragon.rep_dim),
            nn.ELU(),
            nn.Linear(cfg.model_dragon.rep_dim, cfg.model_dragon.rep_dim)
        )

        self.y0_hidden = nn.Sequential(
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.model_dragon.rep_dim, cfg.model_dragon.hypo_dim),  # Hypothesis
            nn.ELU(),
            nn.Linear(cfg.model_dragon.hypo_dim, cfg.model_dragon.sec_dim),  # second
            nn.ELU(),
            nn.Linear(cfg.model_dragon.sec_dim, cfg.model_dragon.thir_dim),  # third
            nn.ELU(),
        )
        self.y1_hidden = nn.Sequential(
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.model_dragon.rep_dim, cfg.model_dragon.hypo_dim),  # Hypothesis
            nn.ELU(),
            nn.Linear(cfg.model_dragon.hypo_dim, cfg.model_dragon.sec_dim),  # second
            nn.ELU(),
            nn.Linear(cfg.model_dragon.sec_dim, cfg.model_dragon.thir_dim),  # third
            nn.ELU(),
        )

        self.t_hidden = nn.Sequential(
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.model_dragon.rep_dim, cfg.model_dragon.thir_dim),
            nn.ELU(),
        )

        self.y0_pred = nn.Linear(2 * cfg.model_dragon.thir_dim, 2)  # pred
        self.y1_pred = nn.Linear(2 * cfg.model_dragon.thir_dim, 2)  # pred

        # Propensity prediction
        self.propen_pred = nn.Sequential(
            nn.Linear(cfg.model_dragon.thir_dim, 1)
        )

        self.epsilons = EpsilonLayer()

        # ==== For classification ====
        self.flatten = nn.Flatten()
        self.fc_meta = nn.Sequential(
            nn.Linear(fc_in + cfg.model_dragon.rep_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 2)
        )

    def _forward_classify(self, phi, out_3d):
        return self.fc_meta(torch.cat([out_3d, phi], dim=1))

    def _forward_conv3d(self, x):
        out = self.conv1(x)
        out = self.avg_pool(out)
        out = self.flatten(out)
        return out

    def _forward_dragon(self, x):
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

    def forward(self, imgs, x_drag):
        out_3d = self._forward_conv3d(imgs)
        out_drag, phi = self._forward_dragon(x_drag)
        out_total = self._forward_classify(phi, out_3d)
        return out_drag, out_total

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

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

        init_weights_(self.rep_blocks)
        init_weights_([self.y0_hidden])
        init_weights_([self.y1_hidden])
        init_weights_([self.y0_pred])
        init_weights_([self.y1_pred])
        init_weights_([self.propen_pred])

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

        return torch.cat([pred_0, pred_1, pred_propen, epsilons, phi], dim=1)

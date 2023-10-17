import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import cv2
from models.resnet.layers import Hook


class conv_sim(nn.Module):
    def __init__(self, cfg):
        super(conv_sim, self).__init__()
        self.cfg = cfg
        #
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.cfg.inchannel, cfg.model_tiscls.num_tis_classes-1, kernel_size=11, stride=5, padding=0, bias=False),
            nn.BatchNorm2d(cfg.model_tiscls.num_tis_classes-1),
            nn.SiLU(),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        #
        self.hookF = [Hook(layer[1]) for layer in list(self._modules.items())]
        self.hookB = [Hook(layer[1], backward=True) for layer in list(self._modules.items())]
        #
        self.layer_gradcam = 'conv1'
        self.am_factor = 0.5

    def _forward_cl(self, x):
        out = self.conv1(x)
        out = self.avg_pool(out)
        out = self.flatten(out)
        return out

    def forward(self, input, label, gain=False):
        # Classification
        out_cl = self._forward_cl(input)
        if gain:
            # Get Attention Map (Ac)
            self.gcam, Ac = self._forward_attention_map(out_cl, label, input.shape)
            # Equation (5)
            with torch.no_grad():
                # make masked input
                self.maskedimg = self._make_masked_input(input, Ac)
                #
                output_am = self._forward_cl(self.maskedimg)
                output_am_score_softmax = F.softmax(output_am, dim=1)
                # Get loss_am
                loss_am = 0
                loss_cnt = 0
                for batch_idx in range(label.shape[0]):
                    if label[batch_idx, 1] == 1:
                        loss_cnt += 1
                        loss_am += output_am_score_softmax[batch_idx, 1]
                self.loss_am = loss_am / loss_cnt * self.am_factor if loss_cnt != 0 else 0
        return out_cl

    def _forward_attention_map(self, out_cl, label, input_size):
        layer_index = np.argmax(
            np.array([name == self.layer_gradcam for name in self._modules.keys()], dtype=np.int_))
        #
        score = out_cl * label
        score.mean().backward(retain_graph=True)
        # Equation (1)
        gradient = self.hookB[layer_index].output[0]
        w_c = gradient.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True) / (gradient.shape[2] * gradient.shape[3])
        # Equation (2)
        feature_maps = self.hookF[layer_index].output
        Ac = F.relu((feature_maps.detach() * w_c.detach()).sum(dim=1).unsqueeze(dim=1))
        Ac = F.interpolate(Ac, size=(input_size[2], input_size[3]))
        gcam = Ac
        # Make mask_image
        scaled_Ac = (Ac - Ac.min()) / (Ac.max() - Ac.min())
        #
        return gcam, scaled_Ac

    @staticmethod
    def _make_masked_input(input, Ac):
        mask = torch.zeros_like(Ac)
        mask[Ac > 0.5] = 1
        mask[Ac <= 0.5] = 0
        maskedImg = input * (1 - mask)
        return maskedImg

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplanes, out_channels, stride))
            self.inplanes = out_channels * block.expansion
        return nn.Sequential(*layers)

    def get_gcam(self):
        return self.gcam

    @staticmethod
    def predict(out):
        y = F.softmax(out, dim=1)
        return y

    def get_loss_am(self):
        return self.loss_am

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    print('Debug StackedAE')
    model = resnet('resnet152')
    model.init_weights()
    x = torch.rand(1, 3, 52, 52)
    model.set_input(x)
    model.forward()
    output_1 = model.get_outputs()
    output = model.get_output()
    print(output.shape)
    print('Debug StackedAE')

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import cv2
from models.resnet.layers import Hook


class basicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(basicBlock, self).__init__()
        expanded_out = out_channels * basicBlock.expansion
        #
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, expanded_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(expanded_out)
        )
        # projection mapping using 1x1conv
        if stride != 1 or in_channels != expanded_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, expanded_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(expanded_out)
            )
        else:
            self.shortcut = nn.Sequential()
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.act(x)
        return x


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleNeck, self).__init__()
        expanded_out = out_channels * BottleNeck.expansion
        #
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, expanded_out, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(expanded_out),
        )

        if stride != 1 or in_channels != expanded_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, expanded_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(expanded_out)
            )
        else:
            self.shortcut = nn.Sequential()
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.act(x)
        return x


class resnet(nn.Module):
    def __init__(self, cfg, net_name='resnet6', num_classes=5):
        super(resnet, self).__init__()
        self.input_channel = cfg.inchannel
        self.inplanes = 64
        self.cfg = cfg
        #
        ch_per_block = [self.inplanes, 128, 256, 512]
        if net_name == 'resnet4':
            block = basicBlock
            num_block = [1, 0, 0, 0]
            fc_in = ch_per_block[0] * block.expansion
            self.layer_gradcam = 'conv2_x'
        elif net_name == 'resnet6':
            block = basicBlock
            num_block = [1, 1, 0, 0]
            fc_in = ch_per_block[1] * block.expansion
            self.layer_gradcam = 'conv3_x'
        elif net_name == 'resnet10':
            block = basicBlock
            num_block = [1, 1, 1, 1]
            fc_in = ch_per_block[3] * block.expansion
            self.layer_gradcam = 'conv3_x'
        elif net_name == 'resnet18':
            block = basicBlock
            num_block = [2, 2, 2, 2]
            fc_in = ch_per_block[3] * block.expansion
            self.layer_gradcam = 'conv5_x'
        elif net_name == 'resnet24':
            block = basicBlock
            num_block = [2, 2, 3, 3]
            fc_in = ch_per_block[3] * block.expansion
            self.layer_gradcam = 'conv5_x'
        elif net_name == 'resnet34':
            block = basicBlock
            num_block = [3, 4, 6, 3]
            fc_in = ch_per_block[3] * block.expansion
            self.layer_gradcam = 'conv5_x'
        elif net_name == 'resnet50':
            block = BottleNeck
            num_block = [3, 4, 6, 3]
            fc_in = ch_per_block[3] * block.expansion
            self.layer_gradcam = 'conv5_x'
        else:
            raise NotImplementedError('Invalide net_name!!!:{}'.format(net_name))
        #
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_channel, int(self.inplanes/2), kernel_size=7, stride=3, padding=0, bias=False),
            nn.BatchNorm2d(int(self.inplanes/2)),
            nn.SiLU(),
            nn.Conv2d(int(self.inplanes/2), self.inplanes, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.SiLU(),
            nn.AdaptiveMaxPool2d((512, 512)),
        )
        pool_layer_avg = nn.AvgPool2d(2, 2)
        pool_layer_max = nn.MaxPool2d(2, 2)
        self.conv2_x = nn.Sequential(
            self._make_layer(block, self.inplanes, num_block[0], 1) if num_block[0] != 0 else nn.Sequential(),
            pool_layer_avg
        )
        self.conv3_x = nn.Sequential(
            self._make_layer(block, 128, num_block[1], 1) if num_block[1] != 0 else nn.Sequential(),
            pool_layer_max
        )
        self.conv4_x = nn.Sequential(
            self._make_layer(block, 256, num_block[2], 1) if num_block[2] != 0 else nn.Sequential(),
        )
        self.conv5_x = nn.Sequential(
            self._make_layer(block, 512, num_block[3], 1) if num_block[3] != 0 else nn.Sequential(),
        )

        self.avg_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc_tis = nn.Linear(fc_in, num_classes)
        self.fc_meta = nn.Linear(fc_in, 2)
        #
        self.hookF = [Hook(layer[1]) for layer in list(self._modules.items())]
        self.hookB = [Hook(layer[1], backward=True) for layer in list(self._modules.items())]
        #
        self.am_factor = 1.0
        self.am_neg_factor = 2.0
        self.mask_thres = 0.5

    def _forward_cl(self, x):
        out = self.conv1(x)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        out = self.avg_pool(out)
        out = self.flatten(out)
        out_tis = self.fc_tis(out)
        out_meta = self.fc_meta(out)
        return out_tis, out_meta

    def forward(self, input, onehot_tis, mask_label, onehot_t, gain=False):
        # Classification
        out_tis, out_meta = self._forward_cl(input)
        #
        self.loss_am = 0
        self.gcam_cls = None
        self.gcam_tis = None
        # gain loss for metastasis
        if gain:
            loss_am = 0
            self.gcam_cls = self._forward_attention_map(out_meta, onehot_t, input.shape)
            with torch.no_grad():
                maskedimg, _ = self._make_masked_input(input, self.gcam_cls)
                for batch_idx in range(input.shape[0]):
                    temp_label_t = onehot_t[batch_idx]
                    if temp_label_t[1] == 1:
                        _, output_am = self._forward_cl(maskedimg[1][batch_idx].unsqueeze(dim=0))
                        output_am_score_softmax = F.softmax(output_am, dim=1)
                        #
                        loss_am += output_am_score_softmax[0, 1]
            self.loss_am += loss_am / onehot_t.shape[0] * self.am_factor
        if (gain and
                (sum(mask_label.mean([1, 2, 3]) == -1).item() != input.shape[0])):
            # Get Attention Map (Ac)
            self.gcam_tis = self._forward_attention_map(out_tis, onehot_tis, input.shape)
            with torch.no_grad():
                # Equation (5)
                # make masked input
                maskedimg, mask_map = self._make_masked_input(input, self.gcam_tis)
                #
                loss_am = 0
                loss_map = 0
                loss_cnt = 0
                for batch_idx in range(onehot_tis.shape[0]):
                    temp_label = onehot_tis[batch_idx]
                    if temp_label.cpu().numpy().mean() != -1:
                        loss_cnt += 1
                        for cls_idx in range(len(temp_label)):
                            if temp_label[cls_idx] == 1:
                                output_am, _ = self._forward_cl(maskedimg[cls_idx][batch_idx].unsqueeze(dim=0))
                                output_am_score_softmax = F.softmax(output_am, dim=1)
                                #
                                for kk in range(len(temp_label)):
                                    if kk == cls_idx:
                                        loss_am += output_am_score_softmax[0, kk]
                                        loss_map += F.l1_loss(mask_label[batch_idx][kk].to(temp_label.device),
                                                              mask_map[kk][batch_idx][0], reduction='sum')
                                    else:
                                        if temp_label[kk] == 1:
                                            loss_am += (1 - output_am_score_softmax[0, kk]) * self.am_neg_factor
            self.loss_am = loss_am / loss_cnt * self.am_factor + loss_map * 2
        return out_tis, out_meta

    def _forward_attention_map(self, out_cl, label, input_size):
        layer_index = np.argmax(
            np.array([name == self.layer_gradcam for name in self._modules.keys()], dtype=np.int_))
        #
        gcam_per_class = []
        for cls_id in range(0, label.shape[1]):
            one_hot_cls_id = torch.zeros([1, label.shape[1]]).to(label[0].device)
            one_hot_cls_id[0, cls_id] = 1
            score = out_cl * (label * one_hot_cls_id)
            score.backward(torch.ones_like(score), retain_graph=True)
            # Equation (1)
            gradient = self.hookB[layer_index].output[0]
            w_c = gradient.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True) / (gradient.shape[2] * gradient.shape[3])
            # Equation (2)
            feature_maps = self.hookF[layer_index].output
            gcam = F.relu((feature_maps.detach() * w_c.detach()).sum(dim=1).unsqueeze(dim=1))
            gcam = F.interpolate(gcam, size=(input_size[2], input_size[3]))
            #
            gcam_per_class.append(gcam)
        return gcam_per_class

    def _make_masked_input(self, input, Ac):
        maskedImg = []
        mask_map = []
        for idx in range(len(Ac)):
            scaled_Ac = (Ac[idx] - Ac[idx].min()) / (Ac[idx].max() - Ac[idx].min() + 1e-6)
            mask = torch.zeros_like(scaled_Ac)
            mask[scaled_Ac > self.mask_thres] = 1
            mask[scaled_Ac <= self.mask_thres] = 0
            masked = input * (1 - mask)
            #
            maskedImg.append(masked)
            mask_map.append(mask)
        return maskedImg, mask_map

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplanes, out_channels, stride))
            self.inplanes = out_channels * block.expansion
        return nn.Sequential(*layers)

    def get_gcam(self):
        return self.gcam_tis, self.gcam_cls

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

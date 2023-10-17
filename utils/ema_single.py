#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# The code is based on
# https://github.com/ultralytics/yolov5/blob/master/utils/torch_utils.py
import math
from copy import deepcopy
import torch
import torch.nn as nn


class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # See
        # https://timm.fast.ai/training_modelEMA#Training-with-EMA
        #
        # self.ema is like a container that maintains moving averaged trained weights
        self.ema = deepcopy(model).eval()  # FP32 EMA
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for param in self.ema.parameters():
            param.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            self.updates += 1
            decay = self.decay(self.updates)

            # Maintain moving averages of the trained model parameters
            state_dict = model.state_dict()  # trained model state_dict
            for k, item in self.ema.state_dict().items():
                if item.dtype.is_floating_point:
                    #       saved parameters         updated parameters during training
                    #        ______|______    _______________|________________
                    # item = decay * item + (1 - decay) * state_dict[k].detach()
                    # The lower two code lines is equal to the upper description.
                    item *= decay
                    item += (1 - decay) * state_dict[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        copy_attr(self.ema, model, include, exclude)


def copy_attr(a, b, include=(), exclude=()):
    """Copy attributes from one instance and set them to another instance."""
    for k, item in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, item)

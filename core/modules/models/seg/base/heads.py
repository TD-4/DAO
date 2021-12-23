import torch.nn as nn
from .modules import Flatten, Activation


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)


class ClassificationHead(nn.Sequential):

    def __init__(self, in_channels, classes, pooling="avg", dropout=0.2, activation=None):
        if pooling not in ("max", "avg"):
            raise ValueError("Pooling should be one of ('max', 'avg'), got {}.".format(pooling))
        conv2d_1 = nn.Conv2d(16, 8, kernel_size=7, padding=3, stride=4)
        bn1 = nn.BatchNorm2d(8)
        conv2d_2 = nn.Conv2d(8, 4, kernel_size=7, padding=3, stride=4)
        bn2 = nn.BatchNorm2d(4)
        conv2d_3 = nn.Conv2d(4, classes, kernel_size=7, padding=3, stride=4)
        bn3 = nn.BatchNorm2d(classes)
        relu = nn.ReLU6(inplace=True)
        pool = nn.AdaptiveAvgPool2d(1) if pooling == 'avg' else nn.AdaptiveMaxPool2d(1)
        flatten = Flatten()
        # dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        # linear = nn.Linear(in_channels, classes, bias=True)
        # activation = Activation(activation)
        super().__init__(conv2d_1, bn1, relu, conv2d_2, bn2, relu, conv2d_3, bn3, pool, flatten)

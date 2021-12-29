# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Copy From: https://paperswithcode.com/paper/blindly-assess-image-quality-in-the-wild
# @Paper: https://paperswithcode.com/paper/blindly-assess-image-quality-in-the-wild

import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init

from core.modules.register import Registers
import core.modules.models.backbone


@Registers.iqa_models.register
class HyperNet(nn.Module):
    """
    Hyper network for learning perceptual rules.

    models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
        lda_out_channels, 16
        hyper_in_channels, 112
        target_in_size, 224
        target_fc1_size, 112
        target_fc2_size, 56
        target_fc3_size, 28
        target_fc4_size, 14
        feature_size, 7

    Args:
        lda_out_channels: local distortion aware module output size.
        hyper_in_channels: input feature channels for hyper network.
        target_in_size: input vector size for target network.
        target_fc(i)_size: fully connection layer size of target network.
        feature_size: input feature map width/height for hyper network.

    Note:
        For size match, input args must satisfy: 'target_fc(i)_size * target_fc(i+1)_size' is divisible by 'feature_size ^ 2'.

    """
    def __init__(self,
                 backbone_kwargs=None,
                 lda_out_channels=16,
                 hyper_in_channels=112,
                 target_in_size=224,
                 target_fc1_size=112,
                 target_fc2_size=56,
                 target_fc3_size=28,
                 target_fc4_size=14,
                 feature_size=7,
                 num_classes=10,
                 encoder_channels=[30, 128, 256, 512, 1024, 2048],
                 ):
        super(HyperNet, self).__init__()    # models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
        self.feature_size = feature_size
        self.num_classes = num_classes


        # part1 backbone
        self.backbone = Registers.backbones.get("TIMM")(backbone_kwargs)
        # local distortion aware module
        self.lda1_pool = nn.Sequential(
            nn.Conv2d(encoder_channels[2], 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(7, stride=7),
        )
        self.lda1_fc = nn.Linear(16 * 64, lda_out_channels)

        self.lda2_pool = nn.Sequential(
            nn.Conv2d(encoder_channels[3], 32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(7, stride=7),
        )
        self.lda2_fc = nn.Linear(32 * 16, lda_out_channels * 2)

        self.lda3_pool = nn.Sequential(
            nn.Conv2d(encoder_channels[4], 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(7, stride=7),
        )
        self.lda3_fc = nn.Linear(64 * 4, lda_out_channels * 4)

        self.lda4_pool = nn.AvgPool2d(7, stride=7)
        self.lda4_fc = nn.Linear(encoder_channels[5], target_in_size - lda_out_channels * 7)


        # part2 content understanding hyper network
        self.hyperInChn = hyper_in_channels
        self.target_in_size = target_in_size
        self.f1 = target_fc1_size
        self.f2 = target_fc2_size
        self.f3 = target_fc3_size
        self.f4 = target_fc4_size
        self.feature_size = feature_size

        # Conv layers for resnet output features
        self.conv1 = nn.Sequential(
            nn.Conv2d(encoder_channels[-1], 1024, 1, padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, 1, padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, self.hyperInChn, 1, padding=(0, 0)),
            nn.ReLU(inplace=True)
        )
        # Hyper network part, conv for generating target fc weights, fc for generating target fc biases
        self.fc1w_conv = nn.Conv2d(self.hyperInChn, int(self.target_in_size * self.f1 / feature_size ** 2), 3,
                                   padding=(1, 1))
        self.fc1b_fc = nn.Linear(self.hyperInChn, self.f1)

        self.fc2w_conv = nn.Conv2d(self.hyperInChn, int(self.f1 * self.f2 / feature_size ** 2), 3, padding=(1, 1))
        self.fc2b_fc = nn.Linear(self.hyperInChn, self.f2)

        self.fc3w_conv = nn.Conv2d(self.hyperInChn, int(self.f2 * self.f3 / feature_size ** 2), 3, padding=(1, 1))
        self.fc3b_fc = nn.Linear(self.hyperInChn, self.f3)

        self.fc4w_conv = nn.Conv2d(self.hyperInChn, int(self.f3 * self.f4 / feature_size ** 2), 3, padding=(1, 1))
        self.fc4b_fc = nn.Linear(self.hyperInChn, self.f4)

        self.fc5w_fc = nn.Linear(self.hyperInChn, self.f4 * 10)
        self.fc5b_fc = nn.Linear(self.hyperInChn, 10)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))


        # part3 quality prediction target network
        self.target_net = TargetNet()

    def forward(self, x):
        feature_size = self.feature_size

        backbone_r = self.backbone(x)
        # pass part1
        lda_1 = self.lda1_fc(self.lda1_pool(backbone_r[0]).view(x.size(0), -1))
        lda_2 = self.lda2_fc(self.lda2_pool(backbone_r[1]).view(x.size(0), -1))
        lda_3 = self.lda3_fc(self.lda3_pool(backbone_r[2]).view(x.size(0), -1))
        lda_4 = self.lda4_fc(self.lda4_pool(backbone_r[3]).view(x.size(0), -1))

        target_in_vec = torch.cat((lda_1, lda_2, lda_3, lda_4), 1)
        hyper_in_feat = backbone_r[3]

        # input vector for target net
        target_in_vec = target_in_vec.view(-1, self.target_in_size, 1, 1)

        # input features for hyper net
        hyper_in_feat = self.conv1(hyper_in_feat).view(-1, self.hyperInChn, feature_size, feature_size)

        # generating target net weights & biases
        target_fc1w = self.fc1w_conv(hyper_in_feat).view(-1, self.f1, self.target_in_size, 1, 1)
        target_fc1b = self.fc1b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f1)

        target_fc2w = self.fc2w_conv(hyper_in_feat).view(-1, self.f2, self.f1, 1, 1)
        target_fc2b = self.fc2b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f2)

        target_fc3w = self.fc3w_conv(hyper_in_feat).view(-1, self.f3, self.f2, 1, 1)
        target_fc3b = self.fc3b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f3)

        target_fc4w = self.fc4w_conv(hyper_in_feat).view(-1, self.f4, self.f3, 1, 1)
        target_fc4b = self.fc4b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f4)

        # target_fc5w = self.fc5w_fc(self.pool(hyper_in_feat).squeeze()).view(-1, 1, self.f4, 1, 1)
        # target_fc5b = self.fc5b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, 1)
        target_fc5w = self.fc5w_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.num_classes, self.f4, 1, 1)
        target_fc5b = self.fc5b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.num_classes)

        out = {}
        out['target_in_vec'] = target_in_vec
        out['target_fc1w'] = target_fc1w
        out['target_fc1b'] = target_fc1b
        out['target_fc2w'] = target_fc2w
        out['target_fc2b'] = target_fc2b
        out['target_fc3w'] = target_fc3w
        out['target_fc3b'] = target_fc3b
        out['target_fc4w'] = target_fc4w
        out['target_fc4b'] = target_fc4b
        out['target_fc5w'] = target_fc5w
        out['target_fc5b'] = target_fc5b

        self.target_net.init_(out)
        res = self.target_net.forward(target_in_vec)
        return res


class TargetNet(nn.Module):
    """
    Target network for quality prediction.
    """
    def __init__(self):
        super(TargetNet, self).__init__()

    def init_(self, paras):
        super(TargetNet, self).__init__()
        self.l1 = nn.Sequential(
            TargetFC(paras['target_fc1w'], paras['target_fc1b']),
            nn.Sigmoid(),
        )
        self.l2 = nn.Sequential(
            TargetFC(paras['target_fc2w'], paras['target_fc2b']),
            nn.Sigmoid(),
        )

        self.l3 = nn.Sequential(
            TargetFC(paras['target_fc3w'], paras['target_fc3b']),
            nn.Sigmoid(),
        )

        self.l4 = nn.Sequential(
            TargetFC(paras['target_fc4w'], paras['target_fc4b']),
            nn.Sigmoid(),
            TargetFC(paras['target_fc5w'], paras['target_fc5b']),
        )

    def forward(self, x):
        q = self.l1(x)
        # q = F.dropout(q)
        q = self.l2(q)
        q = self.l3(q)
        q = self.l4(q).squeeze()
        return q


class TargetFC(nn.Module):
    """
    Fully connection operations for target net

    Note:
        Weights & biases are different for different images in a batch,
        thus here we use group convolution for calculating images in a batch with individual weights & biases.
    """
    def __init__(self, weight, bias):
        super(TargetFC, self).__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, input_):

        input_re = input_.view(-1, input_.shape[0] * input_.shape[1], input_.shape[2], input_.shape[3])
        weight_re = self.weight.view(self.weight.shape[0] * self.weight.shape[1], self.weight.shape[2], self.weight.shape[3], self.weight.shape[4])
        bias_re = self.bias.view(self.bias.shape[0] * self.bias.shape[1])
        out = F.conv2d(input=input_re, weight=weight_re, bias=bias_re, groups=self.weight.shape[0])

        return out.view(input_.shape[0], self.weight.shape[1], input_.shape[2], input_.shape[3])


if __name__ == "__main__":
    import torch
    from dotmap import DotMap

    model_kwargs = DotMap({
        "type": "hpyerIQA",
        "summary_size": [30, 224, 224],
        "backbone": {
            "kwargs": {
                "model_name": "resnet50",
                "pretrained": True,
                "checkpoint_path": "",
                "exportable": True,
                "in_chans": 30,
                "features_only": True,
                "out_indices": [1, 2, 3, 4]
            }
        },
        "kwargs": {
            "num_classes": 10,
            "encoder_channels": [30, 128, 256, 512, 1024, 2048],
            "lda_out_channels": 128,
            "hyper_in_channels": 896,
            "target_in_size": 1792,
            "target_fc1_size": 896,
            "target_fc2_size": 448,
            "target_fc3_size": 224,
            "target_fc4_size": 112,
            "feature_size": 7,
        }
    })

    x = torch.rand(8, 30, 224, 224)
    model = HyperNet(model_kwargs.backbone, **model_kwargs.kwargs)

    output = model(x)
    print(output.shape)
    print(model)

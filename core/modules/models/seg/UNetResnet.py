#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @auther:FelixFu
# @Date: 2021.10.1
# @github:https://github.com/felixfu520
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.modules.utils import initialize_weights, set_trainable
from core.modules.register import Registers


# Unet with a resnet backbone
@Registers.seg_models.register
class UNetResnet(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, backbone=None, pretrained=True,
                 freeze_bn=False, freeze_backbone=False, **_):
        super(UNetResnet, self).__init__()
        model = Registers.backbones.get(backbone)(pretrained=pretrained, in_channels=in_channels, feature_only=True)

        self.initial = list(model.children())[:4]
        # if in_channels != 3:
        #     self.initial[0] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.initial = nn.Sequential(*self.initial)

        # encoder
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        # decoder
        self.conv1 = nn.Conv2d(2048, 192, kernel_size=3, stride=1, padding=1)
        self.upconv1 = nn.ConvTranspose2d(192, 128, 4, 2, 1, bias=False)

        self.conv2 = nn.Conv2d(1152, 128, kernel_size=3, stride=1, padding=1)
        self.upconv2 = nn.ConvTranspose2d(128, 96, 4, 2, 1, bias=False)

        self.conv3 = nn.Conv2d(608, 96, kernel_size=3, stride=1, padding=1)
        self.upconv3 = nn.ConvTranspose2d(96, 64, 4, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(320, 64, kernel_size=3, stride=1, padding=1)
        self.upconv4 = nn.ConvTranspose2d(64, 48, 4, 2, 1, bias=False)

        self.conv5 = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1)
        self.upconv5 = nn.ConvTranspose2d(48, 32, 4, 2, 1, bias=False)

        self.conv6 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(32, num_classes, kernel_size=1, bias=False)

        initialize_weights(self)

        if freeze_bn:
            self.freeze_bn()
        if freeze_backbone:
            set_trainable([self.initial, self.layer1, self.layer2, self.layer3, self.layer4], False)

    def forward(self, x):
        """

                                                 (num_class,224,224)
                                                 (32,224,224)
                    (3,224,224)                  (48,112,112)
        layer1      (256,56,56)                  (320,56,56)    = (64,56,56) + (256,56,56)
        layer2      (512,28,28)                  (608, 28, 28)  = (96,28,28) + (512,28,28)
        layer3      (1024,14,14)                 (1152, 14, 14) = (128,14,14) + (1024,14,14)
        layer4      (2048,7,7)          ->       (128,7,7)

        """
        H, W = x.size(2), x.size(3)
        x1 = self.layer1(self.initial(x))  # down 4x
        x2 = self.layer2(x1)  # down 8x
        x3 = self.layer3(x2)  # down 16x
        x4 = self.layer4(x3)  # down 32x

        x = self.upconv1(self.conv1(x4))
        x = F.interpolate(x, size=(x3.size(2), x3.size(3)), mode="bilinear", align_corners=True)
        x = torch.cat([x, x3], dim=1)

        x = self.upconv2(self.conv2(x))
        x = F.interpolate(x, size=(x2.size(2), x2.size(3)), mode="bilinear", align_corners=True)
        x = torch.cat([x, x2], dim=1)

        x = self.upconv3(self.conv3(x))
        x = F.interpolate(x, size=(x1.size(2), x1.size(3)), mode="bilinear", align_corners=True)
        x = torch.cat([x, x1], dim=1)

        x = self.upconv4(self.conv4(x))

        x = self.upconv5(self.conv5(x))

        # if the input is not divisible by the output stride
        if x.size(2) != H or x.size(3) != W:
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True)

        x = self.conv7(self.conv6(x))
        return x

    def get_backbone_params(self):
        return chain(self.initial.parameters(), self.layer1.parameters(), self.layer2.parameters(),
                     self.layer3.parameters(), self.layer4.parameters())

    def get_decoder_params(self):
        return chain(self.conv1.parameters(), self.upconv1.parameters(), self.conv2.parameters(),
                     self.upconv2.parameters(),
                     self.conv3.parameters(), self.upconv3.parameters(), self.conv4.parameters(),
                     self.upconv4.parameters(),
                     self.conv5.parameters(), self.upconv5.parameters(), self.conv6.parameters(),
                     self.conv7.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()

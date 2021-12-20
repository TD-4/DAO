
# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.12.17
# @GitHub:https://github.com/felixfu520
# @Copy From: https://github.com/rvorias/ind_knn_ad/blob/master/indad/models.py

import torch

from core.modules.models.backbone import timm
from core.modules.register import Registers

from .utils import GaussianBlur


@Registers.anomaly_models.register
class PaDiM(torch.nn.Module):
    def __init__(self, backbone=None, out_indices=(2, 3), pool_last=False, pretrained=True, features_only=True,
                 checkpoint_path='', in_channels=3, f_coreset=0.01, coreset_eps=0.90, image_size=224, **kwargs):
        super().__init__()
        self.feature_extractor = timm.create_model(
            backbone,
            out_indices=out_indices,
            features_only=features_only,
            pretrained=pretrained,
            in_chans=in_channels,
            checkpoint_path=checkpoint_path,
            **kwargs
        )
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()

        self.pool = torch.nn.AdaptiveAvgPool2d(1) if pool_last else None
        self.backbone_name = backbone  # for results metadata
        self.out_indices = out_indices

        self.f_coreset = f_coreset
        self.coreset_eps = coreset_eps
        self.image_size = image_size
        self.average = torch.nn.AvgPool2d(3, stride=1)
        self.blur = GaussianBlur(4)
        self.n_reweight = 3

        self.patch_lib = []
        self.resize = None

    def forward(self, x):
        with torch.no_grad():
            feature_maps = self.feature_extractor(x)
        # feature_maps = [fmap.to("cpu") for fmap in feature_maps]
        if self.pool:
            # spit into fmaps and z
            return feature_maps[:-1], self.pool(feature_maps[-1])
        else:
            return feature_maps
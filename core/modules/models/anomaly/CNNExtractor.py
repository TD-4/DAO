# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Copy From:

import torch
import timm


class CNNExtractor(torch.nn.Module):
    def __init__(
            self,
            backbone_name,
            out_indices=None,
            in_chans=3,
            pretrained=True,
            features_only=True,
            device=None,
            **kwargs
    ):
        super().__init__()
        self.feature_extractor = timm.create_model(
            backbone_name,
            out_indices=out_indices,
            features_only=features_only,
            pretrained=pretrained,
            in_chans=in_chans,
            **kwargs
        )
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()

        self.backbone_name = backbone_name  # for results metadata
        self.out_indices = out_indices

        self.device = device
        self.feature_extractor = self.feature_extractor.to(self.device)

    def __call__(self, x):
        with torch.no_grad():
            feature_maps = self.feature_extractor(x.to(self.device))
        feature_maps = [fmap.to("cpu") for fmap in feature_maps]
        return feature_maps


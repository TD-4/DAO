# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Copy From: https://github.com/rvorias/ind_knn_ad/blob/master/indad/models.py

from typing import Tuple
from tqdm import tqdm

import torch
from torch import tensor
from torch.utils.data import DataLoader
import timm

import numpy as np
from sklearn.metrics import roc_auc_score

from .utils import get_tqdm_params


class KNNExtractor(torch.nn.Module):
    def __init__(self, backbone_name, out_indices=None, in_chans=3, pretrained=True,
                 pool_last=False, features_only=True, device=None):
        super().__init__()

        self.feature_extractor = timm.create_model(
            backbone_name,
            out_indices=out_indices,
            features_only=features_only,
            pretrained=pretrained,
            in_chans=in_chans
        )
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()

        self.pool = torch.nn.AdaptiveAvgPool2d(1) if pool_last else None
        self.backbone_name = backbone_name  # for results metadata
        self.out_indices = out_indices

        self.device = device
        self.feature_extractor = self.feature_extractor.to(self.device)

    def __call__(self, x: tensor):
        with torch.no_grad():
            feature_maps = self.feature_extractor(x.to(self.device))
        feature_maps = [fmap.to("cpu") for fmap in feature_maps]
        if self.pool:
            # spit into fmaps and z
            return feature_maps[:-1], self.pool(feature_maps[-1])
        else:
            return feature_maps

    def fit(self, _: DataLoader):
        raise NotImplementedError

    def predict(self, _: tensor):
        raise NotImplementedError

    def evaluate(self, test_dl: DataLoader) -> Tuple[float, float]:
        """Calls predict step for each test sample."""
        image_preds = []
        image_labels = []
        pixel_preds = []
        pixel_labels = []

        for image, mask, label, image_path in tqdm(test_dl, **get_tqdm_params()):
            z_score, fmap = self.predict(image)

            image_preds.append(z_score.numpy())
            image_labels.append(label)

            pixel_preds.extend(fmap.flatten().numpy())
            pixel_labels.extend(mask.flatten().numpy())

        image_preds = np.stack(image_preds)

        # image_rocauc = roc_auc_score(image_labels, image_preds)
        image_rocauc = roc_auc_score([i.numpy().item() for i in image_labels], image_preds)
        pixel_rocauc = roc_auc_score(pixel_labels, pixel_preds)

        return image_rocauc, pixel_rocauc

    def get_parameters(self, extra_params: dict = None) -> dict:
        return {
            "backbone_name": self.backbone_name,
            "out_indices": self.out_indices,
            **extra_params,
        }




# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.12.17
# @GitHub:https://github.com/felixfu520
# @Copy From: https://github.com/rvorias/ind_knn_ad/blob/master/indad/models.py

from loguru import logger

import torch

from core.modules.register import Registers

from .utils import GaussianBlur, get_coreset_idx_randomp, get_tqdm_params
from .KNNExtractor import KNNExtractor


@Registers.anomaly_models.register
class PaDiM(KNNExtractor):
    def __init__(self, backbone, out_indices=(1, 2, 3), in_channels=3, pretrained=True,
                 pool_last=False, features_only=True,
                 d_reduced: int = 100, image_size=224, device=None,
                 **kwargs
                 ):
        super().__init__(
            backbone_name=backbone,
            out_indices=out_indices,
            in_chans=in_channels,
            pretrained=pretrained,
            pool_last=pool_last,
            features_only=features_only,
            device=device
        )
        self.image_size = image_size
        self.d_reduced = d_reduced  # your RAM will thank you
        self.epsilon = 0.04  # cov regularization
        self.patch_lib = []
        self.resize = None

    def fit(self, train_dl):
        for image, mask, label, image_path in train_dl:
            feature_maps = self(image)
            if self.resize is None:
                largest_fmap_size = feature_maps[0].shape[-2:]
                self.resize = torch.nn.AdaptiveAvgPool2d(largest_fmap_size)
            resized_maps = [self.resize(fmap) for fmap in feature_maps]
            self.patch_lib.append(torch.cat(resized_maps, 1))
        self.patch_lib = torch.cat(self.patch_lib, 0)

        # random projection
        if self.patch_lib.shape[1] > self.d_reduced:
            logger.info(f"   PaDiM: (randomly) reducing {self.patch_lib.shape[1]} dimensions to {self.d_reduced}.")
            self.r_indices = torch.randperm(self.patch_lib.shape[1])[:self.d_reduced]
            self.patch_lib_reduced = self.patch_lib[:, self.r_indices, ...]
        else:
            logger.info(
                "   PaDiM: d_reduced is higher than the actual number of dimensions, copying self.patch_lib ...")
            self.patch_lib_reduced = self.patch_lib

        # calcs
        self.means = torch.mean(self.patch_lib, dim=0, keepdim=True)
        self.means_reduced = self.means[:, self.r_indices, ...]
        x_ = self.patch_lib_reduced - self.means_reduced

        # cov calc
        self.E = torch.einsum(
            'abkl,bckl->ackl',
            x_.permute([1, 0, 2, 3]),  # transpose first two dims
            x_,
        ) * 1 / (self.patch_lib.shape[0] - 1)
        self.E += self.epsilon * torch.eye(self.d_reduced).unsqueeze(-1).unsqueeze(-1)
        self.E_inv = torch.linalg.inv(self.E.permute([2, 3, 0, 1])).permute([2, 3, 0, 1])

    def predict(self, sample):
        feature_maps = self(sample)
        resized_maps = [self.resize(fmap) for fmap in feature_maps]
        fmap = torch.cat(resized_maps, 1)

        # reduce
        x_ = fmap[:, self.r_indices, ...] - self.means_reduced

        left = torch.einsum('abkl,bckl->ackl', x_, self.E_inv)
        s_map = torch.sqrt(torch.einsum('abkl,abkl->akl', left, x_))
        scaled_s_map = torch.nn.functional.interpolate(
            s_map.unsqueeze(0), size=(self.image_size, self.image_size), mode='bilinear'
        )

        return torch.max(s_map), scaled_s_map[0, ...]

    def get_parameters(self):
        return super().get_parameters({
            "d_reduced": self.d_reduced,
            "epsilon": self.epsilon,
        })

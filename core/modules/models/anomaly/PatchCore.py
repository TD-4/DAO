
# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.12.17
# @GitHub:https://github.com/felixfu520
# @Copy From: https://github.com/rvorias/ind_knn_ad/blob/master/indad/models.py
# @Paper: https://paperswithcode.com/paper/towards-total-recall-in-industrial-anomaly

import torch

from core.modules.register import Registers

from .KNNExtractor import KNNExtractor
from .utils import GaussianBlur, get_coreset_idx_randomp, get_tqdm_params


@Registers.anomaly_models.register
class PatchCore(KNNExtractor):
    def __init__(self, backbone, out_indices=(2, 3),  in_channels=3, pretrained=True,
                 pool_last=False, features_only=True,
                 f_coreset=0.01, coreset_eps=0.90, image_size=224, device=None,
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
        self.f_coreset = f_coreset
        self.coreset_eps = coreset_eps
        self.image_size = image_size
        self.average = torch.nn.AvgPool2d(3, stride=1)
        self.blur = GaussianBlur(4)
        self.n_reweight = 3

        self.patch_lib = []
        self.resize = None

    def fit(self, train_dl):
        for image, mask, label, image_path in train_dl:
            feature_maps = self(image)

            if self.resize is None:
                largest_fmap_size = feature_maps[0].shape[-2:]
                self.resize = torch.nn.AdaptiveAvgPool2d(largest_fmap_size)
            resized_maps = [self.resize(self.average(fmap)) for fmap in feature_maps]
            patch = torch.cat(resized_maps, 1)  # torch.Size([32, 1536, 28, 28])
            patch = patch.reshape(patch.shape[1], -1).T  # torch.Size([25088, 1536])

            self.patch_lib.append(patch)

        self.patch_lib = torch.cat(self.patch_lib, 0)   # torch.Size([163856, 1536])

        if self.f_coreset < 1:
            self.coreset_idx = get_coreset_idx_randomp(
                self.patch_lib,
                n=int(self.f_coreset * self.patch_lib.shape[0]),
                eps=self.coreset_eps,
            )   # 1638 length
            self.patch_lib = self.patch_lib[self.coreset_idx]   # torch.Size([1638, 1536])

    def predict(self, sample):
        feature_maps = self(sample)
        resized_maps = [self.resize(self.average(fmap)) for fmap in feature_maps]
        patch = torch.cat(resized_maps, 1)
        patch = patch.reshape(patch.shape[1], -1).T  # torch.Size([25088, 1536])

        dist = torch.cdist(patch, self.patch_lib)   # torch.Size([25088, 1638])
        min_val, min_idx = torch.min(dist, dim=1)
        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch
        m_star = self.patch_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
        w_dist = torch.cdist(m_star, self.patch_lib)  # find knn to m_star pt.1
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2
        # equation 7 from the paper
        m_star_knn = torch.linalg.norm(m_test - self.patch_lib[nn_idx[0, 1:]], dim=1)
        # Softmax normalization trick as in transformers.
        # As the patch vectors grow larger, their norm might differ a lot.
        # exp(norm) can give infinities.
        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star

        # segmentation map
        s_map = min_val.view(-1, 1, *feature_maps[0].shape[-2:])
        s_map = torch.nn.functional.interpolate(
            s_map, size=(self.image_size, self.image_size), mode='bilinear'
        )
        s_map = self.blur(s_map)

        return s, s_map

    def get_parameters(self):
        return super().get_parameters({
            "f_coreset": self.f_coreset,
            "n_reweight": self.n_reweight,
        })

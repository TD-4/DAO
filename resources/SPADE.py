# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.12.14
# @GitHub:https://github.com/felixfu520
# @Copy From: https://github.com/rvorias/ind_knn_ad/blob/master/indad/models.py
# @Paper: https://paperswithcode.com/paper/sub-image-anomaly-detection-with-deep-pyramid


import torch

from core.modules.register import Registers

from .KNNExtractor import KNNExtractor
from .utils import GaussianBlur, get_coreset_idx_randomp, get_tqdm_params


@Registers.anomaly_models.register
class SPADE(KNNExtractor):
    def __init__(self, backbone, k=5, out_indices=(1, 2, 3, -1),  in_channels=3,
                 pretrained=True, pool_last=True, features_only=True,
                 image_size=224, device=None,
                 **kwargs
                 ):
        super().__init__(
            backbone_name=backbone,
            out_indices=out_indices,
            pool_last=pool_last,
            in_chans=in_channels,
            features_only=features_only,
            pretrained=pretrained,
            device=device
        )
        self.k = k
        self.image_size = image_size
        self.z_lib = []
        self.feature_maps = []
        self.threshold_z = None
        self.threshold_fmaps = None
        self.blur = GaussianBlur(4)

    def fit(self, train_dl):
        for image, mask, label, image_path in train_dl:
            feature_maps, z = self(image)

            # z vector
            self.z_lib.append(z)

            # feature maps
            if len(self.feature_maps) == 0:
                for fmap in feature_maps:
                    self.feature_maps.append([fmap])
            else:
                for idx, fmap in enumerate(feature_maps):
                    self.feature_maps[idx].append(fmap)

        self.z_lib = torch.vstack(self.z_lib)

        for idx, fmap in enumerate(self.feature_maps):
            self.feature_maps[idx] = torch.vstack(fmap)

    def predict(self, sample):
        feature_maps, z = self(sample)

        distances = torch.linalg.norm(self.z_lib - z, dim=1)
        values, indices = torch.topk(distances.squeeze(), self.k, largest=False)

        z_score = values.mean()

        # Build the feature gallery out of the k nearest neighbours.
        # The authors migh have concatenated all features maps first, then check the minimum norm per pixel.
        # Here, we check for the minimum norm first, then concatenate (sum) in the final layer.
        scaled_s_map = torch.zeros(1, 1, self.image_size, self.image_size)
        for idx, fmap in enumerate(feature_maps):
            nearest_fmaps = torch.index_select(self.feature_maps[idx], 0, indices)
            # min() because kappa=1 in the paper
            s_map, _ = torch.min(torch.linalg.norm(nearest_fmaps - fmap, dim=1), 0, keepdims=True)
            scaled_s_map += torch.nn.functional.interpolate(
                s_map.unsqueeze(0), size=(self.image_size, self.image_size), mode='bilinear'
            )

        scaled_s_map = self.blur(scaled_s_map)

        return z_score, scaled_s_map

    def get_parameters(self):
        return super().get_parameters({
            "k": self.k,
        })

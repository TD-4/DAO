#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
"""
Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""

import importlib
import albumentations


def get_transformer(transform_params):
    trans_albumentations = []
    for i, (k, v) in enumerate(transform_params.items()):
        if getattr(albumentations, k, False):
            trans_albumentations.append(getattr(albumentations, k)(**v))
        else:
            custom = importlib.import_module(
                "core.modules.dataloaders.augments.custom.{}".format(k)).Custom(v)
            trans_albumentations.append(custom)
    return albumentations.Compose(trans_albumentations)



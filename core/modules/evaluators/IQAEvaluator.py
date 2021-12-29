#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import contextlib
import io
import itertools
import json
import tempfile
import time
from loguru import logger
from tqdm import tqdm

import torch

from core.utils import is_main_process, synchronize, time_synchronized, gather
from core.modules.register import Registers
from core.modules.utils import MeterClsTrain


import contextlib
import io
import os
import cv2
import itertools
import json
import numpy as np
import shutil
from PIL import Image
import itertools
import matplotlib.pyplot as plt
import tempfile
import time
from loguru import logger
from tqdm import tqdm

import torch
from torchcam.cams import SmoothGradCAMpp, CAM
from torchcam.utils import overlay_mask
import torch.nn.functional as F
from torchvision.transforms.functional import normalize, resize, to_pil_image

from core.modules.utils import MeterClsEval
from core.utils import is_main_process, synchronize, time_synchronized, gather
from core.modules.register import Registers


@Registers.evaluators.register
class IQAEvaluator:
    def __init__(self, is_distributed=False, dataloader=None, num_classes=None, is_industry=False, industry=None):
        self.dataloader, self.iters_per_epoch = Registers.dataloaders.get(dataloader.type)(
            is_distributed=is_distributed,
            dataset=dataloader.dataset,
            **dataloader.kwargs
        )
        self.meter = MeterClsEval(num_classes)
        self.best_acc = 0
        self.num_class = num_classes
        self.is_industry = is_industry
        self.industry = industry

    def evaluate(self, model, distributed=False, half=False, device=None, output_dir=None):
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        data_list = []  # 存放最终的结果，多个gpu上汇集来的结果
        progress_bar = tqdm if is_main_process() else iter

        for imgs, targets, paths in progress_bar(self.dataloader):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)
                outputs = model(imgs)
                data_list.append((outputs, targets, paths))

        if distributed:
            output_s = []
            target_s = []
            path_s = []
            data_list = gather(data_list, dst=0)
            for data_ in data_list:     # multi gpu
                for pred, target, path in data_:  # 每个gpu所具有的sample
                    output_s.append(pred)
                    target_s.append(target)
                    path_s.append(path)
        else:
            output_s = []
            target_s = []
            path_s = []
            for data_ in data_list:
                output_s.append(data_[0])
                target_s.append(data_[1])
                path_s.append(data_[2])
        if not is_main_process():
            top1, top2, confu_ma = 0, 0, None
        else:
            self.meter.update(
                outputs=torch.cat([output.to(device=device) for output in output_s]),
                targets=torch.cat([output.to(device=device) for output in target_s])
            )
            self.meter.eval_confusionMatrix(
                preds=torch.cat([output.to(device=device) for output in output_s]),
                labels=torch.cat([output.to(device=device) for output in target_s])
            )

            top1 = self.meter.precision_top1.avg
            top2 = self.meter.precision_top2.avg
            confu_ma = self.meter.confusion_matrix
            self.meter.precision_top1.initialized = False
            self.meter.precision_top2.initialized = False
            self.meter.confusion_matrix = [[0 for j in range(self.num_class)] for i in range(self.num_class)]
        synchronize()

        return top1, top2, confu_ma

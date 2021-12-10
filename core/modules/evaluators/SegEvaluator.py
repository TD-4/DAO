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
from core.modules.utils import MeterSegEval


@Registers.evaluators.register
class SegEvaluator:
    def __init__(
            self,
            is_distributed=False,
            dataloader=None,
            num_classes=None,
    ):
        self.dataloader, self.iters_per_epoch = Registers.dataloaders.get(dataloader.type)(
            is_distributed=is_distributed,
            dataset=dataloader.dataset,
            **dataloader.kwargs
        )
        self.meter = MeterSegEval(num_classes)
        self.num_classes = num_classes

    def evaluate(self, model, distributed=False, half=False, device=None):
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        data_list = []

        progress_bar = tqdm if is_main_process() else iter

        self.meter.reset_metrics()
        for cur_iter, (imgs, targets, paths) in enumerate(progress_bar(self.dataloader)):
            with torch.no_grad():
                imgs = imgs.to(device=device)
                targets = targets.to(device=device)
                imgs = imgs.type(tensor_type)
                outputs = model(imgs)
                data_list.append((outputs, targets, paths))
        if distributed:
            output_s = []
            target_s = []
            path_s = []
            data_list = gather(data_list, dst=0)
            for data_ in data_list:     # multi gpu
                for pred, target in data_:  # 每个gpu所具有的sample
                    output_s.append(pred)
                    target_s.append(target)
        else:
            output_s = []
            target_s = []
            path_s = []
            for o, t, p in data_list:     # data_list [batchsize, batchsize,...]
                output_s.append(o)  # torch.Size([6, 21, 224, 224])
                target_s.append(t)  # torch.Size([6, 224, 224])
                path_s.append(p)    # list[list of jpg, list of png]

        if not is_main_process():
            return 0, 0, None
        else:
            for o, t, p in zip(output_s, target_s, path_s):
                seg_metrics = self.meter.eval_metrics(o, t, self.num_classes)
                self.meter.update_seg_metrics(*seg_metrics)
        # PRINT INFO
        pixAcc, mIoU, Class_IoU = self.meter.get_seg_metrics().values()
        return pixAcc, mIoU, Class_IoU

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
from core.modules.utils import Meter_Cls


@Registers.evaluators.register
class CLS_TXT_Evaluator:
    def __init__(
            self,
            is_distributed=False,
            type_=None,
            dataset=None,
            num_classes=None,
            **kwargs
    ):
        self.dataloader, self.iters_per_epoch = Registers.dataloaders.get(type_)(
            is_distributed=is_distributed,
            dataset=dataset,
            **kwargs
        )
        self.meter = Meter_Cls(num_classes)
        self.best_acc = 0
        self.num_class = num_classes

    def evaluate(self, model, distributed=False, half=False, device=None):
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        n_samples = max(len(self.dataloader)-1, 1)
        for cur_iter, (imgs, targets, paths) in enumerate(progress_bar(self.dataloader)):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)
                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()
                outputs = model(imgs)
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start
                data_list.append((outputs, targets))

        if distributed:
            output_s = []
            target_s = []
            tmp = []
            data_list = gather(data_list, dst=0)
            for data_ in data_list:     # multi gpu
                for pred, target in data_:  # 每个gpu所具有的sample
                    output_s.append(pred)
                    target_s.append(target)
        else:
            output_s = []
            target_s = []
            tmp = []
            data_list = data_list
            for data_ in data_list:
                output_s.append(data_[0])
                target_s.append(data_[1])
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
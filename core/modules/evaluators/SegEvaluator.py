#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.


from tqdm import tqdm

import torch

from core.utils import is_main_process, synchronize, time_synchronized, gather
from core.modules.register import Registers
from core.modules.utils import MeterSegEval


@Registers.evaluators.register
class SegEvaluator:
    def __init__(self, is_distributed=False, dataloader=None, num_classes=None):
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
        data_list = []  # 用于存放all world size汇集的数据

        progress_bar = tqdm if is_main_process() else iter

        self.meter.reset_metrics()
        for imgs, targets, paths in progress_bar(self.dataloader):
            with torch.no_grad():
                imgs = imgs.to(device=device)
                targets = targets.to(device=device)
                imgs = imgs.type(tensor_type)
                outputs = model(imgs)
                data_list.append((outputs, targets, paths)) # data_list:[([32, 21, 224, 224], [32, 21, 224, 224],[(image_p,...),(mask_p,...)]), ...]

        if distributed:  # 如果是分布式，将结果gather到0设备上
            output_s = []   #
            target_s = []
            path_s = []
            data_list = gather(data_list, dst=0)
            # data_list [[([32, 21, 224, 224], [32, 21, 224, 224],[(image_p,...),(mask_p,...)]), ...], ..., N_gpus]
            for data_ in data_list:     # multi gpu
                for pred, target, path in data_:  # 每个gpu所具有的sample
                    output_s.append(pred)  # [(16, 21, 224, 224), ...] 16为设定batchsize/ngpus
                    target_s.append(target)  # [(16, 224, 224), ...]
                    path_s.append(path)  # [(16个path, 16个path), ...]
        else:
            output_s = []
            target_s = []
            path_s = []
            for o, t, p in data_list:  # data_list:[([32, 21, 224, 224], [32, 21, 224, 224],[(image_p,...),(mask_p,...)]), ...]
                output_s.append(o)  # [[6, 21, 224, 224], ...]
                target_s.append(t)  # [[6, 21, 224, 224], ...]
                path_s.append(p)    # list[list of jpg, list of png]

        if not is_main_process():
            return 0, 0, None
        else:
            for o, t, p in zip(output_s, target_s, path_s):
                seg_metrics = self.meter.eval_metrics(o, t, self.num_classes)
                self.meter.update_seg_metrics(*seg_metrics)
        # PRINT INFO
        pixAcc, mIoU, Class_IoU = self.meter.get_seg_metrics().values()
        Class_IoU_dict = {}
        for k, v in Class_IoU.items():
            Class_IoU_dict[self.dataloader.dataset.labels_dict[str(k)]] = v
        return pixAcc, mIoU, Class_IoU_dict

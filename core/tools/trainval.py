#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import random
import warnings
from loguru import logger
from dotmap import DotMap

import torch
import torch.backends.cudnn as cudnn

from core.utils import configure_nccl, configure_omp, get_num_devices
from core.trainers import launch, ClsTrainer


def TrainVal(config=None):
    exp = DotMap(config)

    # get env info
    num_gpu = get_num_devices() if exp.envs.gpus.devices is None else exp.envs.gpus.devices
    assert num_gpu <= get_num_devices()
    dist_url = "auto" if exp.envs.gpus.dist_url is None else exp.envs.gpus.dist_url
    num_machines = exp.envs.gpus.num_machines
    machine_rank = exp.envs.gpus.machine_rank
    dist_backend = exp.envs.gpus.dist_backend

    cache = exp.train_loader.dataset.kwargs.cache

    launch(main, num_gpu, num_machines, machine_rank, backend=dist_backend, dist_url=dist_url, cache=cache, args=(exp,))


@logger.catch
def main(exp):
    if exp.seed != 0:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # set environment variables for distributed training
    configure_nccl()
    configure_omp()
    cudnn.benchmark = True

    if exp.type == "cls":
        trainer = ClsTrainer(exp)
        trainer.train()


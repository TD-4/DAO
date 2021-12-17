
# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.12.17
# @GitHub:https://github.com/felixfu520
# @Copy From:

import random
import warnings
from loguru import logger
from dotmap import DotMap

import torch
import torch.backends.cudnn as cudnn

from core.utils import configure_nccl, configure_omp, get_num_devices
from core.trainers import *
from core.tools import register_modules


def TrainVal(config=None, custom_modules=None):
    exp = DotMap(config)

    # get env info
    num_gpu = get_num_devices() if exp.envs.gpus.devices is None else exp.envs.gpus.devices
    assert num_gpu <= get_num_devices()
    dist_url = "auto" if exp.envs.gpus.dist_url is None else exp.envs.gpus.dist_url
    num_machines = exp.envs.gpus.num_machines
    machine_rank = exp.envs.gpus.machine_rank
    dist_backend = exp.envs.gpus.dist_backend

    cache = exp.dataloader.dataset.kwargs.cache

    launch(main, num_gpu, num_machines, machine_rank, backend=dist_backend, dist_url=dist_url, cache=cache, args=(exp, custom_modules))


@logger.catch
def main(exp, custom_modules):
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

    register_modules(custom_modules=custom_modules)   # 注册所有组件

    if exp.type == "cls":
        trainer = ClsTrainer(exp)
        trainer.train()
    elif exp.type == "seg":
        trainer = SegTrainer(exp)
        trainer.train()
    elif exp.type == 'det':
        trainer = DetTrainer(exp)
        trainer.train()
    elif exp.type == 'anomaly':
        trainer = AnomalyTrainer(exp)
        trainer.train()
    else:
        logger.error("this type {} is not supported, now supported cls, det, seg, anomaly.".format(exp.type))



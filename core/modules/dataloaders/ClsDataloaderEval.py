#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @auther:FelixFu
# @Date: 2021.10.1
# @github:https://github.com/felixfu520
from loguru import logger

import torch.multiprocessing

from core.modules.register import Registers
from core.modules.dataloaders.augments import get_transformer
from core.modules.dataloaders.utils.dataloading import DataLoader, worker_init_reset_seed
from core.modules.dataloaders.utils.samplers import InfiniteSampler, BatchSampler
from core.modules.dataloaders.utils.data_prefetcher import DataPrefetcherCls
from core.utils import wait_for_the_master, get_local_rank, get_world_size


@Registers.dataloaders.register
def ClsDataloaderEval(
        is_distributed=False,
        batch_size=None,
        num_workers=None,
        dataset=None,
        **kwargs
):
    valdataset = Registers.datasets.get(dataset.type)(
        preproc=get_transformer(dataset.transforms.kwargs),
        **dataset.kwargs
    )
    if is_distributed:
        batch_size = batch_size // get_world_size()
        sampler = torch.utils.data.distributed.DistributedSampler(valdataset, shuffle=False)
    else:
        sampler = torch.utils.data.SequentialSampler(valdataset)

    dataloader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": True,
        "sampler": sampler,
    }
    dataloader_kwargs["batch_size"] = batch_size
    val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)
    return val_loader, len(val_loader)
    # return DataPrefetcherCls(val_loader), len(val_loader)

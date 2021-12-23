# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Copy From:


from torch.utils.data import DataLoader

from core.modules.register import Registers
from core.modules.dataloaders.augments import get_transformer


@Registers.dataloaders.register
class MVTecDataloader(DataLoader):
    def __init__(self, dataset, batch_size=1, num_workers=2, shuffle=False):
        dataset = Registers.datasets.get(dataset.type)(
            preproc=get_transformer(dataset.transforms.kwargs), **dataset.kwargs)
        super(MVTecDataloader, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )




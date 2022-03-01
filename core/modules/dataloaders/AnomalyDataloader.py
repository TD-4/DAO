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


if __name__ == "__main__":
    from core.modules.dataloaders.augments import get_transformer
    from dotmap import DotMap
    dataloader_c = {
        "dataset": {
            "type": "MVTecDataset",
            "kwargs": {
                "data_dir": "/root/data/DAO/mvtec_anomaly_detection/bottle",
                "image_set": "test.txt",
                "in_channels": 3,
                "cache": True,
                "image_suffix": ".png",
                "mask_suffix": ".png"
            },
            "transforms": {
                "kwargs": {
                    "Resize": {"height": 224, "width": 224, "p": 1, "interpolation": 0},
                    "Normalize": {"mean": 0, "std": 1, "p": 1}

                }
            }
        },
        "kwargs": {
            "batch_size": 1,
            "num_workers": 2,
            "shuffle": False
        }

    }
    dataloader_c = DotMap(dataloader_c)
    transformer = get_transformer(dataloader_c.transforms.kwargs)
    dataloader = MVTecDataloader(dataset=dataloader_c.dataset, **dataloader_c.kwargs)
    print(dataloader)


# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.12.17
# @GitHub:https://github.com/felixfu520
# @Copy From: https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master/blob/main/datasets/mvtec.py

import os
import numpy as np
from PIL import Image
from loguru import logger

from torch.utils.data import Dataset

from core.modules.register import Registers


@Registers.datasets.register
class MVTecDataset(Dataset):
    def __init__(self,
                 data_dir=None,
                 preproc=None,
                 image_set="",
                 in_channels=1,
                 input_size=(224, 224),
                 cache=False,
                 image_suffix=".png",
                 mask_suffix=".png",
                 **kwargs
                 ):
        """
        异常检测数据集，（MVTecDataset类型）

        data_dir:str  数据集文件夹路径，文件夹要求是
            📂datasets
             ┗ 📂your_custom_dataset
              ┣ 📂 ground_truth
              ┃ ┣ 📂 defective_type_1
              ┃ ┗ 📂 defective_type_2
              ┣ 📂 test
              ┃ ┣ 📂 defective_type_1
              ┃ ┣ 📂 defective_type_2
              ┃ ┗ 📂 good
              ┗ 📂 train
              ┃ ┗ 📂 good

        preproc:albumentations.Compose 对图片进行预处理
        image_set:str "train.txt or val.txt or test.txt"
        in_channels:int  输入图片的通道数，目前只支持1和3通道
        input_size:tuple 输入图片的HW
        cache:bool 是否对图片进行内存缓存
        image_suffix:str 可接受的图片后缀
        mask_suffix:str 可接受的图片后缀
        """
        # set attr
        self.root = data_dir
        self.preproc = preproc
        self.is_train = True if image_set == "train.txt" else False
        self.in_channels = in_channels
        self.img_size = input_size
        self.image_suffix = image_suffix
        self.mask_suffix = mask_suffix

        # 存储image-mask pair
        self.x, self.y, self.mask = self.load_dataset_folder()

        if cache:
            logger.warning("MVTecDataset not supported cache !")

    def __getitem__(self, index):
        image, mask, image_path = self.pull_item(index)  # image:ndarray, label:ndarray, image_path:string
        if self.preproc is not None:
            transformed = self.preproc(image=image, mask=mask)
            image, mask = transformed['image'], transformed["mask"]
        image = image.transpose(2, 0, 1)  # c, h, w
        mask = mask.astype(np.int64)  # c, h, w
        return image, mask, image_path

    def pull_item(self, index):
        img, mask = self._load_img(index)
        if self.in_channels == 1:
            img = np.expand_dims(img.copy(), axis=2)
        elif self.in_channels == 3:
            img = img.copy()
        return img, mask, self.x[index]

    def _load_img(self, index):
        image_path = self.x[index]
        has_mask = self.y[index]
        mask_path = self.mask[index]

        # get image
        image = None
        if self.in_channels == 1:
            image = Image.open(image_path).convert('L')
        elif self.in_channels == 3:
            image = Image.open(image_path)
        image = np.array(image)

        # get mask
        if has_mask == 0:
            mask = np.zeros(image.shape[:2])
        else:
            mask = Image.open(mask_path)
            mask = np.array(mask)

        return image, mask

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.root, phase)
        gt_dir = os.path.join(self.root, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:
            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith(self.image_suffix)])
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask' + self.mask_suffix)
                                 for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)

    def __len__(self):
        return len(self.x)

    def __repr__(self):
        fmt_str = "Dataset:" + self.__class__.__name__
        fmt_str += "; Length:{}".format(self.__len__())
        fmt_str += "; Data_dir:{}".format(self.root)
        return fmt_str


if __name__ == "__main__":
    from core.modules.dataloaders.augments import get_transformer
    from dotmap import DotMap
    kk = {
        "kwargs": {
            "Resize": {"p": 1, "height": 224, "width": 224, "interpolation": 0},
            "Normalize": {"mean": 0, "std": 1, "p": 1}
        }
    }

    dataset = MVTecDataset(
        data_dir="/root/data/DAO/mvtec_anomaly_detection/bottle",
        preproc=get_transformer(kk["kwargs"]),
        image_set="val.txt",
        in_channels=1,
        input_size=(224, 224),
        cache=True
    )

    for i in range(len(dataset)):
        img, mask, img_p = dataset.__getitem__(i)
        print("image path:{}-->mask unique:{}".format(img_p, np.unique(mask)))

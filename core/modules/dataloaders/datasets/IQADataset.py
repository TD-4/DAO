
# -*- coding: utf-8 -*-
# @auther:FelixFu
# @Date: 2021.4.14
# @github:https://github.com/felixfu520
# @copy from:

import os
import numpy as np
from PIL import Image
from loguru import logger
import cv2

import torch
from torch.utils.data import Dataset

from core.modules.register import Registers


@Registers.datasets.register
class IQADataset(Dataset):
    def __init__(self,
                 data_dir=None,
                 preproc=None,
                 image_set="",
                 in_channels=3,
                 num_classes=10,
                 input_size=(224, 224),
                 cache=False,
                 image_suffix=".jpg",
                 separator="_"
                 ):
        """
        分割数据集

        data_dir:str  数据集文件夹路径，文件夹要求是
            |-dataset
                |- images
                    |-图片
                |- masks

        image_set:str "train.txt or val.txt or test.txt"
        in_channels:int  输入图片的通道数，目前只支持1和3通道
        input_size:tuple 输入图片的HW
        preproc:albumentations.Compose 对图片进行预处理
        cache:bool 是否对图片进行内存缓存
        separator:str None
        train_ratio:float 生成train.txt的比例
        shuffle:bool 生成train.txt时，folder中的数据是否随机打乱
        sample_range:tuple 每类允许的最多图片数量的范围
        images_suffix:list[str] 可接受的图片后缀
        """
        # set attr
        self.root = data_dir
        self.preproc = preproc
        self.image_set = image_set
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = input_size
        self.image_suffix = image_suffix
        self.separator = separator

        # 存储数据
        self.ids = []
        self.labels = list()
        self.labels_dict = dict()

        self._set_ids()  # 获取所有文件名，存放到self.ids中
        self.imgs = None
        self.masks = None
        if cache:
            logger.warning("this dataset don't supported cache")

    def __getitem__(self, index):
        images, label, image_path = self.pull_item(index)  # image:ndarray, label:ndarray, image_path:string
        imgs = []
        for image in images:
            if self.preproc is not None:
                transformed = self.preproc(image=image)
                image = transformed['image']
            image = image.transpose(2, 0, 1)  # c, h, w
            imgs.append(image)
        imgs = np.concatenate(imgs, 0)
        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        return imgs, label, image_path


    def pull_item(self, index):
        return self._load_img(index)

    def _load_img(self, index):
        """
        功能：通过文件名获得，图片
        :param index:
        :return:
        """
        image_path = self.ids[index]
        label = self.labels[index]

        # get image
        imgs = []
        for i, img_p in enumerate(os.listdir(os.path.join(self.root, image_path))):
            image = None
            name = str(i) + ".bmp"
            if self.in_channels == 1:
                image = Image.open(os.path.join(self.root, image_path, img_p)).convert('L')
            elif self.in_channels == 3:
                image = Image.open(os.path.join(self.root, image_path, img_p))
            image = np.array(image)
            imgs.append(image)
        while len(imgs) < self.num_classes:
            if self.in_channels == 1:
                imgs.append(np.zeros((224, 224, 1), np.uint8))
            elif self.in_channels == 3:
                imgs.append(np.zeros((224, 224, 3), np.uint8))
        return imgs, label, image_path

    def _load_resize_img(self, index):
        pass

    def _set_ids(self, separator="_"):
        """
        功能：获取所有文件的文件名和标签
        """
        list_path = os.path.join(self.root, self.image_set)

        with open(list_path, 'r', encoding='utf-8') as images_labels:
            for image_label in images_labels:
                self.ids.append(os.path.join(self.root, image_label.strip()))
                self.labels.append(image_label.strip().split(separator)[1])

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    # Root: {}".format(self.root)
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

    transforms = get_transformer(kk["kwargs"])
    seg_d = IQADataset(
        data_dir="/root/data/DAO/old_hh", preproc=transforms,  image_set="vallist.txt", in_channels=3,
        num_classes=10,
        input_size=(224, 224), cache=False,  image_suffix=".bmp", separator = "_")
    a, b, c = seg_d.__getitem__(2)
    # print(a, np.unique(b), c)
    print(c)
    print(np.unique(b))


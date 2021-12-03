# _*_coding:utf-8_*_
# @auther:FelixFu
# @Date: 2021.4.14
# @github:https://github.com/felixfu520

import os
import random
import functools
import numpy as np
from PIL import Image
from loguru import logger
import cv2

import torch
from torch.utils.data import Dataset

from core.modules.register import Registers


@Registers.datasets.register
class CLS_TXT(Dataset):
    def __init__(self, data_dir=None, image_set="", in_channels=1,
                 input_size=(224, 224), preproc=None, cache=False,
                 separator=":", train_ratio=0.9, shuffle=True,
                 sample_range=(2000, 3000), images_suffix=None):
        """
        分类数据集

        data_dir:str  数据集文件夹路径，文件夹要求是
            |-dataset
                |- 类别1
                    |-图片
                |- 类别2

        image_set:str "trainlist.txt or vallist.txt"
        in_channels:int  输入图片的通道数，目前只支持1和3通道
        input_size:tuple 输入图片的HW
        preproc:albumentations.Compose 对图片进行预处理
        cache:bool 是否对图片进行内存缓存
        separator:str labels.txt id与name的分隔符
        train_ratio:float 生成trianlist.txt的比例
        shuffle:bool 生成trainlist.txt时，folder中的数据是否随机打乱
        sample_range:tuple 每类允许的最多图片数量的范围
        images_suffix:list[str] 可接受的图片后缀
        """
        if images_suffix is None:
            images_suffix = [".bmp"]
        self.root = data_dir
        self.image_set = image_set
        self.in_channels = in_channels
        self.img_size = input_size
        self.preproc = preproc

        self.ids = []
        self.labels = list()
        self.labels_dict = dict()

        # 将Folder格式数据集 转为 TXT格式数据集
        split_txt = os.path.join(data_dir, image_set) if (data_dir is not None and image_set is not None) else None
        if not os.path.exists(split_txt):
            logger.info("generate dataset  in '{}' folder labels.txt,trainlist.txt,vallist.txt".format(data_dir))
            self._gen_label_txt(data_dir)
            self._gen_trainvallist(data_dir, train_ratio=train_ratio, shuffle=shuffle, suffix=images_suffix,
                                   sample_range=sample_range)

        self._set_ids(separator=separator)  # 获取所有文件的路径和标签，存放到files中

        self.imgs = None
        if cache:
            self._cache_images()

    def __getitem__(self, index):
        image, label, image_path = self.pull_item(index)  # image:mem, label:tensor(long), image_path:string
        image = np.asarray(image)
        if self.preproc is not None:
            image = self.preproc(image=image)['image']
        image = image.transpose(2, 0, 1)  # c, h, w
        return image, label, image_path

    def pull_item(self, index):
        if self.imgs is not None:
            img = self.imgs[index]
        else:
            img = self._load_resize_img(index)
            if self.in_channels == 1:
                img = np.expand_dims(img.copy(), axis=2)
            elif self.in_channels == 3:
                img = img.copy()
        label = torch.from_numpy(np.array(self.labels[index], dtype=np.int32)).long()
        # label = np.array(self.labels[index], dtype=np.int32)
        return img, label, self.ids[index]

    def _load_resize_img(self, index):
        img = self._load_img(index)
        return np.asarray(cv2.resize(img, self.img_size))

    def _load_img(self, index):
        """
        功能：通过文件名获得，图片
        :param index:
        :return:
        """
        image_path = self.ids[index]
        img = None
        if self.in_channels == 1:
            img = Image.open(image_path).convert('L')
        elif self.in_channels == 3:
            img = Image.open(image_path)
        return np.asarray(img)

    def _cache_images(self):
        """
        预加载所有图片到RAM中
        """
        logger.warning(
            "\n********************************************************************************\n"
            "You are using cached images in RAM to accelerate training.\n"
            "This requires large system RAM.\n"
            "Make sure you have Enough RAM and Available disk space.\n"
            "********************************************************************************\n"
        )
        max_h = self.img_size[0]
        max_w = self.img_size[1]
        cache_file = self.root + "/img_resized_cache.array"
        if not os.path.exists(cache_file):
            logger.info("Caching images for the frist time. This might take sometime")
            # np.memmap为存储在磁盘上的二进制文件中的数组创建内存映射。
            # 内存映射文件用于访问磁盘上的大段文件，而无需将整个文件读入内存。
            self.imgs = np.memmap(
                cache_file,
                shape=(len(self.ids), max_h, max_w, self.in_channels),
                dtype=np.uint8,
                mode="w+",
            )
            from tqdm import tqdm
            from multiprocessing.pool import ThreadPool

            NUM_THREADs = min(8, os.cpu_count())
            loaded_images = ThreadPool(NUM_THREADs).imap(
                lambda x: self._load_resize_img(x),
                range(len(self.ids)),
            )
            pbar = tqdm(enumerate(loaded_images), total=len(self.labels))
            for k, out in pbar:
                if self.in_channels == 1:
                    self.imgs[k][: out.shape[0], : out.shape[1], :] = np.expand_dims(out.copy(), axis=2)
                elif self.in_channels == 3:
                    self.imgs[k][: out.shape[0], : out.shape[1], :] = out.copy()
            self.imgs.flush()
            pbar.close()
        else:
            logger.warning(
                "You are using cached imgs! Make sure your dataset is not changed!!"
            )

        logger.info("Loading cached imgs...")
        self.imgs = np.memmap(
            cache_file,
            shape=(len(self.ids), max_h, max_w, self.in_channels),
            dtype=np.uint8,
            mode="r+",
        )

    def _set_ids(self, separator=" "):
        """
        功能：获取所有文件的文件名和标签
        """
        list_path = os.path.join(self.root, self.image_set)

        with open(list_path, 'r', encoding='utf-8') as images_labels:
            for image_label in images_labels:
                self.ids.append(os.path.join(self.root, image_label.strip().split(separator)[0]))
                self.labels.append(image_label.strip().split(separator)[1])

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    # Root: {}".format(self.root)
        return fmt_str

    def _gen_label_txt(self, data_dir):
        label_txt = os.path.join(data_dir, "labels.txt")
        all_folder = [p for p in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, p))]  # 文件夹下所有文件
        all_folder.sort()
        with open(label_txt, "a+") as labels_file:
            for i, folder in enumerate(all_folder):
                labels_file.write("{}:{}\n".format(folder, i))
                self.labels_dict[folder] = i

    def _gen_trainvallist(self, data_dir, train_ratio=0.9, shuffle=True, suffix=None, sample_range=(2000, 3000)):
        suffix = [".bmp"] if shuffle is None else suffix
        all_folders = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]
        all_folders.sort()
        train_labels = os.path.join(data_dir, "trainlist.txt")
        val_labels = os.path.join(data_dir, "vallist.txt")
        data_count = [0, 0, 0]  # class_one:[n_total, len(train_list), len(val_list)] + ...
        with open(train_labels, "a+") as train_file, open(val_labels, "a+") as val_file:
            for folder in all_folders:  # 处理每个文件夹
                label = self.labels_dict[str(folder)]  # 获取标签

                all_images = [img_p for img_p in os.listdir(os.path.join(data_dir, folder)) if img_p[-4:] in suffix]
                n_total = min(random.randint(sample_range[0], sample_range[1]), len(all_images))  # 控制每类样本数量
                offset = int(n_total * train_ratio)
                train_list = all_images[:offset]  # 训练集图片路径
                val_list = all_images[offset:n_total]  # 验证集图片路径
                logger.info("class '{}', train set {} images, val set {} images".format(str(folder), len(train_list), len(val_list)))
                tmp = [n_total, len(train_list), len(val_list)]
                data_count = [d[0] + d[1] for d in zip(tmp, data_count)]
                if shuffle:
                    random.shuffle(all_images)
                # 写入训练和测试图片路径到train/vallist.txt文件中
                for train_img in train_list:
                    train_file.write("{}/{}:{}\n".format(folder, train_img, int(label)))
                for val_img in val_list:
                    val_file.write("{}/{}:{}\n".format(folder, val_img, int(label)))

        logger.info("trainlist {} images, vallist {} images, total {} images".format(data_count[1], data_count[2], data_count[0]))


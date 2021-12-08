#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import contextlib
import io
import os
import cv2
import itertools
import json
import numpy as np
import shutil
from PIL import Image
import itertools
import matplotlib.pyplot as plt
import tempfile
import time
from loguru import logger
from tqdm import tqdm

import torch
from torchcam.cams import SmoothGradCAMpp, CAM
from torchcam.utils import overlay_mask
import torch.nn.functional as F
from torchvision.transforms.functional import normalize, resize, to_pil_image

from core.utils import is_main_process, synchronize, time_synchronized, gather
from core.modules.register import Registers
from core.modules.utils import Meter_Cls


@Registers.evaluators.register
class ClsEvaluatorIndustry:
    def __init__(
            self,
            is_distributed=False,
            dataloader=None,
            num_classes=None,
            industry=None,
            **kwargs
    ):
        self.dataloader, self.iters_per_epoch = Registers.dataloaders.get(dataloader.type)(
            is_distributed=is_distributed,
            dataset=dataloader.dataset,
            **dataloader.kwargs
        )
        self.meter = Meter_Cls(num_classes)
        self.best_acc = 0
        self.num_class = num_classes
        self.industry = industry

    def evaluate(self, model, distributed=False, device=None, output_dir=None):
        model = model.eval()
        data_list = []
        path_list = []
        progress_bar = tqdm if is_main_process() else iter

        for cur_iter, (imgs, targets, paths) in enumerate(progress_bar(self.dataloader)):
            targets = targets.to(device=device)
            with torch.no_grad():
                outputs = model(imgs.to(device=device))
                data_list.append((outputs, targets))
                path_list.append(paths)

        if distributed:
            output_s = []
            target_s = []
            path_s = []
            data_list = gather(data_list, dst=0)
            for data_ in data_list:     # multi gpu
                for pred, target in data_:  # 每个gpu所具有的sample
                    output_s.append(pred)
                    target_s.append(target)
            for p in path_list:     # multi gpu
                for p_ in p:  # 每个gpu所具有的sample
                    path_s.append(p_)
        else:
            output_s = []
            target_s = []
            path_s = []
            data_list = data_list
            for data_ in data_list:
                output_s.append(data_[0])
                target_s.append(data_[1])
            for p in path_list:
                path_s.append(p)

        if not is_main_process():
            top1, top2, confu_ma = 0, 0, None
        else:
            self._industry(
                model=model,
                outputs=torch.cat([output.to(device=device) for output in output_s]),
                targets=torch.cat([output.to(device=device) for output in target_s]),
                paths=[p for path_ in path_s for p in path_],
                output_dir=output_dir
            )
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

    def _industry(self, model=None, outputs=None, targets=None, output_dir=None, paths=None):
        # path_ng + path_ok + path_other = all images
        path_ng = os.path.join(output_dir, "validate", "ng")  # scores >threshold, then label != pred
        path_ok = os.path.join(output_dir, "validate", "ok")  # scores >threshold, then label == pred
        path_other = os.path.join(output_dir, "validate", "other")  # scores < threshold
        path_tolerate = os.path.join(output_dir, "validate", "tolerate")
        path_gj = os.path.join(output_dir, "validate", "gj")
        path_lj = os.path.join(output_dir, "validate", "lj")
        os.makedirs(path_ng, exist_ok=True)
        os.makedirs(path_ok, exist_ok=True)
        os.makedirs(path_other, exist_ok=True)
        os.makedirs(path_tolerate, exist_ok=True)
        os.makedirs(path_gj, exist_ok=True)
        os.makedirs(path_lj, exist_ok=True)
        m = self.dataloader.dataset.labels_dict
        labels_ = dict(zip(m.values(), m.keys()))
        class_names = list(m.keys())
        # 定义混淆矩阵
        num_classes = self.num_class
        confusion_matrix = [[0 for j in range(num_classes)] for i in range(num_classes)]
        pred_target_list = []  # 存放混淆矩阵中所用到的数据
        pred_target_list_tolerate_count = {}  # 计算容忍混淆矩阵使用
        logger.info("\n预测的分数为scores，预测标签为pred， 实际标签为label\n"
                    "1.当scores<阈值, 将结果放入validate/other目录下\n"
                    "2.当scores>阈值, 且为过检， 将过检放到validate/gj目录下\n"
                    "3.当scores>阈值, 且为漏检， 将漏检放到validate/lj目录下\n"
                    "3.当scores>阈值, 且pred!=label, 且可容忍，将结果放入validate/tolerate目录下\n"
                    "3.当scores>阈值, 且pred!=label, 且不可容忍，将结果放入validate/ng目录下\n"
                    "4.当scores>阈值，且pred==label， 将结果放入validate/ok目录下\n"
                    "总的图片 = other + ok + tolerate + ng")
        with torch.no_grad():
            cam_extractor = CAM(model, target_layer="_conv_head")
            for output_tensor, label, img_p in zip(outputs, targets, paths):
                label = str(label.cpu().numpy().item())
                prediction = output_tensor.squeeze(0).cpu().detach().numpy()
                prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()

                scores = F.softmax(output_tensor.squeeze(0), dim=0).cpu().numpy().max()  # top1 分数
                pred = prediction.item()  # top1下标

                # 1、预测值小于阈值 --> 将图片、CAM、增强图放入other文件夹中
                if scores < 0.3:
                    self._copy_img_his_cam(labels_=labels_, pred=pred, label=label, scores=scores, img_p=img_p,
                                           dst_path=path_ok, cam_extractor=cam_extractor, output_tensor=output_tensor)
                    continue

                pred_target_list.append((pred, label))  # 记录预测值和标签值，供计算混淆矩阵使用 (int,str)
                # -----------------过检、漏检图片输出------------------------
                pred_, target_ = pred, label
                # 为每个预测和target标记OK或者NG
                pred_flag = "ok" if labels_[str(pred_)] in self.industry.kwargs.ok_ng_class.ok else "ng"
                target_flag = "ok" if labels_[str(target_)] in self.industry.kwargs.ok_ng_class.ok else "ng"
                # 2. 过检--> 将图片、CAM、增强图放入gj文件夹中
                if pred_flag == "ng" and target_flag == "ok":
                    self._copy_img_his_cam(labels_=labels_, pred=pred, label=label, scores=scores, img_p=img_p,
                                           dst_path=path_gj, cam_extractor=cam_extractor, output_tensor=output_tensor)
                # 3. 漏检--> 将图片、CAM、增强图放入lj文件夹中
                elif pred_flag == "ok" and target_flag == "ng":
                    self._copy_img_his_cam(labels_=labels_, pred=pred, label=label, scores=scores, img_p=img_p,
                                           dst_path=path_lj, cam_extractor=cam_extractor, output_tensor=output_tensor)
                # ---------------过检、漏检图片输出--END----------------------------------

                # 预测错误， 输出图片、CAM
                if pred != int(label):
                    # 2、可容忍的类别分错
                    tolerate_class = self.industry.kwargs.tolerate_class
                    if labels_[str(pred)] in tolerate_class.keys() and labels_[label] in tolerate_class[
                        labels_[str(pred)]]:
                        if (pred, int(label)) in pred_target_list_tolerate_count.keys():
                            pred_target_list_tolerate_count[(pred, int(label))] += 1
                        else:
                            pred_target_list_tolerate_count[(pred, int(label))] = 1
                        self._copy_img_his_cam(labels_=labels_, pred=pred, label=label, scores=scores, img_p=img_p,
                                               dst_path=path_tolerate, cam_extractor=cam_extractor,
                                               output_tensor=output_tensor)
                    else:  # 3、不可容忍的类别分错
                        self._copy_img_his_cam(labels_=labels_, pred=pred, label=label, scores=scores, img_p=img_p,
                                               dst_path=path_ng, cam_extractor=cam_extractor,
                                               output_tensor=output_tensor)
                # 4、预测正确
                else:
                    self._copy_img_his_cam(labels_=labels_, pred=pred, label=label, scores=scores, img_p=img_p,
                                           dst_path=path_ok, cam_extractor=cam_extractor,
                                           output_tensor=output_tensor)

        logger.info("输出严格混淆矩阵")
        for p, t in pred_target_list:  # (int,str)
            confusion_matrix[int(t)][p] += 1
        self.plot_confusion_matrix(confusion_matrix, class_names, title="Confusion Matrix",
                              num_classes=self.num_class,
                              dst_path=os.path.join(output_dir, "validate", "cm.png"))

        logger.info("输出可容忍混淆矩阵")
        for k in list(pred_target_list_tolerate_count.keys()):
            if k in pred_target_list_tolerate_count.keys():
                p, t = k[0], k[1]  # int, int
                confusion_matrix[int(p)][p] += int(pred_target_list_tolerate_count[(p, t)])
                confusion_matrix[int(t)][p] -= int(pred_target_list_tolerate_count[(p, t)])
        self.plot_confusion_matrix(confusion_matrix, class_names, title="Tolerate Confusion Matrix",
                              num_classes=self.num_class,
                              dst_path=os.path.join(output_dir, "validate", "cm_tolerate.png"))

        # ----------------------------计算过检、漏检------------------------------------------------------------
        total_len = len(pred_target_list)
        gjc = 0  # 过检数量 pred NG -- target OK
        ljc = 0  # 漏检数量 pred OK -- target NG
        for pred_target in pred_target_list:  # pred_target_list.append((pred, label))  # 记录预测值和标签值，供计算混淆矩阵使用 (int,str)
            pred_, target_ = pred_target
            # 为每个预测和target标记OK或者NG
            pred_flag = "ok" if labels_[str(pred_)] in self.industry.kwargs.ok_ng_class.ok else "ng"
            target_flag = "ok" if labels_[str(target_)] in self.industry.kwargs.ok_ng_class.ok else "ng"
            if pred_flag == "ng" and target_flag == "ok":
                gjc += 1
            elif pred_flag == "ok" and target_flag == "ng":
                ljc += 1

        gjl = float(gjc) / total_len  # 过检率
        ljl = float(ljc) / total_len  # 漏检率

        logger.info("过检率：{}\n漏检率：{}\n".format(gjl, ljl))

    def _copy_img_his_cam(self, labels_=None, pred=None, label=None, scores=None,
                          img_p=None, dst_path=None, cam_extractor=None, output_tensor=None):
        # 1. 拷贝图片
        output_name = "pred-{}__target-{}__{}__score-{}.bmp".format(labels_[str(pred)], labels_[label],
                                                                    str(time.time()).split('.')[0] +
                                                                    str(time.time()).split('.')[1],
                                                                    str(scores))
        shutil.copy(os.path.join(img_p), os.path.join(dst_path, output_name))

        # 2. 拷贝直方图
        output_name = output_name[:-4] + ".jpg"
        img = np.array(Image.open(img_p))  # 原始图片img
        if len(img.shape) == 2:
            image = np.expand_dims(img, axis=2)  # 扩展图片image（224，224，1）
        rows, cols, channels = image.shape
        assert channels == 1
        flat_gray = image.reshape((cols * rows,)).tolist()
        A = min(flat_gray)
        B = max(flat_gray)
        image_his = np.uint8(255 / (B - A + 0.1) * (image - A) + 0.5)  # histogram图片 image_his
        cv2.imencode('.jpg', image_his)[1].tofile(os.path.join(dst_path, output_name))

        # # 3.拷贝CAM图
        # output_name = output_name[:-4] + ".png"
        #
        # activation_map = cam_extractor(output_tensor.cpu().squeeze(0).argmax().item(), output_tensor.cpu())
        # result = overlay_mask(Image.open(img_p).convert("RGB"), to_pil_image(activation_map, mode='F'), alpha=0.5)
        #
        # cv2.imencode('.png', np.array(result)[:, :, ::-1])[1].tofile(
        #     os.path.join(dst_path, output_name))

    def plot_confusion_matrix(self, cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues,
                              num_classes=38,
                              dst_path=None):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        cm = np.asarray(cm)
        length = 10 if num_classes < 10 else num_classes // 2
        fig = plt.figure(dpi=100, figsize=(length, length))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(dst_path)
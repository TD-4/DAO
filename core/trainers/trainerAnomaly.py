
# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.12.17
# @GitHub:https://github.com/felixfu520
# @Base On:

import os
import datetime
import numpy as np
from PIL import Image
from loguru import logger


import torch
import torch.nn.functional as F

from core.modules import Registers
from core.trainers.utils import setup_logger
from core.modules.dataloaders.augments import get_transformer


class AnomalyTrainer:
    def __init__(self, exp):
        self.exp = exp  # DotMap 格式 的配置文件
        self.start_time = datetime.datetime.now().strftime('%m-%d_%H-%M')   # 此次trainer的开始时间

    def train(self):
        self._before_train()
        self._train()
        self._after_train()

    def _before_train(self):
        self.output_dir = os.path.join(self.exp.trainer.log.log_dir, self.exp.name)
        setup_logger(self.output_dir, distributed_rank=0, filename=f"train_log.txt", mode="a")

        logger.warning("Anomaly Detection only supported one machine and one gpu !!!!")
        logger.info("....... Train Before, Setting something ...... ")

        logger.info("1. Logging Setting ...")
        logger.info(f"create log file {self.output_dir}/train_log.txt")  # log txt
        logger.info("exp value:{}".format(self.exp))

        logger.info("2. Model Setting ...")
        self.device = torch.device("cuda:{}".format(self.exp.envs.gpu.gpuid))
        self.model = Registers.anomaly_models.get(self.exp.model.type)(
            self.exp.model.backbone,
            device=self.device,
            **self.exp.model.kwargs)  # get model from register

        logger.info("3. Dataloader Setting ...")
        self.train_loader = Registers.dataloaders.get(self.exp.dataloader.type)(
            dataset=self.exp.dataloader.dataset,
            **self.exp.dataloader.kwargs)
        self.val_loader = Registers.dataloaders.get(self.exp.evaluator.type)(
            dataset=self.exp.evaluator.dataset,
            **self.exp.evaluator.kwargs)

    def _train(self):
        self.model.fit(self.train_loader, output_dir=self.output_dir)

    def _after_train(self):
        self.model.evaluate(self.val_loader, output_dir=self.output_dir)


class AnomalyEval:
    def __init__(self, exp):
        self.exp = exp  # DotMap 格式 的配置文件
        self.start_time = datetime.datetime.now().strftime('%m-%d_%H-%M')   # 此次trainer的开始时间

    def eval(self):
        self._before_eval()
        self._eval()
        self._after_eval()

    def _before_eval(self):
        self.output_dir = os.path.join(self.exp.trainer.log.log_dir, self.exp.name)
        setup_logger(self.output_dir, distributed_rank=0, filename=f"train_log.txt", mode="a")

        logger.warning("Anomaly Detection only supported one machine and one gpu !!!!")
        logger.info("....... Train Before, Setting something ...... ")

        logger.info("1. Logging Setting ...")
        logger.info(f"create log file {self.output_dir}/train_log.txt")  # log txt
        logger.info("exp value:{}".format(self.exp))

        logger.info("2. Model Setting ...")
        self.device = torch.device("cuda:{}".format(self.exp.envs.gpu.gpuid))
        self.model = Registers.anomaly_models.get(self.exp.model.type)(
            device=self.device, **self.exp.model.kwargs)  # get model from register

        logger.info("3. Dataloader Setting ...")
        self.train_loader = Registers.dataloaders.get(self.exp.dataloader.type)(
            dataset=self.exp.dataloader.dataset,
            **self.exp.dataloader.kwargs)
        self.val_loader = Registers.dataloaders.get(self.exp.evaluator.type)(
            dataset=self.exp.evaluator.dataset,
            **self.exp.evaluator.kwargs)

    def _eval(self):
        self.model.fit(self.train_loader, output_dir=self.output_dir)

    def _after_eval(self):
        self.model.evaluate(self.val_loader, output_dir=self.output_dir)


class AnomalyDemo:
    def __init__(self, exp):
        self.exp = exp  # DotMap 格式 的配置文件
        self.output_dir = os.path.join(self.exp.trainer.log.log_dir, self.exp.name)
        os.makedirs(self.output_dir, exist_ok=True)
        self.feature_extractor = self._get_model()
        self.images = self._get_images()  # ndarray [(image, shape, img_p),..., ]

    def _get_model(self):
        import timm
        feature_extractor = timm.create_model(
            self.exp.model.backbone.type,
            **self.exp.model.backbone.kwargs
        )
        for param in feature_extractor.parameters():
            param.requires_grad = False
        feature_extractor.eval()
        return feature_extractor

    def _img_ok(self, img_p):
        flag = False
        for m in self.exp.images.image_ext:
            if img_p.endswith(m):
                flag = True
        return flag

    def _get_images(self):
        results = []
        all_paths = []

        if self.exp.images.type == "image":
            all_paths.append(self.exp.images.path)
        elif self.exp.images.type == "images":
            all_p = [p for p in os.listdir(self.exp.images.path) if self._img_ok(p)]
            for p in all_p:
                all_paths.append(os.path.join(self.exp.images.path, p))

        for img_p in all_paths:
            image = np.array(Image.open(img_p))  # h,w
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=2)  # h,w,1
            shape = image.shape
            transform = get_transformer(self.exp.images.transforms.kwargs)
            image = transform(image=image)['image']
            image = image.transpose(2, 0, 1)  # c, h, w
            results.append((image, image.shape, img_p))
        return results

    def embedding_concat(self, x, y):
        B, C1, H1, W1 = x.size()  # (209,256,56,56)
        _, C2, H2, W2 = y.size()  # (209,512,28,28)
        s = int(H1 / H2)
        x = F.unfold(x, kernel_size=s, dilation=1, stride=s)  # (209, 1024, 784)
        x = x.view(B, C1, -1, H2, W2)  # torch.Size([209, 256, 4, 28, 28])
        z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)  # torch.Size([209, 768, 4, 28, 28])
        for i in range(x.size(2)):
            z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
        z = z.view(B, -1, H2 * W2)  # torch.Size([209, 3072, 784])
        z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)  # torch.Size([209, 768, 56, 56])

        return z

    def demo(self):
        import pickle
        from tqdm import tqdm
        from scipy.spatial.distance import mahalanobis
        from scipy.ndimage import gaussian_filter


        with open(self.exp.model.ckpt, 'rb') as f:
            train_output = pickle.load(f)

        for image, shape, img_p in self.images:
            image_ = image
            image = torch.tensor(image).unsqueeze(0)  # 1, c, h, w
            feature_map = self.feature_extractor(image)

            # 将feature_maps 转为 embedding_vectors: torch.Size([200, 1792, 56, 56])
            embedding_vector = feature_map[0]
            for layer in feature_map[1:]:
                embedding_vector = self.embedding_concat(embedding_vector, layer)

            # randomly select d dimension
            embedding_vectors = torch.index_select(embedding_vector, 1, train_output[2])

            # calculate distance matrix
            B, C, H, W = embedding_vectors.size()
            embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
            dist_list = []
            for i in tqdm(range(H * W), desc="Evaluate calculate cov::"):
                mean = train_output[0][:, i]
                conv_inv = np.linalg.inv(train_output[1][:, :, i])
                dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
                dist_list.append(dist)

            dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

            # upsample
            dist_list = torch.tensor(dist_list)  # torch.Size([49, 56, 56])
            score_map = F.interpolate(dist_list.unsqueeze(1), size=224, mode='bilinear',
                                      align_corners=False).squeeze().numpy()  # (49, 224, 224)

            # apply gaussian smoothing on the score map
            for i in range(score_map.shape[0]):
                score_map[i] = gaussian_filter(score_map[i], sigma=4)

            # Normalization
            max_score = score_map.max()
            min_score = score_map.min()
            scores = (score_map - min_score) / (max_score - min_score)  # (49, 224, 224)

            # 绘制每张test图片预测信息
            # test_imgs:[(3, 224, 224), ..., batchsize]
            # scores: (batchsize, 224, 224)
            # gt_mask_list: [(1, 224, 224), ..., batchsize]
            # threshold: float
            # save_dir: str
            # test_imgs_path: [img_path, ..., batchsize]
            self.plot_fig(image_, scores, img_p)

    def denormalization(self, x):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)

        return x

    def plot_fig(self, test_img, scores, img_p):
        import matplotlib.pyplot as plt
        import matplotlib
        from skimage import morphology
        from skimage.segmentation import mark_boundaries

        vmax = scores.max() * 255.
        vmin = scores.min() * 255.

        img = self.denormalization(test_img)
        heat_map = scores * 255
        mask = scores
        mask[mask > np.mean(scores)] = 1
        mask[mask <= np.mean(scores)] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 4, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
            ax_img[0].imshow(img)
            ax_img[0].title.set_text('Image')
            ax = ax_img[1].imshow(heat_map, cmap='jet', norm=norm)
            ax_img[1].imshow(img, cmap='gray', interpolation='none')
            ax_img[1].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
            ax_img[1].title.set_text('Predicted heat map')
            ax_img[2].imshow(mask, cmap='gray')
            ax_img[2].title.set_text('Predicted mask')
            ax_img[3].imshow(vis_img)
            ax_img[3].title.set_text('Segmentation result')
            left = 0.92
            bottom = 0.15
            width = 0.015
            height = 1 - 2 * bottom
            rect = [left, bottom, width, height]
            cbar_ax = fig_img.add_axes(rect)
            cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
            cb.ax.tick_params(labelsize=8)
            font = {
                'family': 'serif',
                'color': 'black',
                'weight': 'normal',
                'size': 8,
            }
            cb.set_label('Anomaly Score', fontdict=font)

            fig_img.savefig(os.path.join(self.output_dir, img_p.split("/")[-1]), dpi=100)
            plt.close()


class AnomalyExport:
    def __init__(self, exp):
        self.exp = exp  # DotMap 格式 的配置文件
        self.output_dir = os.path.join(self.exp.onnx.onnx_path, self.exp.name)
        os.makedirs(self.output_dir, exist_ok=True)
        self.feature_extractor = self._get_model()

    def _get_model(self):
        import timm
        feature_extractor = timm.create_model(
            self.exp.model.backbone.type,
            **self.exp.model.backbone.kwargs
        )
        for param in feature_extractor.parameters():
            param.requires_grad = False
        feature_extractor.eval()
        return feature_extractor

    @logger.catch
    def export(self):
        x = torch.randn(self.exp.onnx.x_size)
        onnx_path = os.path.join(self.output_dir, "export.onnx")
        torch.onnx.export(self.feature_extractor,
                          x,
                          onnx_path,
                          **self.exp.onnx.kwargs)

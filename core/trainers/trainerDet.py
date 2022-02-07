#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.12.17
# @GitHub:https://github.com/felixfu520
# @Copy From:https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/core/trainer.py

import os
import time
import random
import datetime
import numpy as np
from PIL import Image
from loguru import logger

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from core.utils import get_rank, get_local_rank, get_world_size, all_reduce_norm, synchronize
from core.trainers.utils import setup_logger, load_ckpt, save_checkpoint, occupy_mem, ModelEMA, is_parallel, gpu_mem_usage, get_palette, colorize_mask
from core.modules.utils import MeterDetTrain, get_model_info, MeterBuffer
from core.modules.dataloaders.utils.data_prefetcher import DataPrefetcherDet
from core.modules.dataloaders.augments import get_transformer

from core.modules.register import Registers


@Registers.trainers.register
class DetTrainer:
    def __init__(self, exp):
        self.exp = exp  # DotMap 格式 的配置文件
        self.start_time = datetime.datetime.now().strftime('%m-%d_%H-%M')   # 此次trainer的开始时间
        self.input_size = self.exp.dataloader.dataset.dataset1.kwargs.input_size
        # Actual multiscale ranges: [640-5*32, 640+5*32].
        # To disable multiscale training, set the
        # self.multiscale_range to 0.
        self.multiscale_range = self.exp.trainer.multiscale_range
        self.data_type = torch.float16 if self.exp.trainer.amp else torch.float32
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.exp.trainer.amp)  # 在训练开始之前实例化一个Grad Scaler对象

    def train(self):
        self._before_train()
        # epochs
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self._before_epoch()
            # iters
            for self.iter in range(self.max_iter):
                self._before_iter()
                self._train_one_iter()
                self._after_iter()
            self._after_epoch()
        self._after_train()

    def _before_train(self):
        """
        1.Logger Setting
        2.Model Setting;
        3.Optimizer Setting;
        4.Resume setting;
        5.DataLoader Setting;
        6.Loss Setting;
        7.Scheduler Setting;
        8.Evaluator Setting;
        """
        self.output_dir = os.getcwd() if self.exp.trainer.log.log_dir is None else \
            os.path.join(self.exp.trainer.log.log_dir, self.exp.name, self.start_time)
        setup_logger(self.output_dir, distributed_rank=get_rank(), filename=f"train_log.txt", mode="a")
        logger.info("....... Train Before, Setting something ...... ")
        logger.info("1. Logging Setting ...")
        logger.info(f"create log file {self.output_dir}/train_log.txt")  # log txt
        logger.info("exp value:\n{}".format(self.exp))
        logger.info(f"create Tensorboard logger {self.output_dir}")
        if get_rank() == 0:
            self.tblogger = SummaryWriter(self.output_dir)  # log tensorboard

        logger.info("2. Model Setting ...")
        torch.cuda.set_device(get_local_rank())
        model = Registers.det_models.get(self.exp.model.type)(**self.exp.model.kwargs)  # get model from register
        logger.info("\n{}".format(model))   # log model structure
        # TODO 使用summary报错，所以修改为使用get_model_info
        # summary(model, input_size=tuple(self.exp.model.summary_size), device="{}".format(next(model.parameters()).device))  # log torchsummary model
        # logger.info(
        #     "Model Summary: {}".format(get_model_info(model, self.exp.model.summary_size[1:]))
        # )
        model.to("cuda:{}".format(get_local_rank()))    # model to self.device

        logger.info("3. Optimizer Setting")
        self.optimizer = Registers.optims.get(self.exp.optimizer.type)(model=model, **self.exp.optimizer.kwargs)

        logger.info("4. Resume/FineTuning Setting ...")
        model = self._resume_train(model)

        logger.info("5. Dataloader Setting ... ")
        self.max_epoch = self.exp.trainer.max_epochs
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.trainer.no_aug_epochs
        self.train_loader = Registers.dataloaders.get(self.exp.dataloader.type)(
            is_distributed=get_world_size() > 1,
            dataset=self.exp.dataloader.dataset,
            seed=self.exp.seed,
            no_aug=self.no_aug,
            **self.exp.dataloader.kwargs
        )
        logger.info("init prefetcher, this might take one minute or less...")
        self.prefetcher = DataPrefetcherDet(self.train_loader)
        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)

        logger.info("6. Loss Setting ... - no loss setting ,loss in model")
        # self.loss = Registers.losses.get(self.exp.loss.type)(**self.exp.loss.kwargs)
        # self.loss.to(device="cuda:{}".format(get_local_rank()))

        logger.info("7. Scheduler Setting ... ")
        self.lr_scheduler = Registers.schedulers.get(self.exp.lr_scheduler.type)(
            lr=self.exp.optimizer.kwargs.lr,
            iters_per_epoch=self.max_iter,
            total_epochs=self.exp.trainer.max_epochs,
            **self.exp.lr_scheduler.kwargs
        )

        logger.info("8. Other Setting ... ")
        logger.info("occupy mem")
        if self.exp.trainer.occupy:
            occupy_mem(get_local_rank())

        logger.info("Model DDP Setting")
        if get_world_size() > 1:
            model = DDP(model, device_ids=[get_local_rank()], broadcast_buffers=False, output_device=[get_local_rank()])

        logger.info("Model EMA Setting")
        # Exponential moving average
        # 用EMA方法对模型的参数做平均，以提高测试指标并增加模型鲁棒性（减少模型权重抖动）
        self.use_model_ema = self.exp.trainer.ema
        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.model = model
        self.model.train()

        logger.info("8. Evaluator Setting ... ")
        self.evaluator = Registers.evaluators.get(self.exp.evaluator.type)(
            is_distributed=get_world_size() > 1,
            dataloader=self.exp.evaluator.dataloader,
            num_classes=self.exp.model.kwargs.head.num_classes,
            **self.exp.evaluator.kwargs
        )
        self.train_metrics = MeterBuffer(window_size=self.exp.trainer.log.log_per_iter)
        self.best_acc = 0
        logger.info("Now Training Start ......")

    def _before_epoch(self):
        """
        判断此次epoch是否需要马赛克增强
        :return:
        """
        logger.info("---> start train epoch{}".format(self.epoch + 1))
        # self.train_loader.close_mosaic()

        if self.epoch + 1 == self.max_epoch - self.exp.trainer.no_aug_epochs or self.no_aug:
            logger.info("--->No mosaic aug now!")
            self.train_loader.close_mosaic()
            logger.info("--->Add additional L1 loss now!")
            if get_world_size() > 1:
                self.model.module.head.use_l1 = True
            else:
                self.model.head.use_l1 = True
            self.exp.trainer.log.eval_interval = 1
            if not self.no_aug:
                self._save_ckpt(ckpt_name="last_mosaic_epoch")

    def _before_iter(self):
        pass

    def _train_one_iter(self):
        iter_start_time = time.time()

        inps, targets, img_info, next_ids = self.prefetcher.next()
        # print("----{}".format(next_ids))
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        inps, targets = self.preprocess(inps, targets, self.exp.dataloader.dataset.dataset1.kwargs.input_size)
        data_end_time = time.time()

        with torch.cuda.amp.autocast(enabled=self.exp.trainer.amp):    # 开启auto cast的context manager语义（model+loss）
            outputs = self.model(inps, targets)

        loss = outputs["total_loss"]

        self.optimizer.zero_grad()  # 梯度清零
        self.scaler.scale(loss).backward()  # 反向传播；Scales loss. 为了梯度放大
        # scaler.step() 首先把梯度的值unscale回来.
        # 如果梯度的值不是infs或者NaNs, 那么调用optimizer.step()来更新权重,
        # 否则，忽略step调用，从而保证权重不更新（不被破坏）
        self.scaler.step(self.optimizer)  # optimizer.step 进行参数更新
        self.scaler.update()  # 准备着，看是否要增大scaler

        if self.use_model_ema:
            self.ema_model.update(self.model)

        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        iter_end_time = time.time()
        self.train_metrics.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )

    def _after_iter(self):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        if (self.iter + 1) % self.exp.trainer.log.log_per_iter == 0 and get_rank() == 0:
            # TODO check ETA logic
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.train_metrics["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = f"epoch: {self.epoch + 1}/{self.max_epoch}, iter: {self.iter + 1}/{self.max_iter} "
            loss_meter = self.train_metrics.get_filtered_meter("loss")
            loss_str = ", ".join(["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()])
            time_meter = self.train_metrics.get_filtered_meter("time")
            time_str = ", ".join(["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()])

            logger.info(
                "{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    gpu_mem_usage(),
                    time_str,
                    loss_str,
                    self.train_metrics["lr"].latest,
                )
                + (", size: {:d}, {}".format(self.input_size[0], eta_str))
            )
            tensorboard_losses = [
                {i:loss_meter[i].global_avg.cpu().numpy().item()
                if isinstance(loss_meter[i].global_avg, torch.Tensor)
                else loss_meter[i].global_avg}
                for i in [loss for loss in loss_meter]
            ]
            for tensorboard_loss in tensorboard_losses:
                key = list(tensorboard_loss)[0]
                value = tensorboard_loss[list(tensorboard_loss)[0]]
                self.tblogger.add_scalar('train/{}'.format(key), value, self.progress_in_iter)
            self.tblogger.add_scalar('train/lr', self.train_metrics["lr"].latest, self.progress_in_iter)
            self.train_metrics.clear_meters()

        # random resizing
        # 每隔一定间隔，改变输出图片尺寸，并且保证多卡之间的图片尺寸相同
        # if (self.progress_in_iter + 1) % 10 == 0:
        #     self.input_size = self.random_resize(
        #         self.train_loader, self.epoch, get_rank(), get_world_size() > 1
        #     )

    def _after_epoch(self):
        self._save_ckpt(ckpt_name="latest")

        if (self.epoch + 1) % self.exp.trainer.log.eval_interval == 0:
            all_reduce_norm(self.model)
            self._evaluate_and_save_model()

    def _after_train(self):
        logger.info(
            "Training of experiment is done and the best AP is {:.2f}".format(self.best_acc)
        )

    def _evaluate_and_save_model(self):
        if self.use_model_ema:
            evalmodel = self.ema_model.ema
        else:
            evalmodel = self.model
            if is_parallel(evalmodel):
                evalmodel = evalmodel.module

        ap50_95, ap50, summary = self.evaluator.evaluate(evalmodel, get_world_size() > 1)
        self.model.train()
        if get_rank() == 0:
            self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
            self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)
            logger.info("\n" + summary)

        synchronize()
        self._save_ckpt("last_epoch", ap50_95 > self.best_acc)
        self.best_acc = max(self.best_acc, ap50_95)

    def _save_ckpt(self, ckpt_name, update_best_ckpt=False):
        if get_rank() == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model
            logger.info("Save weights to {}".format(self.output_dir))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.output_dir,
                ckpt_name,
            )

    def _resume_train(self, model):
        """
        如果args.resume为true，将args.ckpt权重resume；
        如果args.resume为false，将args.ckpt权重fine turning；
        :param model:
        :return:
        """
        if self.exp.trainer.resume:
            logger.info("resume training")
            # 获取ckpt路径
            assert self.exp.trainer.ckpt is not None
            ckpt_file = self.exp.trainer.ckpt
            # 加载ckpt
            ckpt = torch.load(ckpt_file, map_location="cuda:{}".format(get_local_rank()))
            # resume the model/optimizer state dict
            model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            # resume the training states variables
            self.start_epoch = ckpt["start_epoch"]
            logger.info(
                "loaded checkpoint '{}' (epoch {})".format(self.exp.trainer.ckpt, self.start_epoch)
            )  # noqa
        else:
            if self.exp.trainer.ckpt is not None:
                logger.info("loading checkpoint for fine tuning")
                ckpt_file = self.exp.trainer.ckpt
                ckpt = torch.load(ckpt_file, map_location="cuda:{}".format(get_local_rank()))["model"]
                model = load_ckpt(model, ckpt)
            self.start_epoch = 0

        return model

    # 随机改变输出图片尺寸
    def random_resize(self, data_loader, epoch, rank, is_distributed):
        tensor = torch.LongTensor(2).cuda()

        if rank == 0:
            # 随机采样一个新的size
            size_factor = self.input_size[1] * 1.0 / self.input_size[0]
            if not hasattr(self, 'random_size'):
                min_size = int(self.input_size[0] / 32) - self.multiscale_range
                max_size = int(self.input_size[0] / 32) + self.multiscale_range
                self.random_size = (min_size, max_size)
            size = random.randint(*self.random_size)
            size = (int(32 * size), 32 * int(size * size_factor))
            tensor[0] = size[0]
            tensor[1] = size[1]

        if is_distributed:
            dist.barrier()
            dist.broadcast(tensor, 0)  # 广播到其余卡，实现不同卡间的输入图片尺寸相同

        # 改变图片size
        input_size = (tensor[0].item(), tensor[1].item())
        return input_size

    def preprocess(self, inputs, targets, tsize):
        scale_y = tsize[0] / self.input_size[0]
        scale_x = tsize[1] / self.input_size[1]
        if scale_x != 1 or scale_y != 1:
            inputs = nn.functional.interpolate(
                inputs, size=tsize, mode="bilinear", align_corners=False
            )
            targets[..., 1::2] = targets[..., 1::2] * scale_x
            targets[..., 2::2] = targets[..., 2::2] * scale_y
        return inputs, targets

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  # apply decay

            # 优化器 SGD + nesterov + momentum
            optimizer = torch.optim.SGD(
                pg0, lr=lr, momentum=self.momentum, nesterov=True
            )
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})
            self.optimizer = optimizer

        return self.optimizer

    def get_lr_scheduler(self, lr, iters_per_epoch):
        from yolox.utils import LRScheduler

        scheduler = LRScheduler(
            self.scheduler,
            lr,
            iters_per_epoch,
            self.max_epoch,
            warmup_epochs=self.warmup_epochs,
            warmup_lr_start=self.warmup_lr,
            no_aug_epochs=self.no_aug_epochs,
            min_lr_ratio=self.min_lr_ratio,
        )
        return scheduler

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import COCODataset, ValTransform

        valdataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann if not testdev else "image_info_test-dev2017.json",
            name="val2017" if not testdev else "test2017",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator

    def eval(self, model, evaluator, is_distributed, half=False):
        return evaluator.evaluate(model, is_distributed, half)
    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter


@Registers.trainers.register
class DetEval:
    def __init__(self, exp):
        self.exp = exp  # DotMap 格式 的配置文件
        self.start_time = datetime.datetime.now().strftime('%m-%d_%H-%M')   # 此次trainer的开始时间

    def eval(self):
        self._before_eval()
        self.evaluator.evaluate(self.model, get_world_size() > 1,
                                                               device="cuda:{}".format(get_local_rank()),
                                                               output_dir=self.output_dir)

    def _before_eval(self):
        """
        1.Logger Setting
        2.Model Setting;
        3.Evaluator Setting;
        """
        self.output_dir = os.getcwd() if self.exp.trainer.log.log_dir is None else \
            os.path.join(self.exp.trainer.log.log_dir, self.exp.name, self.start_time)
        setup_logger(self.output_dir, distributed_rank=get_rank(), filename=f"val_log.txt", mode="a")
        logger.info("....... Train Before, Setting something ...... ")
        logger.info("1. Logging Setting ...")
        logger.info(f"create log file {self.output_dir}/train_log.txt")  # log txt
        logger.info("exp value:\n{}".format(self.exp))
        logger.info(f"create Tensorboard logger {self.output_dir}")

        logger.info("2. Model Setting ...")
        torch.cuda.set_device(get_local_rank())
        model = Registers.cls_models.get(self.exp.model.type)(**self.exp.model.kwargs)  # get model from register
        logger.info("\n{}".format(model))  # log model structure
        summary(model, input_size=tuple(self.exp.model.summary_size),
                device="{}".format(next(model.parameters()).device))  # log torchsummary model
        model.to("cuda:{}".format(get_local_rank()))  # model to self.device

        ckpt_file = self.exp.trainer.ckpt
        ckpt = torch.load(ckpt_file, map_location="cuda:{}".format(get_local_rank()))["model"]
        model = load_ckpt(model, ckpt)

        logger.info("Model DDP Setting")
        if get_world_size() > 1:
            model = DDP(model, device_ids=[get_local_rank()], broadcast_buffers=False, output_device=[get_local_rank()])

        self.model = model
        self.model.eval()

        logger.info("8. Evaluator Setting ... ")
        self.evaluator = Registers.evaluators.get(self.exp.evaluator.type)(
            is_distributed=get_world_size() > 1,
            dataloader=self.exp.evaluator.dataloader,
            num_classes=self.exp.model.kwargs.num_classes,
            industry=self.exp.evaluator.industry,
            **self.exp.evaluator.kwargs
        )
        self.train_metrics = MeterSegTrain(self.exp.model.kwargs.num_classes)
        logger.info("Now Eval Start ......")


@Registers.trainers.register
class DetDemo:
    def __init__(self, exp):
        self.exp = exp  # DotMap 格式 的配置文件
        self.start_time = datetime.datetime.now().strftime('%m-%d_%H-%M')  # 此次trainer的开始时间

        self.model = self._get_model()
        self.images = self._get_images()  # ndarray

    def _get_model(self):
        logger.info("model setting, on cpu")
        model = Registers.seg_models.get(self.exp.model.type)(**self.exp.model.kwargs)  # get model from register
        logger.info("\n{}".format(model))  # log model structure
        summary(model, input_size=tuple(self.exp.model.summary_size),
                device="{}".format(next(model.parameters()).device))  # log torchsummary model
        ckpt = torch.load(self.exp.model.ckpt, map_location="cpu")["model"]
        model = load_ckpt(model, ckpt)
        model.eval()
        return model

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
            results.append((image, shape, img_p))
        return results

    def demo(self):
        results = []
        for image, shape, img_p in self.images:
            image = torch.tensor(image).unsqueeze(0)  # 1, c, h, w
            output = self.model(image)
            output = np.uint8(output.data.max(1)[1].cpu().numpy()[0])
            output = colorize_mask(output, get_palette(self.exp.model.kwargs.num_classes))
            output = output.resize((shape[1], shape[0]))
            results.append((output, img_p))
        return results


@Registers.trainers.register
class DetExport:
    def __init__(self, exp):
        self.exp = exp  # DotMap 格式 的配置文件
        self.start_time = datetime.datetime.now().strftime('%m-%d_%H-%M')  # 此次trainer的开始时间
        self.model = self._get_model()

    def _get_model(self):
        logger.info("model setting, on cpu")
        model = Registers.seg_models.get(self.exp.model.type)(**self.exp.model.kwargs)  # get model from register
        logger.info("\n{}".format(model))  # log model structure
        summary(model, input_size=tuple(self.exp.model.summary_size),
                device="{}".format(next(model.parameters()).device))  # log torchsummary model
        ckpt = torch.load(self.exp.model.ckpt, map_location="cpu")["model"]
        model = load_ckpt(model, ckpt)
        model.eval()
        return model

    @logger.catch
    def export(self):
        x = torch.randn(self.exp.onnx.x_size)
        onnx_path = self.exp.onnx.onnx_path
        torch.onnx.export(self.model,
                          x,
                          onnx_path,
                          **self.exp.onnx.kwargs)


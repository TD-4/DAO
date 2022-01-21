
# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.12.17
# @GitHub:https://github.com/felixfu520
# @Copy From:

import os
import time
import datetime
import numpy as np
from PIL import Image
from loguru import logger

import torch
from torchsummary import summary
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from core.utils import get_rank, get_local_rank, get_world_size, all_reduce_norm, synchronize
from core.trainers.utils import (
    setup_logger, load_ckpt, save_checkpoint, occupy_mem,
    ModelEMA, EMA, is_parallel, gpu_mem_usage,
    get_palette, colorize_mask
)
from core.modules.utils import MeterSegTrain, denormalization
from core.modules.dataloaders.augments import get_transformer
from core.modules.register import Registers


@Registers.trainers.register
class SegTrainer:
    def __init__(self, exp):
        self.exp = exp  # DotMap 格式 的配置文件
        self.start_time = datetime.datetime.now().strftime('%m-%d_%H-%M')   # 此次trainer的开始时间

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
        model = Registers.seg_models.get(self.exp.model.type)(self.exp.model.backbone, **self.exp.model.kwargs)
        logger.info("\n{}".format(model))   # log model structure
        summary(model, input_size=tuple(self.exp.model.summary_size), device="{}".format(next(model.parameters()).device))
        model.to("cuda:{}".format(get_local_rank()))    # model to self.device

        logger.info("3. Optimizer Setting")
        self.optimizer = Registers.optims.get(self.exp.optimizer.type)(model=model, **self.exp.optimizer.kwargs)

        logger.info("4. Resume/FineTuning Setting ...")
        model = self._resume_train(model)

        logger.info("5. Dataloader Setting ... ")
        self.max_epoch = self.exp.trainer.max_epochs
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.trainer.no_aug_epochs
        self.train_loader, self.max_iter = Registers.dataloaders.get(self.exp.dataloader.type)(
            is_distributed=get_world_size() > 1,
            dataset=self.exp.dataloader.dataset,
            seed=self.exp.seed,
            **self.exp.dataloader.kwargs
        )

        logger.info("6. Loss Setting ... ")
        self.loss = Registers.losses.get(self.exp.loss.type)(**self.exp.loss.kwargs)
        self.loss.to(device="cuda:{}".format(get_local_rank()))
        if "aux_params" in self.exp.model.kwargs:
            self.loss_aux = Registers.losses.get(self.exp.aux_loss.type)(**self.exp.aux_loss.kwargs)
            self.loss_aux.to(device="cuda:{}".format(get_local_rank()))

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
            # self.ema_model = ModelEMA(model, 0.9998)
            # self.ema_model.updates = self.max_iter * self.start_epoch
            self.ema_model = EMA(model, 0.9998)
            self.ema_model.register()

        self.model = model
        self.model.train()

        logger.info("9. Evaluator Setting ... ")
        self.evaluator = Registers.evaluators.get(self.exp.evaluator.type)(
            is_distributed=get_world_size() > 1,
            dataloader=self.exp.evaluator.dataloader,
            num_classes=self.exp.model.kwargs.num_classes,
        )
        self.train_metrics = MeterSegTrain()
        self.best_acc = 0
        logger.info("Now Training Start ......")

    def _before_epoch(self):
        """
        每次epoch前的操作，例如multi-scale， 马赛克增强，等取消与否。
        :return:
        """
        logger.info("------------------> start train epoch{}".format(self.epoch + 1))

    def _before_iter(self):
        pass

    def _train_one_iter(self):
        iter_start_time = time.time()

        inps, targets, path = self.train_loader.next()
        # show img and mask
        # Image.fromarray(denormalization(inps[0].cpu().numpy(), [0.398993, 0.431193, 0.452234],
        #                                 [0.285205, 0.273126, 0.276610])).save("/root/code/t.png")
        # import cv2;cv2.imwrite("/root/code/t3.png",
        #             np.expand_dims(np.where(targets[0].cpu().numpy() == 1, 255, 0).astype(np.uint8), axis=2))
        inps = inps.to(self.data_type)
        # targets = targets.to(self.data_type)
        targets.requires_grad = False
        data_end_time = time.time()

        with torch.cuda.amp.autocast(enabled=self.exp.trainer.amp):    # 开启auto cast的context manager语义（model+loss）
            outputs = self.model(inps)
            if "aux_params" in self.exp.model.kwargs:
                loss = self.loss(outputs[0], targets)  # PSP(master_branch)损失， 其他类似
                loss += self.loss_aux(outputs[1], targets) * 0.4  # FCN损失
            else:
                loss = self.loss(outputs, targets)
        self.optimizer.zero_grad()   # 梯度清零
        self.scaler.scale(loss).backward()   # 反向传播；Scales loss. 为了梯度放大
        # scaler.step() 首先把梯度的值unscale回来.
        # 如果梯度的值不是infs或者NaNs, 那么调用optimizer.step()来更新权重,
        # 否则，忽略step调用，从而保证权重不更新（不被破坏）
        self.scaler.step(self.optimizer)    # optimizer.step 进行参数更新
        self.scaler.update()    # 准备着，看是否要增大scaler

        if self.use_model_ema:
            # self.ema_model.update(self.model)
            self.ema_model.update()
        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        iter_end_time = time.time()
        self.train_metrics.update_metrics(
            data_time=data_end_time - iter_start_time,
            batch_time=iter_end_time - iter_start_time,
            total_loss=loss.item(),
            lr=lr
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
            eta_seconds = (self.train_metrics.batch_time.avg + self.train_metrics.data_time.avg) * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = f"epoch: {self.epoch + 1}/{self.max_epoch}, iter: {self.iter + 1}/{self.max_iter} "
            loss_str = "loss:{:2f}".format(self.train_metrics.total_loss.avg)
            time_str = "iter time:{:2f}, data time:{:2f}".format(self.train_metrics.batch_time.avg, self.train_metrics.data_time.avg)

            logger.info(
                "{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}, {}".format(
                    progress_str,
                    gpu_mem_usage(),
                    time_str,
                    loss_str,
                    self.train_metrics.lr,
                    eta_str
                )
            )
            self.tblogger.add_scalar('train/loss', self.train_metrics.total_loss.avg, self.progress_in_iter)
            self.tblogger.add_scalar('train/lr', self.train_metrics.lr, self.progress_in_iter)
            self.train_metrics.reset_metrics()

    def _after_epoch(self):
        self._save_ckpt(ckpt_name="latest")

        if (self.epoch + 1) % self.exp.trainer.log.eval_interval == 0:
            all_reduce_norm(self.model)
            self._evaluate_and_save_model()

    def _after_train(self):
        logger.info("Training of experiment is done and the best Acc is {:.2f}".format(self.best_acc))

    def _evaluate_and_save_model(self):
        if self.use_model_ema:
            # evalmodel = self.ema_model.ema
            evalmodel = self.ema_model.model
        else:
            evalmodel = self.model
            if is_parallel(evalmodel):
                evalmodel = evalmodel.module

        pixAcc, mIoU, Class_IoU = self.evaluator.evaluate(evalmodel, get_world_size() > 1, device="cuda:{}".format(get_local_rank()))
        self.model.train()
        logger.info("pixAcc:{}, mIoU:{}, Class_IoU:{}".format(pixAcc, mIoU, Class_IoU))

        if get_rank() == 0:
            self.tblogger.add_scalar("val/pixAcc", pixAcc, self.epoch + 1)
            self.tblogger.add_scalar("val/mIoU", mIoU, self.epoch + 1)
            for k, v in Class_IoU.items():
                self.tblogger.add_scalar("val_detail/{} IoU".format(k), v, self.epoch + 1)

        synchronize()
        self._save_ckpt("last_epoch", mIoU > self.best_acc)
        self.best_acc = max(self.best_acc, mIoU)

    def _save_ckpt(self, ckpt_name, update_best_ckpt=False):
        if get_rank() == 0:
            # save_model = self.ema_model.ema if self.use_model_ema else self.model
            save_model = self.ema_model.model if self.use_model_ema else self.model
            logger.info("Save weights to {} - update_best_ckpt:{}".format(self.output_dir, update_best_ckpt))
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

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter


@Registers.trainers.register
class SegEval:
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
class SegDemo:
    def __init__(self, exp):
        self.exp = exp  # DotMap 格式 的配置文件
        self.start_time = datetime.datetime.now().strftime('%m-%d_%H-%M')  # 此次trainer的开始时间

        self.model = self._get_model()
        self.images = self._get_images()  # ndarray

    def _get_model(self):
        logger.info("model setting, on cpu")
        model = Registers.seg_models.get(self.exp.model.type)(self.exp.model.backbone, **self.exp.model.kwargs)  # get model from register
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
        for i, (image, img_p) in enumerate(results):
            image.save(img_p[:-4]+".png")
        return results


@Registers.trainers.register
class SegExport:
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


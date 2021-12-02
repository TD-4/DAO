# AI+工业

[TOC]

## 1. 项目由来

在**工业流程**中，深度学习应用过程包括：

- Train/Val(针对特定场景，特定数据集训练一个模型)
- Eval(使用测试集测试，得到工业上的性能指标)
- Demo(将模型做个demo给客户展示)
- Export(将模型转成其他格式)
- Deploy(将模型部署到具体的设备上，并权衡速度与准确率)
- APP(将整个工业流程封装成带界面的APP)

在**深度学习训练**中，训练过程包括：

- multigpus(是否使用多GPU形式训练)-->utils
- mixedprecisions(是否使用混合精度进行训练)-->utils
- loggers(训练过程日志输出)-->utils
  - tensorboard
  - log文字日志
- dataloaders(数据加载)-->dataloaders**(1)**
- models(模型：分类、分割、检测等)-->models**(2)**
- losses(损失函数)-->losses**(3)**
- optims(优化器)-->optims**(4)**
- schedulers(学习率调整策略)-->schedulers**(5)**
- metrics(训练过程中的性能评价)-->utils
- utils(其他工具单元)-->utils
- trainers(训练过程)-->trainers**(6)**
  - resume
  - fune turning
  - 日志监控
  - 权重输出
  - ...



目前，我所遇到的深度学习项目基本都能用这两个维度概括，为了方便以后使用，在此将两个维度整理成这个项目，将**工业流程（train、eval、demo、export、deploy的调用API）**封装到tools库中，将**深度学习训练(train & eval核心过程)**封装到core中，将**界面UI部分**分装成app包。

## 2. 目录结构

### 2.1 项目结构

```tree
AI/
├── app
│   └── __init__.py
├── configs
│   ├── det_yolox_coco_default_sgd_yoloxwarmcos_trainval_ubuntu20.04.json
│   ├── det_yolox_voc_default_sgd_yoloxwarmcos_trainval_ubuntu20.04.json
│   └── README.md
├── core
│   ├── dataloaders
│   │   ├── detection
│   │   │   ├── data_augment.py
│   │   │   ├── dataloading.py
│   │   │   ├── data_prefetcher.py
│   │   │   ├── datasets
│   │   │   │   ├── coco_classes.py
│   │   │   │   ├── coco.py
│   │   │   │   ├── datasets_wrapper.py
│   │   │   │   ├── __init__.py
│   │   │   │   ├── mosaicdetection.py
│   │   │   │   ├── voc_classes.py
│   │   │   │   └── voc.py
│   │   │   ├── __init__.py
│   │   │   └── samplers.py
│   │   ├── __init__.py
│   ├── evaluators
│   │   ├── detection
│   │   │   ├── coco_evaluator.py
│   │   │   ├── __init__.py
│   │   │   ├── voc_eval.py
│   │   │   └── voc_evaluator.py
│   │   ├── __init__.py
│   ├── experiments(实验配置包)
│   │   ├── base_exp.py:所有实验的基类
│   │   ├── build.py:通过configs.json生成Exp
│   │   ├── __init__.py
│   │   └── yolox.py:Yolox模型Exp, 实验中所需要的dataloader,model,...等都从此类中获取
│   ├── __init__.py
│   ├── losses
│   │   ├── CrossEntropyLoss2d.py
│   │   └── __init__.py
│   ├── models
│   │   ├── backbone
│   │   │   ├── efficientnet_pytorch
│   │   │   │   ├── __init__.py
│   │   │   │   ├── model.py
│   │   │   │   └── utils.py
│   │   │   └── __init__.py
│   │   ├── classification
│   │   │   ├── EfficientNet.py
│   │   │   └── __init__.py
│   │   ├── detection
│   │   │   ├── __init__.py
│   │   │   └── yolox
│   │   │       ├── darknet.py
│   │   │       ├── __init__.py
│   │   │       ├── losses.py
│   │   │       ├── network_blocks.py
│   │   │       ├── yolo_fpn.py
│   │   │       ├── yolo_head.py
│   │   │       ├── yolo_pafpn.py
│   │   │       └── yolox.py
│   │   ├── __init__.py
│   │   └── segmentation
│   │       └── __init__.py
│   ├── optims
│   │   ├── __init__.py
│   ├── schedulers
│   │   ├── __init__.py
│   │   ├── lr_scheduler.py
│   ├── trainers
│   │   ├── __init__.py
│   │   ├── launch.py: 所有试验的启动文件
│   │   ├── trainerCls.py: 分类的trainer过程
│   │   ├── trainerDet.py: 目标检测的trainer过程
│   │   └── trainerSeg.py: 分割的trainer过程
│   ├── utils
│   │   ├── allreduce_norm.py: 多卡同步操作
│   │   ├── boxes.py
│   │   ├── checkpoint.py
│   │   ├── dist.py:multi-gpu communication.
│   │   ├── ema.py
│   │   ├── __init__.py
│   │   ├── logger.py: 日志设置（重定向，存储等操作）
│   │   ├── metricDet.py
│   │   ├── model_utils.py
│   │   ├── setup_env.py: 环境变量设置，cv2多线程读图，nccl环境配置，...
│   │   └── visualize.py
│   └── version.py
├── datasets
|   ├── COCO
|   | 	└── annotations
|   |   ├── test2017
|   |   ├── train2017
|   |   └── val2017
|   └── VOCdevkit/
|   |   ├── VOC2007
|   |   │   ├── Annotations
|   |   │   ├── ImageSets
|   |   │   │   ├── Layout
|   |   │   │   ├── Main
|   |   │   │   └── Segmentation
|   |   │   ├── JPEGImages
|   |   │   ├── SegmentationClass
|   |   │   └── SegmentationObject
|   |   └── VOC2012
|   |       ├── Annotations
|   |       ├── ImageSets
|   |       │   ├── Action
|   |       │   ├── Layout
|   |       │   ├── Main
|   |       │   └── Segmentation
|   |       ├── JPEGImages
|   |       ├── SegmentationClass
|   |       └── SegmentationObject
└── tools
|   ├── demo
|   │   ├── demo.py
|   ├── eval
|   │   └── eval.py
|   ├── __init__.py
|   └── trainval
|       └── trainval.py:训练接口
├── LICENSE
├── README.md
├── requirements.txt
├── saved
├── setup.cfg
└── setup.py

```
### 2.2 trainerX.py代码过程

`训练过程中，所需要的信息和内容，都从Exp(eg. yolox文件)中获得`

```
|- trainerX
|	|- before_train(train之前的操作，eg. dataloader,model,optim,... setting)
|	|	|- 1.logger setting：日志路径，tensorboard日志路径，日志重定向等
|	|	|- 2.model setting：获取模型
|	|	|- 3.optimizer setting：获取优化器，不同权重层，优化器参数不同的设置
|	|	|- 4.resume setting：resume，fune turning等设置
|	|	|- 5.dataloader setting：数据集dataset定义-->Transformer(数据增强）-->Dataloader（数据加载)--> ...
|	|   |- 6.loss setting: 损失函数选择，有的实验可以略掉，因为在model中定义了
|	|   |- 7.scheduler setting：学习率调整策略选择
|	|   |- 8.other setting: 补充2model setting，EMA，DDP模型等设置
|	|   |- 9.evaluator setting：验证器设置，包括读取验证集，计算评价指标等
|	|- train_in_epoch(训练一个epoch的操作)
|	|	|- before_epoch(一个epoch之前的操作)
|	|	|	|- 判断此次epoch使用进行马赛克增强；
|	|	|	|- 修改此次epoch的损失函数；
|	|	|	|- 修改此次epoch的日志信息格式;
|	|	|	|- ...
|	|	|- train_in_iter(训练一次iter的操作，一次完整的forward&backwards)
|	|	|	|- before_iter(一次iter之前的操作)
|	|	|	|	|- nothing todo
|	|	|	|- train_one_iter(一次iter的操作)
|	|	|	|	|- 1.记录data time和iter time
|	|	|	|	|- 2.(预)读取数据
|	|	|	|	|- 3.forward
|	|	|	|	|- 4.计算loss
|	|	|	|	|- 5.backwards
|	|	|	|	|- 6.optimizer 更新网络权重
|	|	|	|	|- 7.是否进行EMA操作
|	|	|	|	|- 8.lr_scheduler修改学习率
|	|	|	|	|- 9.记录日志信息(datatime,itertime,各种loss,...)
|	|	|	|- after_iter(一次iter之后的操作)
|	|	|	|	|- 1.打印一个iter的日志信息(epoch,iter,losses,gpu mem,lr,eta,...)
|	|	|	|	|- 2.是否进行图片的resize，即多尺度训练
|	|	|- after_epoch(一个epoch之后的操作)
|	|	|	|- 1.保存模型
|	|	|	|- 2.是否进行，evaluator
|	|- after_train(训练之后的操作)
|	|	|- 输出最优的结果
```

## 3. 如何使用

### 3.1 Detection 

#### 3.1.1 Data Type

##### 1. VOC

详细请百度

##### 2. COCO

详细请百度

##### 3. CCDT

COCO Detection Type，参考[2.COCO](#####2. COCO数据集)数据集

#### 3.1.2  Train

`CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/trainval/trainval.py -f configs/det_yolox_voc_default_sgd_yoloxwarmcos_trainval_ubuntu20.04.json -d 2 -b 64 --fp16 -o --cache`

> 备注：如果使用`CUDA_VISIBLE_DEVICES=4`， 最好将`-o`参数去掉

#### 3.1.3 Eval

`CUDA_VISIBLE_DEVICES=0,1 python tools/eval/evalDet.py -f configs/det_yolox_voc_default_sgd_yoloxwarmcos_trainval_ubuntu20.04.json   -c  saved/det_yolox_voc_default_sgd_yoloxwarmcos_trainval_ubuntu20.04/voc/best_ckpt.pth -b 16 -d 1 --conf 0.001 --fp16 --fuse`

#### 3.1.4 Demo

`CUDA_VISIBLE_DEVICES=0 python tools/demo/demoDet.py image -f configs/det_yolox_voc_default_sgd_yoloxwarmcos_trainval_ubuntu20.04.json   -c  saved/det_yolox_voc_default_sgd_yoloxwarmcos_trainval_ubuntu20.04/voc/best_ckpt.pth --path /root/code/AI/datasets/VOCdevkit/VOC2007/JPEGImages/ or dog.jpg --conf 0.2 --nms 0.5 --tsize 640 --save_result --device gpu`

#### 3.4  Export

`CUDA_VISIBLE_DEVICES=0 python tools/export/export_onnx.py -f configs/det_yolox_voc_default_sgd_yoloxwarmcos_trainval_ubuntu20.04.json   -c  saved/pretrained/yolox_voc/11-14_08-45/best_ckpt.pth --output-name saved/pretrained/yolox_voc/11-14_08-45/best_ckpt.onnx`

#### 3.5  Deploy

### 3.2 Classification

### 3.3 Segmentation

### 3.4 IQA



### 3.5 FSL库使用

### 3.6 双输入库使用


# FelixFu的项目集合

## 1. 项目由来

在**工业流程**中，深度学习应用过程包括：

- TrainVal(针对特定场景，特定数据集训练一个模型)
- Eval(使用测试集测试，得到工业上的性能指标)
- Demo(将模型做个demo给客户展示)
- Export(将模型转成其他格式)
- Deploy(将模型部署到具体的设备上，并权衡速度与准确率)
- APP(将整个工业流程封装成带界面的APP)

在**深度学习训练**中，训练过程包括：

- 组件
  - dataloaders(数据加载)
  - models(模型：分类、分割、检测、异常检测等[Task](https://paperswithcode.com/sota))
  - losses(损失函数)
  - optims(优化器)
  - schedulers(学习率调整策略)
  - evaluator(训练过程中的性能评价)

- 联系（将上述组件联系起来）
  - trainers(训练过程)
    - resume
    - fune turning
    - 日志监控(训练过程日志输出，tensorboard，...)
    - 权重输出
    - multigpus(是否使用多GPU形式训练)
    - mixedprecisions(是否使用混合精度进行训练)
    - ......

目前，我所遇到的深度学习项目基本都能用这两个维度概括，为了方便以后使用，在此将两个维度整理成这个项目，将**工业流程（train、eval、demo、export、deploy的调用API）**封装到core.tools库中，将**深度学习训练(train & eval核心过程)**封装到core中。

## 2. 目录结构

### 2.1 组件支持

详细，请看API。

#### 2.1.1 Models

- Backbone
  - 使用[timm](resources/timm_introduce.md)作为backbone
- Classification
  - ResNets
  - EfficientNets
- Segmentation
  - PSPNet
  - UNet
  - UNet++
- Detection
  - Yolox
- Anomaly
  - PaDiM
  - CFlow-AD
  - 

#### 2.1.2 DataLoader

- DataSets

  - [MVTecDataset](resources/dataset_mvtecdataset.md)

  - ClsDataSet
  - SegDataSet

- DataLoaders

  - BatchDataLoader

- Augments

  - 

#### 2.1.3 Losses

- 

#### 2.1.4 Optims

- 

#### 2.1.5 Schedulers

- 

#### 2.1.6 Evaluators

- 



### 2.1 项目结构

```tree
root@c98fb50f30a8:~/code/DAO# tree
├── configs # 配置文件
│   ├── task-model-optim-datasetType-loss-scheduler-trainval-arch.json
│   ├── ...
│   └── super
│       ├── custom_modules.json
│       ├── dataloaders.json
│       ├── evaluators.json
│       ├── losses.json
│       ├── models.json
│       ├── optims.json
│       └── schedulers.json
├── core	# 核心库
│   ├── __init__.py
│   ├── modules	# 模块组件
│   │   ├── __init__.py
│   │   ├── dataloaders	# 数据加载组件
│   │   │   ├── CLS_TXTD.py
│   │   │   ├── __init__.py
│   │   │   ├── augments
│   │   │   │   ├── __init__.py
│   │   │   │   ├── custom
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── histogram.py
│   │   │   │   └── data_augment.py
│   │   │   ├── data_prefetcher.py
│   │   │   ├── dataloading.py
│   │   │   ├── datasets
│   │   │   │   ├── CLS_TXT.py
│   │   │   │   └── __init__.py
│   │   │   └── samplers.py
│   │   ├── evaluators	# 验证器组件
│   │   │   ├── CLS_TXT_Evaluator.py
│   │   │   └── __init__.py
│   │   ├── losses	# 损失函数组件
│   │   │   ├── CrossEntropyLoss.py
│   │   │   └── __init__.py
│   │   ├── models	# 模型组件
│   │   │   ├── __init__.py
│   │   │   ├── backbone
│   │   │   │   ├── EfficientNet
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── model.py
│   │   │   │   │   └── utils.py
│   │   │   │   └── __init__.py
│   │   │   └── cls
│   │   │       ├── EfficientNet.py
│   │   │       └── __init__.py
│   │   ├── optims	# 优化器组件
│   │   │   ├── __init__.py
│   │   │   └── sgd_warmup_bias_bn_weight.py
│   │   ├── register.py	# 所有模型注册器
│   │   ├── schedulers	# lr调整策略组件
│   │   │   ├── __init__.py
│   │   │   ├── cos_lr.py
│   │   │   ├── multistep_lr.py
│   │   │   ├── warm_cos_lr.py
│   │   │   ├── yolox_semi_warm_cos_lr.py
│   │   │   └── yolox_warm_cos.py
│   │   └── utils	# 工具
│   │       ├── __init__.py
│   │       └── metricCls.py
│   ├── tools	# train、evaluate、demo、export过程
│   │   ├── __init__.py
│   │   ├── demo.py
│   │   ├── eval.py
│   │   ├── export.py
│   │   ├── trainval.py
│   │   └── utils
│   │       ├── __init__.py
│   │       └── register_modules.py
│   ├── trainers	# trainval核心部分
│   │   ├── __init__.py
│   │   ├── launch.py
│   │   ├── trainerCls.py
│   │   └── utils
│   │       ├── __init__.py
│   │       ├── checkpoint.py
│   │       ├── ema.py
│   │       ├── logger.py
│   │       └── metrics.py
│   ├── utils # 工具
│   │   ├── __init__.py
│   │   ├── allreduce_norm.py
│   │   ├── dist.py
│   │   └── setup_env.py
│   └── version.py
├── custom_modules	# 自定义模块组件存放位置
│   ├── model_mine.py
│   └── model_mine2.py
├── desktop.ini
├── main.py
├── pretrained	# 预训练模型
│   └── efficientnet-b0-355c32eb.pth
├── requirements.txt
├── saved	# 结果存放位置
│   └── test
│       ├── 12-03_17-46
│       │   ├── best_ckpt.pth
│       │   ├── events.out.tfevents.1638524781.c98fb50f30a8.1183700.0
│       │   ├── last_epoch_ckpt.pth
│       │   ├── latest_ckpt.pth
│       │   └── train_log.txt
│       └── 12-06_09-43
│           ├── best_ckpt.pth
│           ├── events.out.tfevents.1638755035.c98fb50f30a8.1192675.0
│           ├── last_epoch_ckpt.pth
│           ├── latest_ckpt.pth
│           └── train_log.txt
├── LICENSE
├── README.md
├── setup.cfg
└── setup.py
```
### 2.2 调用流程

![](resources/main.png)

### 2.3 trainerX.py代码过程

#### 2.3.1 TrainVal过程

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

### 3.1 Classification



### 3.2 Detection 

#### 3.2.1 Data Type

##### 1. VOC

详细请百度

##### 2. COCO

详细请百度

##### 3. CCDT

COCO Detection Type，参考[2.COCO](#####2. COCO数据集)数据集

#### 3.2.2  Train

`CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/trainval/trainval.py -f configs/det_yolox_voc_default_sgd_yoloxwarmcos_trainval_ubuntu20.04.json -d 2 -b 64 --fp16 -o --cache`

> 备注：如果使用`CUDA_VISIBLE_DEVICES=4`， 最好将`-o`参数去掉

#### 3.2.3 Eval

`CUDA_VISIBLE_DEVICES=0,1 python tools/eval/evalDet.py -f configs/det_yolox_voc_default_sgd_yoloxwarmcos_trainval_ubuntu20.04.json   -c  saved/det_yolox_voc_default_sgd_yoloxwarmcos_trainval_ubuntu20.04/voc/best_ckpt.pth -b 16 -d 1 --conf 0.001 --fp16 --fuse`

#### 3.2.4 Demo

`CUDA_VISIBLE_DEVICES=0 python tools/demo/demoDet.py image -f configs/det_yolox_voc_default_sgd_yoloxwarmcos_trainval_ubuntu20.04.json   -c  saved/det_yolox_voc_default_sgd_yoloxwarmcos_trainval_ubuntu20.04/voc/best_ckpt.pth --path /root/code/AI/datasets/VOCdevkit/VOC2007/JPEGImages/ or dog.jpg --conf 0.2 --nms 0.5 --tsize 640 --save_result --device gpu`

#### 3.2.5  Export

`CUDA_VISIBLE_DEVICES=0 python tools/export/export_onnx.py -f configs/det_yolox_voc_default_sgd_yoloxwarmcos_trainval_ubuntu20.04.json   -c  saved/pretrained/yolox_voc/11-14_08-45/best_ckpt.pth --output-name saved/pretrained/yolox_voc/11-14_08-45/best_ckpt.onnx`

#### 3.2.6  Deploy

### 3.3 Segmentation

### 3.4 IQA

### 3.5 Anomaly

### 3.6 FSL


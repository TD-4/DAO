# API

## 1. models

### Unet

[source](../core/modules/models/seg/unet/model.py)

**构造函数**

```
class Unet( encoder,
            encoder_depth=5,
            encoder_channels=None,
            decoder_use_batchnorm: bool = True,
            decoder_channels: List[int] = (256, 128, 64, 32, 16),
            decoder_attention_type: Optional[str] = None,
            num_classes=2,
            activation: Optional[Union[str, callable]] = None,
            aux_params: Optional[dict] = None,
    )
"""
encoder: dict encoder的配置字典
encoder_depth: encoder深度
encoder_channels: encoder 的每一层channel数
decoder_use_batchnorm: bool
decoder_channels: List[int] = (256, 128, 64, 32, 16),
decoder_attention_type: Optional[str] = None,
num_classes=2,
activation: Optional[Union[str, callable]] = None,
aux_params: Optional[dict] = None,
"""
```

**configs.json**

```
"model": {
        "type": "Unet",
        "summary_size": [3,224,224],
        "backbone": {
            "kwargs": {
                "model_name": "resnet18",
                "pretrained": true,
                "checkpoint_path": "",
                "exportable": true,
                "in_chans": 3,
                "features_only": true
            }
        },
        "kwargs": {
            "encoder_depth": 5,
            "encoder_channels": [3, 64, 64, 128, 256, 512],
            "decoder_channels": [256, 128, 64, 32, 16],
            "num_classes": 21
        }
    }
```

## 2. optims

### sgd_warmup_bias_bn_weight

[source](../core/modules/optims/sgd_warmup_bias_bn_weight.py)

**构造函数**

```
def sgd_warmup_bias_bn_weight(model=None,
                              lr=0.01,
                              weight_decay=1e-4,
                              momentum=0.9,
                              warmup_lr=0,
                              warmup_epoch=5
                              ):
    """
    model:torch.nn.Module 此trainer的self.model属性
    lr: float 对于整个（多机多卡）batch size的学习率
    weight_decay:float torch.optim.SGD 默认参数
    momentum:float torch.optim.SGD 默认参数
    warmup_lr:float warmup时的学习率
    warmup_epoch:int warmup几个epoch
    """
```

**configs.json**

```
   "optimizer": {
        "type": "sgd_warmup_bias_bn_weight",
        "kwargs": {
            "lr": 0.01,
            "weight_decay": 1e-4,
            "momentum": 0.9,
            "warmup_lr": 0,
            "warmup_epoch": 5
        }
    }
```

## 3. datasets & dataloaders

### SegDataset

[source](../core/modules/dataloaders/datasets/SegDataset.py)

**构造函数**

```
Class SegDataset(data_dir=None, preproc=None, image_set="", in_channels=1, input_size=(224, 224), cache=False, image_suffix=".jpg", mask_suffix=".png"):
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
    images_suffix:str 可接受的图片后缀
    mask_suffix:str
"""
```

**config.json**

```
"dataset": {
	"type": "SegDataset",
	"kwargs": {
		"data_dir": "/root/data/DAO/VOC2012_Seg_Aug",
		"image_set": "val.txt",
        "in_channels": 3,
        "input_size": [380, 380],
        "cache": false,
        "image_suffix":".jpg",
        "mask_suffix":".png"
	},
    "transforms": {
    	"kwargs": {
    		"Resize": {"height": 224, "width": 224, "p": 1},
    		"Normalize": {"mean": [0.398993, 0.431193, 0.452234], "std": [0.285205, 0.273126, 0.276610], "p": 1}
		}
	}
}
```

### SegDataloaderTrain

[source](../core/modules/dataloaders/SegDataloader.py)

**构造函数**

```
def SegDataloaderTrain(is_distributed=False, batch_size=None, num_workers=None, dataset=None, seed=0)
"""
is_distributed : bool 是否是分布式
batch_size : int batchsize大小
num_workers : int 读取数据线程数
dataset : DotMap 数据集配置
seed : int 随机种子
"""
```

**configs.json**

```
    "dataloader": {
        "type": "SegDataloaderTrain",
        "dataset": {
	        "type": "SegDataset",
            "kwargs": {
                    "data_dir": "/root/data/DAO/VOC2012_Seg_Aug",
                    "image_set": "val.txt",
                    "in_channels": 3,
                    "input_size": [380, 380],
                    "cache": false,
                    "image_suffix":".jpg",
                    "mask_suffix":".png"
	            },
            "transforms": {
                "kwargs": {
                    "Resize": {"height": 224, "width": 224, "p": 1},
                    "Normalize": {"mean": [0.398993, 0.431193, 0.452234], "std": [0.285205, 0.273126, 0.276610], "p": 1}
                }
            }
        },
        "kwargs": {
            "num_workers": 4,
            "batch_size": 32
        }
    }
```

### SegDataloaderEval

[source](../core/modules/dataloaders/SegDataloader.py)

**构造函数**

```
def SegDataloaderEval(is_distributed=False, batch_size=None, num_workers=None, dataset=None):
    """
    is_distributed : bool 是否是分布式
    batch_size : int batchsize大小
    num_workers : int 读取数据线程数
    dataset : DotMap 数据集配置
    """
```

**configs.json**

```
"dataloader": {
            "type": "SegDataloaderEval",
            "dataset": {
                "type": "SegDataset",
                "kwargs": {
                    "data_dir": "/root/data/DAO/VOC2012_Seg_Aug",
                    "image_set": "val.txt",
                    "in_channels": 3,
                    "input_size": [380, 380],
                    "cache": false,
                    "image_suffix":".jpg",
                    "mask_suffix":".png"
                },
                "transforms": {
                    "kwargs": {
                        "Resize": {"height": 224, "width": 224, "p": 1},
                        "Normalize": {"mean": [0.398993, 0.431193, 0.452234], "std": [0.285205, 0.273126, 0.276610], "p": 1}

                    }
                }
            },
            "kwargs": {
                "num_workers": 4,
                "batch_size": 32
            }
        }
```

## 4. losses

### CrossEntropyLoss

[source](../core/modules/losses/CrossEntropyLoss.py)

**构造函数**

```
def CrossEntropyLoss(weight=None, ignore_index=255, reduction='mean')

```

**configs.json**

```
"loss": {
        "type": "CrossEntropyLoss",
        "kwargs": {
            "ignore_index": 255,
            "weight": [
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1
            ],
            "reduction": "mean"
        }
    }
```

## 5. scheduler

### warm_cos_lr

[source](../core/modules/schedulers/warm_cos_lr.py)

**构造函数**

```
class warm_cos_lr(self, lr, iters_per_epoch, total_epochs, **kwargs):
        """
        Args:
            lr (float): learning rate.
            iters_per_peoch (int): number of iterations in one epoch.
            total_epochs (int): number of epochs in training.
            kwargs (dict):
                - warmup_epochs
                - warmup_lr_start (default 1e-6)
        """
```

**调用方法**

```
def update_lr(self, iters) 更新lr
```

**configs**

```
"lr_scheduler": {
        "type": "warm_cos_lr",
        "kwargs": {
            "warmup_epochs": 5,
            "warmup_lr_start": 0
        }
    }
```

## 6. evaluators

### SegEvaluator

[source](../core/modules/evaluators/SegEvaluator.py)

**构造函数**

```
class SegEvaluator(is_distributed=False, dataloader=None, num_classes=None)
```

**调用方法**

```
evaluate(self, model, distributed=False, half=False, device=None)
```

**configs**

```
 "evaluator": {
        "type": "SegEvaluator",
        "dataloader": {
            "type": "SegDataloaderEval",
            "dataset": {
                "type": "SegDataset",
                "kwargs": {
                    "data_dir": "/root/data/DAO/VOC2012_Seg_Aug",
                    "image_set": "val.txt",
                    "in_channels": 3,
                    "input_size": [380, 380],
                    "cache": false,
                    "image_suffix":".jpg",
                    "mask_suffix":".png"
                },
                "transforms": {
                    "kwargs": {
                        "Resize": {"height": 224, "width": 224, "p": 1},
                        "Normalize": {"mean": [0.398993, 0.431193, 0.452234], "std": [0.285205, 0.273126, 0.276610], "p": 1}
                    }
                }
            },
            "kwargs": {
                "num_workers": 4,
                "batch_size": 32
            }
        }

    }
```



MVTecDataset

```
MVTecDataset(
    data_dir=None,
    preproc=None,
    image_set="",
    in_channels=1,
    input_size=(224, 224),
    cache=False,
    image_suffix=".png",
    mask_suffix=".png",
    **kwargs
)
```

异常检测数据集，（MVTecDataset类型）

**1. 构造函数**

- data_dir:str  数据集文件夹路径，文件夹要求是
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
- preproc:albumentations.Compose 对图片进行预处理
- image_set:str "train.txt or val.txt or test.txt"
- in_channels:int  输入图片的通道数，目前只支持1和3通道
- input_size:tuple 输入图片的HW
- cache:bool 是否对图片进行内存缓存
- image_suffix:str 可接受的图片后缀
- mask_suffix:str 可接受的图片后缀

**2.configs.json**




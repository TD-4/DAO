# API

### DataSets

#### MVTecDataset

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

```
```



### Models

#### PatchCore


# PaDiM

⌚️:2021年12月17日

📚参考

- [paper with code](https://paperswithcode.com/paper/padim-a-patch-distribution-modeling-framework)
- [paper 中文](https://blog.csdn.net/weixin_38328533/article/details/114596655)

----

![image-20211221114214517](E:\DAO\resources\anomaly\imgs\image-20211221114214517.png)

## 1. Training

其中b_train代表整个good数据个数。

- 提取特征：train_outputs
  - layer1:(b_train, 256, 56, 56)
  - layer2:(b_train, 512, 28, 28)
  - layer3:(b_train, 1024, 14, 14)
- 堆叠特征：embeding_vectors:(b_train, 1792, 56, 56)
- 随机选择：embeding_vectors:(b_train, d, 56, 56), d针对不同的网络选择不同, d=550
  - 变换：embeding_vectors:(b_train,d, 3136)
- 求mean(均值)和cov(协方差):
  - mean:(d, 3136) 可以理解为H*W中，每个点有d维特征向量, 每个特征向量在N张图片的均值
  - cov:(d, d, 3136)可以理解为H*W中每个点的一个协方差，即每个与另外d个点的相关性（包括自身）

## 2. Testing

其中b_test代表整个测试图片数量。

- 特征提取：test_outputs
  - layer1:(b, 256, 56, 56)
  - layer2:(b, 512, 28, 28)
  - layer3:(b, 1024, 14, 14)
- 堆叠特征：embeding_vectors:(b_test, 1792, 56, 56)
- 随机选择：embeding_vectors:(b_test, d, 56, 56), d针对不同的网络选择不同, d=550
  - 变换：embeding_vectors:(b_test,d, 3136)
- 求马氏距离：
  - 求embeding_vectors(b_test, d, 3136)中每一个点到train中的mean(d, 3136)和cov(d, d, 3136)的马氏距离，即dist_list为3136的向量，每个元素为b_test大小。
  - 假设embeding_vectors 第0个batch中， 3136中的第0个点-->embeding_vectors[0, :, 0]到mean(d)和cov(d,d) ----》计算出一个马氏距离float。b_test个计算出b_test个马氏距离。
  - for i in H*W:
    - for j in b_test:
      - dist.append(马氏距离)
    - dist_list.append(马氏距离)
  - 变换dist_list: (b_test, 56, 56)
- 上采样：score_map:(b_test, 224, 244)
- 高斯滤波：处理上采样结果，pixel(0, 255)
- 均值化: scores:(b_test, 224, 224), pixel(0, 1)
- 统计与显示
  - 计算image-level的ROC&AUC
    - img_socres (b_test),  值为224*224中最大值
    - gt_list(b_test), 真实标签， 值为0，1
  - 计算pixel-level的ROC&AUC
    - 获得的threshold
      - 计算P，R， threshold， f1==>==threshold==>threshold[np.argmax(f1)]
      - 计算fpr与tpr，绘制ROC，计算AUC。


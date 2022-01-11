# CFLOW-AD

- [CFLOW-AD: Real-Time Unsupervised Anomaly Detection with Localization via Conditional Normalizing Flows](https://arxiv.org/pdf/2107.12571v1.pdf)

- https://github.com/janosh/awesome-normalizing-flows


---

## 1. Paper📕

#### **Abstract** 

Unsupervised anomaly detection with localization has many practical applications when labeling is infeasible and, moreover, when anomaly examples are completely missing in the train data. <u>（无监督异常检测和定位有很多实用的应用，当在标记方法不可行以及异常实例在训练数据中完全缺失的情况下）</u>。While recently proposed models for such data setup achieve high accuracy metrics, their complexity is a limiting factor for real-time processing. <u>（虽然最近为此类问题提出的模型实现了高精度度量，但它们的复杂性是实时处理的一个限制因素）</u>。In this paper, we propose a real-time model and analytically derive its relationship to prior methods.<u>（在本文中，我们提出了一个实时模型，并分析推导出它与现有方法的关系）</u>。 Our CFLOW-AD model is based on a conditional normalizing flow framework adopted for anomaly detection with localization. In particular, CFLOW-AD consists of a discriminatively pretrained encoder followed by a multi-scale generative decoders where the latter explicitly estimate likelihood of the encoded features. <u>（我们的CFLOW-AD模型是**基于一个有条件的归一化流程框架**，用于异常检测和定位。特别地，CFLOW-AD由一个有区别的预先训练的**编码器**和一个多尺度生成**解码器**组成，后者显式地估计编码特征的可能性）</u>。Our approach results in a computationally and memory-efficient model: CFLOW-AD is faster and smaller by a factor of 10× than prior state-of-the-art with the same input setting. Our experiments on the MVTec dataset show that CFLOW-AD outperforms previous methods by 0.36% AUROC in detection task, by 1.12% AUROC and 2.5% AUPRO in localization task, respectively. We open-source our code with fully reproducible experiments1（<u>我们的方法得到了一个计算和内存效率高的模型:CFLOW-AD的速度更快，体积也更小，在相同的输入设置下，它比之前的先进技术要小10倍。在MVTec数据集上的实验表明，CFLOW-AD在检测任务上的AUROC分别比现有方法提高了0.36%，在定位任务上的AUROC分别提高了1.12%和2.5%。我们用完全可重复的实验来开放我们的代码</u>）。



#### 1. Introduction 

Anomaly detection with localization (AD) is a growing area of research in computer vision with many practical applications e.g. industrial inspection [4], road traffic monitoring [20], medical diagnostics [44] etc. （<u>基于局部化的异常检测是一种新兴的检测方法计算机视觉的研究领域，具有许多实际应用，如工业检测[4]，道路交通监测[20]，医疗诊断[44]等</u>）。However, the common supervised AD [32] is not viable in practical applications due to several reasons. First, it requires labeled data which is costly to obtain. Second, anomalies are usually rare long-tail examples and have low probability to be acquired by sensors. Lastly, consistent labeling of anomalies is subjective and requires extensive domain expertise as illustrated in Figure 1 with industrial cable defects. （<u>但是，由于几个原因，常见的有监督的AD[32]在实际应用中是不可行的。首先，它需要有标签的数据这是昂贵的获得。第二，异常现象通常都存在罕见的长尾例子，被传感器捕获的概率很低。最后，对异常现象进行一致的标记是主观的，需要广泛的领域专业知识，如图1所示，工业电缆缺陷</u>）。



With these limitations of the supervised AD, a more appealing approach is to collect only unlabeled anomaly-free images for train dataset Dtrain as in Figure 1 (top row). （<u>有了监督AD的这些限制，一种更有吸引力的方法是只收集无标记的异常如图1所示(第一行)</u>）。Then, any deviation from anomaly-free images is classified as an anomaly. Such data setup with low rate of anomalies is generally considered to be unsupervised [4]. （然后，对任何与无异常图像的偏差进行分类作为一个异常。这样的数据设置异常率低通常被认为是无监督[4]）。Hence, the AD task can be reformulated as a task of out-of-distribution detection (OOD) with the AD objective. （因此,AD任务可以被重新定义为非分布任务检测(OOD)与AD目标）。



![image-20220104152821143](imgs/image-20220104152821143.png)

Figure 1. An example of the proposed out-of-distribution (OOD) detector for anomaly localization trained on anomaly-free Dtrain (top row). 图1所示。建议的非分配(OOD)示例基于无异常Dtrain训练的异常定位检测器(上面一行)。 Sliced cable images are from the MVTec dataset [4], where the bottom row illustrates Dtest ground truth masks for anomalies (red) and the middle row shows examples of anomalyfree patches (green). The OOD detector learns the distribution of anomaly-free patches x with pX(x) density and transforms it into a Gaussian distribution with pZ (z) density. Threshold τ separates in-distribution patches from the OOD patches with pZ˜(z) density .切片电缆图像来自MVTec数据集[4]，最下面一排是Dtest ground truth masks具异常(红色)和中间一行显示无异常斑块(绿色)的例子。OOD检测器学习的分布将pX(x)密度的无异常patch x转换为pZ (z)密度的高斯分布。阈值τ分离基于pZ ~ (z)密度的OOD patch的内分布patch



While OOD for low-dimensional industrial sensors (e.g.power-line or acoustic) can be accomplished using a common k-nearest-neighbor or more advanced clustering methods [10], it is less trivial for high-resolution images. 而用于低维工业传感器的OOD(如电力线或声学)可以使用普通的k-最近邻或更高级的聚类方法[10]来完成，它对于高分辨率的图像不是那么简单。 Recently, convolutional neural networks (CNNs) have gained bpopularity in extracting semantic information from images into downsampled feature maps [4]. x近年来，卷积神经网络得到了广泛的应用流行于从图像中提取语义信息下采样特征映射[4]。 Though feature extraction using CNNs has relatively low complexity, the postprocessing of feature maps is far from real-time processing in the state-of-the-art unsupervised AD methods [8].虽然利用cnn进行特征提取的复杂度相对较低，但特征图的后处理与实时处理相去甚远在最先进的无监督AD方法中[8]。



To address this complexity drawback, we propose a CFLOW-AD model that is based on conditional normalizing flows. 为了解决这个复杂的缺点，我们提出了一个基于条件归一化流程的CFLOW-AD模型。 CFLOW-AD is agnostic to feature map spatial dimensions similar to CNNs, which leads to a higher accuracy metrics as well as a lower computational and memory requirements.CFLOW-AD对于特征映射空间是不可知的类似cnn的尺寸，这导致了更高的精度指标，以及较低的计算和内存要求。  We present the main idea behind our approach in a toy OOD detector example in Figure 1.我们在图1中的一个玩具OOD检测器示例中展示了这种方法背后的主要思想。  A distribution of the anomaly-free image patches x with probability density function pX(x) is learned by the AD model. 通过AD模型学习无异常图像块x的概率密度函数为pX(x)的分布。Our translation-equivariant model is trained to transform the original distribution with pX(x) density into a Gaussian distribution with pZ(z) density.我们的平移等变模型被训练成变换原密度为pX(x)的分布变为密度为pZ(z)的高斯分布。  Finally, this model separates in-distribution patches z with pZ(z) from the outof-distribution patches with pZ˜(z) using a threshold τ computed as the Euclidean distance from the distribution mean. 最后,这个模型使用阈值τ计算为与分布均值的欧氏距离，将具有pZ(z)的分布内patch z与具有pZ(z)的分布外patch分离。



#### 2. Related work

 We review models2 that employ the data setup from Figure 1 and provide experimental results for popular MVTec dataset [4] with factory defects or Shanghai Tech Campus (STC) dataset [22] with surveillance camera videos. 我们回顾models2它们采用了图1中的数据设置，并为流行的MVTec提供了实验结果数据集[4]工厂缺陷或上海技术园区(STC)数据集[22]与监控摄像机视频。We highlight the research related to a more challenging task of pixel-level anomaly localization (segmentation) rather than a more simple image-level anomaly detection.我们突出研究的一个更有挑战性的任务像素级异常定位(分割)而不是一种更简单的图像级异常检测。



Napoletano et al. [25] propose to use CNN feature extractors followed by a principal component analysis and kmean clustering for AD. Their feature extractor is a ResNet18 [13] pretrained on a large-scale ImageNet dataset [16]. Similarly, SPADE [7] employs a Wide-ResNet-50 [43] with multi-scale pyramid pooling that is followed by a k-nearestneighbor clustering. Unfortunately, clustering is slow at test-time with high-dimensional data. Thus, parallel convolutional methods are preferred in real-time systems.

Napoletano et al.[25]提出使用CNN特征提取器，然后进行主成分分析和kmean聚类来进行AD。他们的特征提取器是在大规模ImageNet数据集[16]上预先训练的ResNet18[13]。类似地，SPADE[7]使用了宽resnet -50 [43]多尺度金字塔池，然后是k近邻聚类。不幸的是，集群是缓慢的高维数据的测试时间。因此，并行卷积方法在实时系统中是首选。



Numerous methods are based on a natural idea of generative modeling. Unlike models with the discriminatively pretrained feature extractors [25, 7], generative models learn distribution of anomaly-free data and, therefore, are able to estimate a proxy metrics for anomaly scores even for the unseen images with anomalies. Recent models employ generative adversarial networks (GANs) [35, 36] and variational autoencoders (VAEs) [3, 38].

许多方法都基于生成建模的自然思想。不像有区别地预先训练的特征提取器的模型[25,7]，生成模型了解无异常数据的分布，因此，是能够估计一个代理指标的异常分数甚至看不见的异常图像。最近的模型采用生成式对抗网络(GANs)[35,36]和变分自动编码器(VAEs)[3,38]。



A fully-generative models [35, 36, 3, 38] are directly applied to images in order to estimate pixel-level probability density and compute per-pixel reconstruction errors as anomaly scores proxies.将完全生成模型[35,36,3,38]直接应用于图像，以估计像素级概率密度，并计算逐像素重建误差作为异常评分代理。 These fully-generative models are unable to estimate the exact data likelihoods [6, 24] and do not perform better than the traditional methods [25, 7] according to MVTec survey in [4]. 这些完全生成的模型是无法估计准确的数据可能性[6,24]和做根据[4]中MVTec的调查，并不比传统方法更好[25,7]。 Recent works [34, 15] show that these models tend to capture only low-level correlations instead of relevant semantic information. 近期作品[34,15]表明这些模型往往只捕获低级的相关性，而不是相关的语义信息。 To overcome the latter drawback, a hybrid DFR model [37] uses a pretrained feature extractor with multi-scale pyramid pooling followed by a convolutional autoencoder (CAE). However, DFR model is unable to estimate the exact likelihoods. 为了克服后一个缺点，混合DFR模型[37]使用预先训练的多尺度金字塔池特征提取器，然后是卷积自动编码器(CAE)。然而，DFR模型无法估计确切的可能性。



Another line of research proposes to employ a studentteacher type of framework [5, 33, 41]. Teacher is a pretrained feature extractor and student is trained to estimate a scoring function for AD. Unfortunately, such frameworks underperform compared to state-of-the-art models

另一项研究建议采用学生教师类型的框架[5,33,41]。教师是经过训练的特征提取器，而学生则是经过训练的估计器AD评分功能。不幸的是,这种框架与最先进的模型相比表现不佳。



Patch SVDD [42] and CutPaste [19] introduce a selfsupervised pretraining scheme for AD. Moreover, Patch SVDD proposes a novel method to combine multi-scale scoring masks to a final anomaly map. 补丁SVDD[42]和CutPaste[19]为AD引入了一种自监督的预训练方案。此外,补丁SVDD提出了一种新的多尺度结合的方法为异常地图打分。Unlike the nearestneighbor search in [42], CutPaste estimates anomaly scores using an efficient Gaussian density estimator. 与[42]中的最近邻搜索不同，CutPaste估计异常分数使用有效的高斯密度估计。 While the self-supervised pretraining can be helpful in uncommon data domains, Schirrmeister et al. [34] argue that large natural-image datasets such as ImageNet can be a more representative for pretraining compared to a small applicationspecific datasets e.g. industrial MVTec [4].而自我监督的预训练在不常见的情况下是有用的数据域，Schirrmeister等人。[34]认为大自然图像数据集(如ImageNet)比小型应用特定数据集(如工业MVTec[4])更能代表预训练。



The state-of-the-art PaDiM [8] proposes surprisingly simple yet effective approach for anomaly localization. 最先进的padm[8]的提议令人惊讶一种简单有效的异常定位方法。 Similarly to [37, 7, 42], this approach relies on ImageNet pretrained feature extractor with multi-scale pyramid pooling. 与[37,7,42]类似，该方法依赖于imagenet pretraining的多尺度金字塔池特征提取器。 However, instead of slow test-time clustering in [7] or nearest-neighbor search in [42], PaDiM uses a wellknown Mahalanobis distance metric [23] as an anomaly score.但是，与在[7]中缓慢的测试时集群不同或在[42]中进行最近邻搜索时，padim使用著名的马氏距离度量[23]作为异常得分。  The metric parameters are estimated for each feature vector from the pooled feature maps. 从集合的特征映射中估计每个特征向量的度量参数。PaDiM has been inspired by Rippel et al. [29] who firstly advocated to use this measure for anomaly detection without localization.PaDiM一直受Rippel等人[29]的启发，他们首先倡导使用这种方法用于不定位的异常检测。



DifferNet [30] uses a promising class of generative models called normalizing flows (NFLOWs) [9] for image-level AD. DifferNet[30]使用了一个很有前途的生成模型类，称为标准化流(NFLOWs)[9]，用于图像级。 The main advantage of NFLOW models is ability to estimate the exact likelihoods for OOD compared to other generative models [35, 36, 3, 38, 37].NFLOW模型的主要优点是能够与其他疾病相比，估计OOD的确切可能性生成模型[35,36,3,38,37]。  In this paper, we extend DifferNet approach to pixel-level anomaly localization task using our CFLOW-AD model. 在本文中，我们将DifferNet方法扩展到像素级异常定位使用我们的CFLOW-AD模型。 In contrast to RealNVP [9] architecture with global average pooling in [30], we propose to use conditional normalizing flows [2] to make CFLOW-AD suitable for low-complexity processing of multi-scale feature maps for localization task. We develop our CFLOW-AD with the following contributions:与RealNVP[9]架构在[30]中使用全局平均池相比，我们建议使用条件归一化流程[2]到使CFLOW-AD适用于低复杂度的处理用于定位任务的多尺度特征图。我们开发CFLOW-AD的贡献如下:

- Our theoretical analysis shows why multivariate Gaussian assumption is a justified prior in previous models and why a more general NFLOW framework objective converges to similar results with the less compute.我们的理论分析表明，为什么多元高斯假设是一个合理的先验在以前的模型以及为什么使用更通用的NFLOW框架目标以较少的计算量收敛到相似的结果。
- We propose to use conditional normalizing flows for unsupervised anomaly detection with localization using computational and memory-efficient architecture.我们建议使用有条件的标准化流程无监督异常检测与定位使用计算和内存高效的架构。
- We show that our model outperforms previous stateof-the art in both detection and localization due to the unique properties of the proposed CFLOW-AD model.我们表明，我们的模型在检测和定位方面优于以前的先进水平，因为所提出的CFLOW-AD模型的独特性质。

#### 3. Theoretical background 

##### 3.1. Feature extraction with Gaussian prior

Consider a CNN h(λ) trained for classification task. Its parameters λ are usually found by minimizing KullbackLeibler (DKL) divergence between joint train data distribution Qx,y and the learned model distribution Px,y(λ), where (x, y) is an input-label pair for supervised learning.考虑为分类任务训练的CNN h(λ)。它的参数λ通常通过最小化联合列车数据分布Qx,y和学习模型分布Px,y(λ)之间的KullbackLeibler (DKL)散度来找到，其中(x, y)为监督学习的输入标签对。

Typically, the parameters λ are initialized by the values sampled from the Gaussian distribution [12] and optimization process is regularized as：通常，参数λ由值初始化采样自高斯分布[12]，优化过程正则化为

![image-20220104160015055](imgs/image-20220104160015055.png)

where R(λ) is a regularization term and α is a hyperparameter that defines regularization strength.其中R(λ)是正则项，α是定义正则化强度的超参数。



The most popular CNNs [13, 43] are trained with L2 weight decay [17] regularization (R(λ) = kλk 2 2 ). That imposes multivariate Gaussian (MVG) prior not only to parameters λ, but also to the feature vectors z extracted from the feature maps of h(λ) [11] intermediate layers.最流行的cnn[13,43]是用L2训练的正则化(R(λ) = kλk22)． 这使得多元高斯(MVG)不仅优先于参数λ，而且优先于提取的特征向量zh(λ)[11]中间层的特征图。

##### 3.2. A case for Mahalanobis distance 

With the same MVG prior assumption, Lee et al. [18] recently proposed to model distribution of feature vectors z by MVG density function and to use Mahalanobis distance [23] as a confidence score in CNN classifiers. 在相同的MVG先验假设下，Lee等人[18]最近提出的特征向量分布模型利用MVG密度函数，利用Mahalanobis距离[23]作为CNN分类器的置信分数。 Inspired by [18], Rippel et al. [29] adopt Mahalanobis distance for anomaly detection task since this measure determines a distance of a particular feature vector z to its MVG distribution. Consider a MVG distribution N (µ, Σ) with a density function pZ(z) for random variable z ∈ R D defined as启发通过[18]，Rippel等人[29]采用马氏距离异常检测任务，因为该测度决定了特定特征向量z到其MVG分布的距离。考虑一个具有密度的MVG分布N(µ，Σ)函数pZ(z)对于随机变量z∈RD定义为

![image-20220104160303169](imgs/image-20220104160303169.png)

where µ ∈ R D is a mean vector and Σ ∈ R D×D is a covariance matrix of a true anomaly-free density pZ(z). Then, the Mahalanobis distance M(z) is calculated as∈R。D是一个均值向量，Σ∈R D×D是一个真正的无异常密度pZ(z)的协方差矩阵。然后，计算马氏距离M(z)为

![image-20220104160403713](imgs/image-20220104160403713.png)

Since the true anomaly-free data distribution is unknown, mean vector and covariance matrix from (3) are replaced by the estimates µˆ and Σˆ calculated from the empirical train dataset Dtrain. At the same time, density function pZ˜(z) of anomaly data has different µ˜ and Σ˜ statistics, which allows to separate out-of-distribution and indistribution feature vectors using M(z) from (3).由于真正的无异常数据分布是未知的，因此(3)的均值向量和协方差矩阵被由经验训练数据集Dtrain计算出的估计µˆ和Σˆ所代替。同时，异常数据的密度函数pZ (z)具有不同的µ~和Σ statistics，允许使用M(z)从(3)中分离出非分布特征向量和分布特征向量。



This framework with MVG distribution assumption shows its effectiveness in image-level anomaly detection task [29] and is adopted by the state-of-the-art PaDiM [8] model in pixel-level anomaly localization task.该框架采用MVG分布假设表明了该方法在图像级异常检测中的有效性任务[29]，被最先进的padm[8]所采用基于像素级异常定位任务的模型。



##### 3.3. Relationship with the flow framework 

Dinh et al. [9] introduce a class of generative probabilistic models called normalizing flows. These models apply change of variable formula to fit an arbitrary density pZ(z) by a tractable base distribution with pU (u) density and a bijective invertible mapping g −1 : Z → U. Then, the loglikelihood of any z ∈ Z can be estimated by

Dinh等人[9]介绍了一类生成概率模型，称为归一化流。这些模型应用变量变换公式，将任意密度pZ(z)拟合为一个具有pU (u)密度的可处理基分布和一个双射可逆映射g−1:z→u，则任意z∈z的对数似然可由

![image-20220104160623913](imgs/image-20220104160623913.png)

where a sample u ∼ pU is usually from standard MVG distribution (u ∼ N (0, I)) and a matrix J = ∇zg −1 (z, θ) is the Jacobian of a bijective invertible flow model (z = g(u, θ) and u = g −1 (z, θ)) parameterized by vector θ.

其中样本u ~ pU通常来自标准MVG分布(u ~ N (0, I))，矩阵J =∇zg−1 (z， θ)是由向量θ参数化的双射可逆流模型(z = g(u， θ)和u = g−1 (z， θ))的雅可比矩阵。

The flow model g(θ) is a set of basic layered transformations with tractable Jacobian determinants. For example, log | det J| in RealNVP [9] coupling layers is a simple sum of layer’s diagonal elements. These models are optimized using stochastic gradient descent by maximizing loglikelihood in (4). Equivalently, optimization can be done by minimizing the reverse DKL [ˆpZ(z, θ)kpZ(z)] [27], where pˆZ(z, θ) is the model prediction and pZ(z) is a target density. The loss function for this objective is defined as

流动模型g(θ)是一组具有可处理雅可比行列式的基本层变换。例如，RealNVP[9]耦合层中的log | det J|是层对角线元素的简单和。这些模型采用随机梯度下降法，通过最大化(4)中的对数似然来优化。同样地，优化可以通过最小化反向DKL[ˆpZ(z， θ)kpZ(z)][27]来完成，其中pˆz (z， θ)为模型预测，pZ(z)为目标密度。此目标的损失函数定义为

![image-20220104160725496](imgs/image-20220104160725496.png)

If pZ(z) is distributed according to Section 3.1 MVG assumption, we can express (5) as a function of Mahalanobis distance M(z) using its definition from (3) as

如果pZ(z)是根据3.1节MVG假设分布的，我们可以将(5)表示为马氏距离M(z)的函数，由(3)定义为

![image-20220104160757783](imgs/image-20220104160757783.png)



where E2 (u) = kuk 2 2 is a squared Euclidean distance of a sample u ∼ N (0, I) (detailed proof in Appendix A)

其中E2 (u) = kuk 2是样本u ~ N (0, I)的平方欧几里德距离(详细证明见附录a)



Then, the loss in (6) converges to zero when the likelihood contribution term | det J| of the model g(θ) (normalized by det Σ−1/2 ) compensates the difference between a squared Mahalanobis distance for z from the target density and a squared Euclidean distance for u ∼ N (0, I).

然后，当模型g(θ)(通过det Σ−1/2归一化)的似然贡献项| det J|补偿z到目标密度的马氏距离的平方和u ~ N (0, I)的欧氏距离的平方之间的差时，(6)的损失收敛到零。



This normalizing flow framework can estimate the exact likelihoods of any arbitrary distribution with pZ density, while Mahalanobis distance is limited to MVG distribution only. For example, CNNs trained with L1 regularization would have Laplace prior [11] or have no particular prior in the absence of regularization. Moreover, we introduce conditional normalizing flows in the next section and show that they are more compact in size and have fully-convolutional parallel architecture compared to [7, 8] models.

这种归一化流框架可以估计任何具有pZ密度的任意分布的精确概率，而马氏距离仅局限于MVG分布。例如，使用L1正则化训练的cnn，要么有拉普拉斯先验[11]，要么在没有正则化的情况下没有特定先验。此外，我们在下一节中引入了条件归一化流，并表明与[7,8]模型相比，它们在尺寸上更紧凑，并具有完全卷积并行架构。



![image-20220104160936220](imgs/image-20220104160936220.png)

Figure 2. Overview of our CFLOW-AD with a fully-convolutional translation-equivariant architecture. Encoder h(λ) is a CNN feature extractor with multi-scale pyramid pooling. Pyramid pooling captures both global and local semantic information with the growing from top to bottom receptive fields. Pooled feature vectors z k i are processed by a set of decoders gk(θk) independently for each kth scale. Our decoder is a conditional normalizing flow network with a feature input z k i and a conditional input c k i with spatial information from a positional encoder (PE). The estimated multi-scale likelihoods p k i are upsampled to the input size and added up to produce anomaly map.

图2。我们的CFLOW-AD与全卷积平移等变结构的概述。编码器h(λ)是一种多尺度金字塔池化的CNN特征提取器。金字塔池捕获全局和局部语义信息，接受域从上到下不断增长。对于每个第k个尺度，合并的特征向量zki由一组解码器gk(θk)独立处理。我们的解码器是一个条件归一化流网络，特征输入zki和带有空间信息的条件输入cki来自位置编码器(PE)。将估计的多尺度似然概率pi上采样到输入大小，并相加生成异常图。



#### 4. The proposed CFLOW-AD model

#####  4.1. CFLOW encoder for feature extraction 

We implement a feature extraction scheme with multiscale feature pyramid pooling similar to recent models [7, 8]. We define the discriminatively-trained CNN feature extractor as an encoder h(λ) in Figure 2. 我们实现了一种多尺度特征金字塔池的特征提取方案，类似于最近的模型[7,8]。我们将区别训练的CNN特征提取器定义为图2中的编码器h(λ)。The CNN encoder maps image patches x into a feature vectors z that contain relevant semantic information about their content.CNN编码器将图像块x映射为一个特征向量z，该特征向量包含关于其内容的相关语义信息。  CNNs accomplish this task efficiently due to their translation equivariant architecture with the shared kernel parameters.cnn高效地完成了这一任务，因为它们具有具有共享内核参数的翻译等变架构 。 In our experiments, we use ImageNet-pretrained encoder following Schirrmeister et al. [34] who show that large natural-image datasets can serve as a representative distribution for pretraining. 在我们的实验中，我们使用了继Schirrmeister等人[34]之后的imagenet预训练编码器，他们表明大型自然图像数据集可以作为预训练的代表性分布。If a large application-domain unlabeled data is available, the self-supervised pretraining from [42, 19] can be a viable option.如果有大量的应用域未标记数据可用，则[42,19]中的自我监督预训练可以是一个可行的选择。



One important aspect of a CNN encoder is its effective receptive field [21]. CNN编码器的一个重要方面是其有效的接收域[21]。Since the effective receptive field is not strictly bounded, the size of encoded patches x cannot be exactly defined. 由于有效接受域不是严格有界的，所以不能精确地定义编码的patch x的大小。 At the same time, anomalies have various sizes and shapes, and, ideally, they have to be processed with the variable receptive fields. 与此同时，异常有各种大小和形状，理想情况下，它们必须用可变的接受域处理。 To address the ambiguity between CNN receptive fields and anomaly variability, we adopt common multi-scale feature pyramid pooling approach. 为了解决CNN接受域与异常变异性之间的模糊性问题，我们采用了常用的多尺度特征金字塔融合方法。 Figure 2 shows that the feature vectors z k i ∈ R Dk , i ∈ {Hk ×Wk} are extracted by K pooling layers.图2显示了由k个池化层提取的特征向量z k i∈R Dk, i∈{Hk ×Wk}。 Pyramid pooling captures both local and global patch information with small and large receptive fields in the first and last CNN layers, respectively. 金字塔池分别在CNN的第一层和最后一层用小的和大的接受域捕获局部和全局的patch信息。For convenience, we number pooling layers in the last to first layer order.为了方便起见，我们按照最后到第一层的顺序给池化层编号。



##### 4.2. CFLOW decoders for likelihood estimation 

We use the general normalizing flow framework from Section 3.3 to estimate log-likelihoods of feature vectors z.  Hence, our generative decoder model g(θ) aims to fit true density pZ(z) by an estimated parameterized density pˆZ(z, θ) from (1). 因此，我们的生成解码器模型g(θ)的目标是用(1)中估计的参数化密度pˆz (z， θ)拟合真实密度pZ(z)。However, the feature vectors are assumed to be independent of their spatial location in the general framework.  然而，在一般框架中，假设特征向量独立于它们的空间位置。To increase efficacy of distribution modeling, we propose to incorporate spatial prior into g(θ) model using conditional flow framework.为了提高分布模型的有效性，我们提出将空间先验纳入使用条件流框架的g(θ)模型。In addition, we model pˆ k Z (z, θ) densities using K independent decoder models gk(θk) due to multi-scale feature pyramid pooling setup.此外，利用多尺度特征金字塔池化方法，利用k个独立解码器模型gk(θk)建立了pˆkz (Z， θ)密度模型。



Our conditional normalizing flow (CFLOW) decoder architecture is presented in Figure 2. 我们的条件归一化流(CFLOW)解码器架构如图2所示。 We generate a conditional vector c k i using a 2D form of conventional positional encoding (PE) [39]. 我们使用传统位置编码(PE)[39]的二维形式生成条件向量cki。 Each c k i ∈ R Ck contains sin and cos harmonics that are unique to its spatial location (hk, wk)i . 每个Ck i∈R Ck包含其空间位置(hk, wk)i所特有的sin和cos谐波。We extend unconditional flow framework to CFLOW by concatenating the intermediate vectors inside decoder coupling layers with the conditional vectors ci as in [2].我们将无条件流框架扩展到CFLOW，方法是将解码器耦合层中的中间向量与[2]中的条件向量ci连接起来。



Then, the kth CFLOW decoder contains a sequence of conventional coupling layers with the additional conditional input.然后，第k个CFLOW解码器包含一系列常规耦合层和附加的条件输入。 Each coupling layer comprises of fully-connected layer with (Dk+Ck)×(Dk+Ck) kernel, softplus activation and output vector permutations. 每个耦合层由(Dk+Ck)×(Dk+Ck)核、软加激活和输出载体排列的全连接层组成。 Usually, the conditional extension does not increase model size since Ck  Dk. For example, we use the fixed Ck = 128 in all our experiments. 通常，条件扩展不会增加模型大小，因为Ck  Dk。例如，我们在所有的实验中都使用固定的Ck = 128。Our CFLOW decoder has translation-equivariant architecture, because it slides along feature vectors extracted from the intermediate feature maps with kernel parameter sharing. 我们的CFLOW解码器具有平移等变结构，因为它沿着从中间特征映射中提取的特征向量滑动，并共享内核参数。As a result, both the encoder h(λ) and decoders gk(θk) have convolutional translation-equivariant architectures.因此，编码器h(λ)和解码器gk(θk)都具有卷积平移等变结构。



We train CFLOW-AD using a maximum likelihood objective, which is equivalent to minimizing loss defined by我们使用一个最大似然目标来训练CFLOW-AD，这个目标等价于最小化由定义的损失

![image-20220104161931119](imgs/image-20220104161931119.png)

where the random variable ui = g −1 (zi , ci , θ), the Jacobian Ji = ∇zg −1 (zi , ci , θ) for CFLOW decoder and an expectation operation in DKL is replaced by an empirical train dataset Dtrain of size N. For brevity, we drop the kth scale notation. The derivation is given in Appendix B.

随机变量的ui = g−1(子,ci,θ),雅可比矩阵霁=∇zg−1(子,ci,θ)CFLOW译码器和一个期望操作DKL被实证Dtrain训练数据集的大小n为简便起见,我们把k比例符号。推导在附录B中给出。



## 2. 学习过程整理

本小节是论文中3. Theoretical Background内容的解释。

### 2.1 CNN

略

### 2.2 马氏距离

参考：https://github.com/FelixFu520/README/blob/main/train/loss/distance.md

### 2.3 INN（可逆网络）

#### 2.3.1 李宏毅视频-[Flow-based Generative Model](https://www.youtube.com/watch?v=uXY18nzdSsM)

##### 生成模型

![image-20220106195409416](imgs/image-20220106195409416.png)

![image-20220106195505954](imgs/image-20220106195505954.png)

![image-20220106200124764](imgs/image-20220106200124764.png)



![image-20220106200200131](imgs/image-20220106200200131.png)

##### Flow-based 数学基础

![image-20220106200230708](imgs/image-20220106200230708.png)

###### 雅可比矩阵是什么？

![image-20220106200849564](imgs/image-20220106200849564.png)

###### 行列式是是什么？

![image-20220106201040805](imgs/image-20220106201040805.png)

行列式的含义是面积、体积、...

![image-20220106201245363](imgs/image-20220106201245363.png)

###### 什么是变分？

![image-20220106201426006](imgs/image-20220106201426006.png)

![image-20220106201639792](imgs/image-20220106201639792.png)

![image-20220106202126822](imgs/image-20220106202126822.png)

![image-20220106202433953](imgs/image-20220106202433953.png)

![image-20220107094505884](imgs/image-20220107094505884.png)

##### Flow-based Model

![image-20220107103436959](imgs/image-20220107103436959.png)

![image-20220107094652779](imgs/image-20220107094652779.png)

Flow-based Model 输入和输出维度一致

![image-20220107104648187](imgs/image-20220107104648187.png)

![image-20220107104751927](imgs/image-20220107104751927.png)

![image-20220107104944766](imgs/image-20220107104944766.png)

![image-20220107105247399](imgs/image-20220107105247399.png)

![image-20220107105502242](imgs/image-20220107105502242.png)

如何逆向呢？

![image-20220107105609830](imgs/image-20220107105609830.png)

如何计算雅可比的行列式呢？

![image-20220107110014405](imgs/image-20220107110014405.png)

![image-20220107110132772](imgs/image-20220107110132772.png)

针对图像，如何拆呢？

![image-20220107110309321](imgs/image-20220107110309321.png)

![image-20220107110725955](imgs/image-20220107110725955.png)

![image-20220107110933799](imgs/image-20220107110933799.png)

![image-20220107111156314](imgs/image-20220107111156314.png)

![image-20220107111258391](imgs/image-20220107111258391.png)

![image-20220107111359386](imgs/image-20220107111359386.png)

![image-20220107111410726](imgs/image-20220107111410726.png)

Glow用来做语音合成比较多

![image-20220107111427896](imgs/image-20220107111427896.png)

#### 2.3.2 技术喵视频 

[神经网络(十五）标准化流(normalizing flow) 与INN (Invertible Neural Networks)](https://www.youtube.com/watch?v=1qPaDhR2uMY)

![image-20220107111711950](imgs/image-20220107111711950.png)

![image-20220107112717513](imgs/image-20220107112717513.png)

![image-20220107112728262](imgs/image-20220107112728262.png)

![image-20220107112942250](imgs/image-20220107112942250.png)

![image-20220107113150142](imgs/image-20220107113150142.png)

![image-20220107113312965](imgs/image-20220107113312965.png)

![image-20220107113405094](imgs/image-20220107113405094.png)

![image-20220107113624481](imgs/image-20220107113624481.png)

![image-20220107113800828](imgs/image-20220107113800828.png)

流函数是构成NF的基本构件，那么如何选择流函数呢？

![image-20220107113943919](imgs/image-20220107113943919.png)

![image-20220107114023476](imgs/image-20220107114023476.png)

![image-20220107114221309](imgs/image-20220107114221309.png)

![image-20220107114445826](imgs/image-20220107114445826.png)

![image-20220107114748437](imgs/image-20220107114748437.png)

![image-20220107114833548](imgs/image-20220107114833548.png)

![image-20220107114911528](imgs/image-20220107114911528.png)

![image-20220107115043575](imgs/image-20220107115043575.png)

![image-20220107115108144](imgs/image-20220107115108144.png)

![image-20220107115236391](imgs/image-20220107115236391.png)

normalizing flow的本质是输入是分布，输出是分布，但是实际上，我们拿到的都是数字信号，都是离散的，所以不是模拟信号，这种情况下，对我们想要你和的这个分布，会产生一定的扭曲，所以通常加一步dequantiazation（反量化）。

![image-20220107115639552](imgs/image-20220107115639552.png)

论文推荐： L. Dinh, D. Krueger, and Y. Bengio, “NICE: Non-linear Independent Components Estimation,” in ICLR Workshop, 2015 L. Dinh, J. Sohl-Dickstein, and S. Bengio, “Density Estima-tion using Real NVP,” in ICLR, 2017. D. P. Kingma and P. Dhariwal, “Glow: Generative flow with invertible 1x1 convolutions,” in Advances in Neural Information Processing Systems, 2018  J. Ho, X. Chen, A. Srinivas, Y. Duan, and P. Abbeel, “Flow++: Improving flow-based generative models with variational dequantization and architecture design,” in Proceedings of the 36th International Conference on Machine Learning, ICML, 2019.  J. Behrmann, D. Duvenaud, and J.-H. Jacobsen, “Invertible residual networks,” in Proceedings of the 36th International Conference on Machine Learning, ICML, 2019.  C.-W. Huang, D. Krueger, A. Lacoste, and A. Courville, “Neural Autoregressive Flows,” in ICML, 2018  W. Grathwohl, R. T. Q Chen, J. Bettencourt, I. Sutskever, and D. Duvenaud, “FFJORD: Free-form continuous dynamics for scalable reversible generative models,” in ICLR, 2019 其它参考资料 [1] Normalizing Flows: An Introduction and Review of Current Methods [2] Introduction to Normalizing Flows (ECCV2020 Tutorial and cvpr 2021 tutorial) by Marcus Brubaker [3] LiLian Weng’s blog [https://lilianweng.github.io/lil-log/...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbWZvWndSUW54UzNWdy1aVTRTaUdnbThjZnVYUXxBQ3Jtc0trU0ZfamdETnFzQXBnVUd2cWhwNlkwLVc4cU1CYU5IakRCYnJQN3NJRDRHcWs4YjE4ZjNhVXVvU3RrVU9qRG5LTF9uQ3FkRExYNE5DR3FKdEpFc0dsOEhjaWE3REYwSnJBTzQ5a216LTVDTTNiRjM0Zw&q=https%3A%2F%2Flilianweng.github.io%2Flil-log%2F2018%2F10%2F13%2Fflow-based-deep-generative-models.html) [4] [https://github.com/janosh/awesome-nor...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbVU0SWRHUkhoQmg5WGVYZlMxMXdOcnkxVzBfd3xBQ3Jtc0ttX3Z3ZjczUURCeERuUUd6MmhGUzFJLWxudHRQMVNoTVhOSUs0TnZubnMweVlJNHU1VU5oT3plNDJiUlV6SjRwNkp1TWF1Ujc5M29nTHVFaDZEQnNqcFpXSU1wUUkxcVkzQnozM24wUjZKVElTQ2VDbw&q=https%3A%2F%2Fgithub.com%2Fjanosh%2Fawesome-normalizing-flows)
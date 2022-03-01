# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Copy From:

# dataset
from .datasets import MVTecDataset, ClsDataset, DetDataset, SegDataset

# classification
from .ClsDataloader import ClsDataloaderTrain, ClsDataloaderEval

# anomaly
from .AnomalyDataloader import MVTecDataloader



# SemanticSegmentation
from .SegDataloader import SegDataloaderTrain, SegDataloaderEval

# detection
from .DetDataloader import DetDataloaderTrain, DetDataloaderEval

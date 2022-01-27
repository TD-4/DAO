# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Copy From:

# dataset
from .datasets import MVTecDataset, ClsDataset, DetDataset, SegDataset, IQADataset

# anomaly
from .AnomalyDataloader import MVTecDataloader

# classification
from .ClsDataloader import ClsDataloaderTrain, ClsDataloaderEval

# SemanticSegmentation
from .SegDataloader import SegDataloaderTrain, SegDataloaderEval

# iqa
from .IQADataloader import IQADataloaderTrain, IQADataloaderEval

# detection
from .DetDataloader import DetDataloaderTrain, DetDataloaderEval

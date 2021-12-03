# _*_coding:utf-8_*_
# @auther:FelixFu
# @Date: 2021.4.14
# @github:https://github.com/felixfu520

import numpy as np
import torch

__all__ = ['Meter_Cls']


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = np.multiply(val, weight)
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum = np.add(self.sum, np.multiply(val, weight))
        self.count = self.count + weight
        self.avg = self.sum / self.count

    @property
    def value(self):
        return self.val

    @property
    def average(self):
        return np.round(self.avg, 5)


class Meter_Cls(object):
    def __init__(self, num_class=38):
        self.batch_time = AverageMeter()  # batch训练时间
        self.data_time = AverageMeter()  # 读取数据时间
        self.total_loss = AverageMeter()
        self.precision_top1, self.precision_top2 = AverageMeter(), AverageMeter()

        self.confusion_matrix = [[0 for j in range(num_class)] for i in range(num_class)]
        self.lr = 0

    def update(self, data_time=None, batch_time=None, total_loss=None,
               outputs=None, targets=None, lr=None):
        if batch_time is not None:
            self.batch_time.update(batch_time)
        if data_time is not None:
            self.data_time.update(data_time)
        if total_loss is not None:
            self.total_loss.update(total_loss)
        if outputs is not None and targets is not None:
            top1, top2 = self._eval_topk(outputs, targets, topk=(1, 2))
            self.precision_top1.update(top1.item())
            self.precision_top2.update(top2.item())
        if lr is not None:
            self.lr = lr

    def _eval_topk(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def eval_confusionMatrix(self, preds, labels):
        preds = torch.argmax(preds, 1)
        for t, p in zip(labels, preds):
            self.confusion_matrix[t][p] += 1
        return self.confusion_matrix

    def initialized(self, flag=False):
        if flag:
            self.batch_time.initialized = False  # batch训练时间
            self.data_time = False  # 读取数据时间
            self.total_loss = False
            self.precision_top1.initialized = False
            self.precision_top2.initialized = False

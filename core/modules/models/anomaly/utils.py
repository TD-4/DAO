# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Copy From:

import sys
import yaml
from tqdm import tqdm
from datetime import datetime

import torch
from torch import tensor
from torchvision import transforms

from PIL import ImageFilter
from sklearn import random_projection

class GaussianBlur:
    def __init__(self, radius: int = 4):
        self.radius = radius
        self.unload = transforms.ToPILImage()
        self.load = transforms.ToTensor()
        self.blur_kernel = ImageFilter.GaussianBlur(radius=4)

    def __call__(self, img):
        map_max = img.max()
        final_map = self.load(
            self.unload(img[0] / map_max).filter(self.blur_kernel)
        ) * map_max
        return final_map

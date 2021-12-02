#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @auther:FelixFu
# @Date: 2021.10.1
# @github:https://github.com/felixfu520
from core.modules.register import Registers


@Registers.cls_models.register
class classTwo:
    def __init__(self, info=None):
        print("info:{}".format(info))
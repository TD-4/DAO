#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @auther:FelixFu
# @Date: 2021.10.1
# @github:https://github.com/felixfu520
from core.modules.register import Registers


@Registers.backbones.register
class classOne:
    def __init__(self, info=None):
        print("info:{}".format(info))
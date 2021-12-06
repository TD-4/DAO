#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.


from dotmap import DotMap

from core.tools import register_modules
from core.trainers import ClsExport


def Export(config=None, custom_modules=None):
    exp = DotMap(config)
    register_modules(custom_modules=custom_modules)

    if exp.type == "cls":
        trainer = ClsExport(exp)
        trainer.export()

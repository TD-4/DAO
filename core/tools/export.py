#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.


from dotmap import DotMap
from loguru import logger

from core.tools import register_modules
from core.trainers import ClsExport, SegExport, DetExport


def Export(config=None, custom_modules=None):
    exp = DotMap(config)
    register_modules(custom_modules=custom_modules)

    if exp.type == "cls":
        trainer = ClsExport(exp)
        trainer.export()
    elif exp.type == "seg":
        trainer = SegExport(exp)
        trainer.export()
    elif exp.type == 'det':
        trainer = DetExport(exp)
        trainer.export()
    else:
        logger.error("this type {} is not supported, now supported cls, det, seg.".format(exp.type))


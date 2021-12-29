
# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.12.17
# @GitHub:https://github.com/felixfu520
# @Copy From:

from dotmap import DotMap
from loguru import logger

from core.tools import register_modules
from core.trainers import *


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
    elif exp.type == 'anomaly':
        trainer = AnomalyExport(exp)
        trainer.export()
    elif exp.type == 'iqa':
        trainer = IQAExport(exp)
        trainer.export()
    else:
        logger.error("this type {} is not supported, now supported cls, det, seg, anomaly, iqa.".format(exp.type))


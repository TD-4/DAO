
# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.12.17
# @GitHub:https://github.com/felixfu520
# @Copy From:

from loguru import logger
from dotmap import DotMap

from core.tools import register_modules
from core.trainers import *


@logger.catch
def Demo(config, custom_modules):
    exp = DotMap(config)
    register_modules(custom_modules=custom_modules)  # 注册所有组件

    if exp.type == "cls":
        trainer = ClsDemo(exp)
        trainer.demo()
    elif exp.type == "seg":
        trainer = SegDemo(exp)
        trainer.demo()
    elif exp.type == 'det':
        trainer = DetDemo(exp)
        trainer.demo()
    elif exp.type == 'anomaly':
        trainer = AnomalyDemo(exp)
        trainer.demo()
    elif exp.type == 'iqa':
        trainer = IQADemo(exp)
        trainer.demo()
    else:
        logger.error("this type {} is not supported, now supported cls, det, seg, anomaly, iqa.".format(exp.type))

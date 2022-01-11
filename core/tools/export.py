
# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.12.17
# @GitHub:https://github.com/felixfu520
# @Copy From:

from dotmap import DotMap
from loguru import logger

from core.tools import register_modules
from core.modules.register import Registers


def Export(config=None, custom_modules=None):
    exp = DotMap(config)
    register_modules(custom_modules=custom_modules)  # 注册所有组件

    logger.info("DAO support Cls/Det/Seg/Anomaly/IQAExport, And you can define custom Trainer")
    trainer = Registers.trainers.get(exp.trainer.type)(exp)  # 调用
    trainer.export()


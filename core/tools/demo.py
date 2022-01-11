
# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.12.17
# @GitHub:https://github.com/felixfu520
# @Copy From:

from loguru import logger
from dotmap import DotMap

from core.tools import register_modules
from core.modules.register import Registers


@logger.catch
def Demo(config, custom_modules):
    exp = DotMap(config)
    register_modules(custom_modules=custom_modules)  # 注册所有组件

    logger.info("DAO support Cls/Det/Seg/Anomaly/IQAExport, And you can define custom Trainer")
    trainer = Registers.trainers.get(exp.trainer.type)(exp)  # 调用
    trainer.demo()

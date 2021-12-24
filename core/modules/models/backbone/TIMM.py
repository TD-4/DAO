# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Copy From:

import timm
from loguru import logger
from core.modules.register import Registers

__all__ = ['TIMM']


@Registers.backbones.register
def TIMM(backbone):
    # 判断model是否在timm支持列表中
    if backbone.kwargs.model_name not in timm.list_models():
        logger.error("timm {} not supported {}".format(
            timm.__version__,
            backbone.kwargs.model_name))
        raise

    # 判断model是否有pretrained
    if backbone.kwargs.pretrained and backbone.kwargs.model_name not in timm.list_models(pretrained=True):
        logger.error("{} hasn't pretrained weight, please set pretrained False".format(
            backbone.kwargs.model_name
        ))
        raise

    model = timm.create_model(**backbone.kwargs)
    return model


# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.12.17
# @GitHub:https://github.com/felixfu520
# @Copy From:

import json
from loguru import logger
from core.tools import TrainVal, Eval, Demo, Export

__all__ = ['DAO', 'DAODict']


def DAO(exp_file, cus_file):
    """
    方法1：传入文件
    exp_file: config.json文件
    cus_file: custom.json文件
    """
    exp = json.load(open(exp_file))     # load config.json
    custom_modules = json.load(open(cus_file))  # load modules.json

    if exp_file[:-5].split('-')[-2] == "trainval":
        TrainVal(config=exp, custom_modules=custom_modules)
    elif exp_file[:-5].split('-')[-2] == "eval":
        Eval(config=exp, custom_modules=custom_modules)
    elif exp_file[:-5].split('-')[-2] == "demo":
        Demo(config=exp, custom_modules=custom_modules)
    elif exp_file[:-5].split('-')[-2] == "export":
        Export(config=exp, custom_modules=custom_modules)
    else:
        logger.error("this type {} is not supported, now supported trainval, eval, demo, export".format(
            exp_file[:-5].split('-')[-2])
        )


def DAODict(exp_file, cus_file, TEDE="trainval"):
    """
    方法2：传入字典
    exp_file: config.json文件内容
    cus_file: custom.json文件内容
    """
    exp = exp_file
    custom_modules = cus_file
    if exp_file[:-5].split('-')[-2] == "trainval":
        TrainVal(config=exp, custom_modules=custom_modules)
    elif exp_file[:-5].split('-')[-2] == "eval":
        Eval(config=exp, custom_modules=custom_modules)
    elif exp_file[:-5].split('-')[-2] == "demo":
        Demo(config=exp, custom_modules=custom_modules)
    elif exp_file[:-5].split('-')[-2] == "export":
        Export(config=exp, custom_modules=custom_modules)
    else:
        logger.error("this type {} is not supported, now supported trainval, eval, demo, export".format(
            exp_file[:-5].split('-')[-2])
        )

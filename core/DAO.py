
# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.12.17
# @GitHub:https://github.com/felixfu520
# @Copy From:
import os
import sys
import json
from loguru import logger
from core.tools import TrainVal, Eval, Demo, Export

__all__ = ['DAO', 'DAODict']


def DAO(exp_file, cus_file):
    """
    方法1：传入文件
    exp_file: config.json文件路径
    cus_file: custom.json文件路径
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


def DAODict(exp_dict, cus_dict):
    """
    方法2：传入字典
    exp_file: config.json文件内容
    cus_file: custom.json文件内容
    """
    type_ = exp_dict['fullName'].split('-')[-2]
    if type_ == "trainval":
        TrainVal(config=exp_dict, custom_modules=cus_dict)
    elif type_ == "eval":
        Eval(config=exp_dict, custom_modules=cus_dict)
    elif type_ == "demo":
        Demo(config=exp_dict, custom_modules=cus_dict)
    elif type_ == "export":
        Export(config=exp_dict, custom_modules=cus_dict)
    else:
        logger.error("this type {} is not supported, now supported trainval, eval, demo, export".format(type_))

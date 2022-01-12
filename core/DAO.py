
# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.12.17
# @GitHub:https://github.com/felixfu520
# @Copy From:

import json
import os
import sys
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 添加 core库 到 sys.path 中
from core.tools import TrainVal, Eval, Demo, Export


def DAO(exp_file, cus_file):

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


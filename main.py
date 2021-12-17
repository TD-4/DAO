
# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.12.17
# @GitHub:https://github.com/felixfu520
# @Copy From:

import argparse
import json
import os
import sys
from loguru import logger

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # 添加 core库 到 sys.path 中
from core.tools import TrainVal, Eval, Demo, Export


if __name__ == "__main__":
    parser = argparse.ArgumentParser("AI TrainVal Parser")
    parser.add_argument("-c", "--config",
                        default=None,
                        type=str,
                        help="please input your experiment description file")
    parser.add_argument("-m", "--custom",
                        default="/root/code/DAO/configs/super/custom_modules.json",
                        type=str,
                        help="please input your modules description file")

    exp = json.load(open(parser.parse_args().exp_file))     # load config.json
    custom_modules = json.load(open(parser.parse_args().cus_file))  # load modules.json

    if parser.parse_args().exp_file[:-5].split('-')[-2] == "trainval":
        TrainVal(config=exp, custom_modules=custom_modules)
    elif parser.parse_args().exp_file[:-5].split('-')[-2] == "eval":
        Eval(config=exp, custom_modules=custom_modules)
    elif parser.parse_args().exp_file[:-5].split('-')[-2] == "demo":
        Demo(config=exp, custom_modules=custom_modules)
    elif parser.parse_args().exp_file[:-5].split('-')[-2] == "export":
        Export(config=exp, custom_modules=custom_modules)
    else:
        logger.error("this type {} is not supported, now supported trainval, eval, demo, export".format(
            parser.parse_args().exp_file[:-5].split('-')[-2])
        )

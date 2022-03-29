
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
from core import DAO, DAODict


if __name__ == "__main__":
    parser = argparse.ArgumentParser("AI TrainVal Parser")
    parser.add_argument("-c", "--exp_file", default=None, type=str, help="please input your experiment description file")
    parser.add_argument("-m", "--cus_file", default="/ai/AI/server/DAO/configs/Modules/custom_modules.json", type=str, help="please input your modules description file")

    exp = json.load(open(parser.parse_args().exp_file))     # load config.json
    custom_modules = json.load(open(parser.parse_args().cus_file))  # load modules.json

    DAO(parser.parse_args().exp_file, parser.parse_args().cus_file)  # test DAO
    # DAODict(exp, custom_modules)  # test DAODict



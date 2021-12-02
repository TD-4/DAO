#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @auther:FelixFu
# @Date: 2021.10.1
# @github:https://github.com/felixfu520
import argparse
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # 添加 core库 到 sys.path 中
from core.tools import register_modules, TrainVal


if __name__ == "__main__":
    parser = argparse.ArgumentParser("AI TrainVal Parser")
    parser.add_argument("-c", "--exp_file", default="/root/code/DAO/configs/cls_trainval.json", type=str,
                        help="please input your experiment description file")
    parser.add_argument("-m", "--cus_file", default="/root/code/DAO/configs/custom_modules.json", type=str,
                        help="please input your experiment description file")

    exp = json.load(open(parser.parse_args().exp_file))
    custom_modules = json.load(open(parser.parse_args().cus_file))

    register_modules(custom_modules=custom_modules)   # 注册所有组件
    TrainVal(config=exp)

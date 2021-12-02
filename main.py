#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @auther:FelixFu
# @Date: 2021.10.1
# @github:https://github.com/felixfu520
import argparse
import json
from core.tools import TrainVal

if __name__ == "__main__":
    parser = argparse.ArgumentParser("AI TrainVal Parser")
    parser.add_argument("-c", "--exp_file", default="configs/cls_trainval.json", type=str,
                        help="please input your experiment description file")
    exp = json.load(open(parser.parse_args().exp_file))

    TrainVal(config=exp)

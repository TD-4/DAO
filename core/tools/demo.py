# #!/usr/bin/env python3
# # -*- coding:utf-8 -*-
# # Copyright (c) Megvii, Inc. and its affiliates.

from loguru import logger
from dotmap import DotMap

from core.tools import register_modules
from core.trainers import ClsDemo, SegDemo


@logger.catch
def Demo(config, custom_modules):
    exp = DotMap(config)
    register_modules(custom_modules=custom_modules)  # 注册所有组件

    if exp.type == "cls":
        trainer = ClsDemo(exp)
        results = trainer.demo()
        print(results)
    if exp.type == "seg":
        trainer = SegDemo(exp)
        results = trainer.demo()
        for i, r in enumerate(results):
            r.save("/root/code/img-{}.png".format(str(i)))

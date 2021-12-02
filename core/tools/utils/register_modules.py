#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @auther:FelixFu
# @Date: 2021.10.1
# @github:https://github.com/felixfu520
from core.modules.register import import_all_modules_for_register

__all__ = ['register_modules']


def register_modules(custom_modules=None):
    import_all_modules_for_register(custom_modules=custom_modules)

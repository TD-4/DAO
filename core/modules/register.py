#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @auther:FelixFu
# @Date: 2021.10.1
# @github:https://github.com/felixfu520

"""Module register."""

__all__ = ['import_all_modules_for_register', 'Registers']

import os
import sys
import importlib
from loguru import logger


class Register(object):
    """Module register"""
    def __init__(self, register_name):
        self._dict = {}
        self._name = register_name

    def __setitem__(self, key, value):
        if not callable(value):
            raise Exception("Value of a Registry must be a callable.")
        if key is None:
            key = value.__name__
        if key in self._dict:
            logger.warning("Key {} already in registry {}.".format(key, self._name))
        self._dict[key] = value

    def register(self, param):
        """Decorator to register a function or class."""

        def decorator(key, value):
            self[key] = value
            return value

        if callable(param):
            # @reg.register
            return decorator(None, param)
        # @reg.register('alias')
        return lambda x: decorator(param, x)

    def __getitem__(self, key):
        try:
            return self._dict[key]
        except Exception as e:
            logger.error(f"module {key} not found: {e}")
            raise e

    def __contains__(self, key):
        return key in self._dict

    def keys(self):
        """key"""
        return self._dict.keys()

    def get(self, key):
        return self.__getitem__(key)


class Registers:
    """All module registers."""

    def __init__(self):
        raise RuntimeError("Registries is not intended to be instantiated")

    backbones = Register("backbones")       # 主干网络
    cls_models = Register("cls_models")     # 分类模型
    seg_models = Register("seg_models")     # 分割模型
    det_models = Register("det_models")     # 目标检测模型
    optims = Register("optim")              # 优化器
    datasets = Register("datasets")         # 数据集
    dataloaders = Register("dataloaders")   # 数据加载器
    losses = Register("losses")             # 损失函数
    schedulers = Register("schedulers")     # 学习调整策略


BACKBONES = ["EfficientNet"]
CLS_MODELS = ["EfficientNet"]
TRAINERS = ["trainerCls"]
ALL_MODULES = [
    ("core.modules.models.backbone", BACKBONES),
    ("core.modules.models.cls", CLS_MODELS),
]


def import_all_modules_for_register(custom_modules=None):
    """Import all modules for register."""
    all_modules = ALL_MODULES

    _add_custom_modules(all_modules, custom_modules)     # 添加自定义modules
    logger.info(f"All modules: {all_modules}")

    errors = []
    for base_dir, modules in all_modules:
        for name in modules:
            try:
                if base_dir != "":
                    full_name = base_dir + "." + name
                else:
                    full_name = name
                importlib.import_module(full_name)
                logger.info(f"{full_name} loaded.")
            except ImportError as error:
                errors.append((name, error))
    _handle_errors(errors)


def _add_custom_modules(all_modules, custom_modules):
    """Add custom modules to all_modules"""
    if custom_modules is not None and "custom_modules" in custom_modules:
        custom_modules = custom_modules["custom_modules"]
        if not isinstance(custom_modules, list):
            custom_modules = [custom_modules]
        for module in custom_modules:
            custom_modules_path = os.path.join(module[:module.rfind("/")])
            sys.path.append(custom_modules_path)
            all_modules.append(("", [_path_to_module_format(module)]))


def _path_to_module_format(py_path):
    """Transform a python file path to module format."""
    return py_path.split("/")[-1].rstrip(".py")


def _handle_errors(errors):
    """Log out and possibly reraise errors during import."""
    if not errors:
        return
    for name, err in errors:
        logger.error("Module {} import failed: {}".format(name, err))
    logger.error("Please check these modules.")








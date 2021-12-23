# _*_coding:utf-8_*_
# @auther:FelixFu
# @Data:2021.4.16
# @github:https://github.com/felixfu520

__all__ = ['TIMMC']

from core.modules.register import Registers


@Registers.cls_models.register
def TIMMC(backbone_kwargs):
    backbone = Registers.backbones.get("TIMM")(backbone_kwargs)
    return backbone

# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Copy From:

from core.modules.register import Registers

@Registers.evaluators.register
class ClsEvaluator:
    def __init__(
            self,
            type_=None,
            dataset=None,
            num_classes=None,
            **kwargs
    ):
        self.dataloader = Registers.dataloaders.get(type_)(
            is_distributed=is_distributed,
            dataset=dataset,
            **kwargs
        )
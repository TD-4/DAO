# _*_coding:utf-8_*_
# @auther:FelixFu
# @Data:2021.4.16
# @github:https://github.com/felixfu520
__all__ = ['EfficientNet']


from core.modules.models.backbone.EfficientNet import EfficientNet as torchEfficientNet
from core.modules.register import Registers


@Registers.cls_models.register
def EfficientNet(model_name="Efficientnet-b0",
                 weights_path=None,
                 in_channels=3,
                 num_classes=1000, **kwargs):
    model = torchEfficientNet.from_pretrained(model_name=model_name, weights_path=weights_path, in_channels=in_channels,
                                              num_classes=num_classes, **kwargs)
    model.set_swish(memory_efficient=False)
    return model

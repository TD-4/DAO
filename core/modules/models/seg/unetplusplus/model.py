#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/11/22 13:41
# @Author  : yangqiang
# @Software: PyCharm
# from .utils import SegmentationModel
# from .utils import SegmentationHead
# import torch.nn as nn
# from .utils import modules as md
# import torch


from core.modules.models.seg.base import SegmentationModel, SegmentationHead
from core.modules.models.seg.unetplusplus.decoder import UnetPlusPlusDecoder
from core.modules.register import Registers


@Registers.seg_models.register
class UnetPlusPlus(SegmentationModel):
    def __init__(
        self,
        encoder,
        num_classes=2,
        encoder_channels=None
    ):
        super().__init__()

        self.encoder = Registers.backbones.get("TIMM")(encoder)
        self.decoder = UnetPlusPlusDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=[256, 128, 64, 32, 16],
            n_blocks=5,
            use_batchnorm=True,
            center=False,
            attention_type=None,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=16,
            out_channels=num_classes,
            activation=None,
            kernel_size=3,
        )
        self.classification_head = None
        self.name = "unetplusplus-{}".format(encoder.kwargs.model_name)
        self.initialize()


if __name__ == '__main__':
    import torch
    from dotmap import DotMap
    model_kwargs = DotMap({
        "type": "UnetPlusPlus",
        "summary_size": [1, 224, 224],
        "backbone": {
            "kwargs": {
                "model_name": "tf_mobilenetv3_small_075",
                "pretrained": True,
                "checkpoint_path": "",
                "exportable": True,
                "in_chans": 3,
                "num_classes": 2,
                "features_only": True
            }
        },
        "kwargs": {
            "num_classes": 2,
            "encoder_channels": [1, 16, 16, 24, 40, 432]

        }
    })
    x = torch.rand(8, 3, 256, 256)
    model = UnetPlusPlus(model_kwargs.backbone, **model_kwargs.kwargs)

    output = model(x)
    print(output.shape)
    # model.eval()
    # torch.onnx.export(model,
    #                   x,
    #                   "onnx_name.onnx",
    #                   opset_version=9,
    #                   input_names=["input"],
    #                   output_names=["output"])

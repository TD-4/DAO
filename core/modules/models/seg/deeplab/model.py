# -*- coding: utf-8 -*-
# @Author:FelixFu
# @Date: 2021.4.14
# @GitHub:https://github.com/felixfu520
# @Copy From:
import torch.nn as nn

from typing import Optional
from core.modules.models.seg.base import SegmentationModel, SegmentationHead, ClassificationHead
from core.modules.models.seg.deeplab.decoder import DeepLabV3Decoder, DeepLabV3PlusDecoder
from core.modules.register import Registers


@Registers.seg_models.register
class DeepLabV3(SegmentationModel):
    """DeepLabV3_ implementation from "Rethinking Atrous Convolution for Semantic Image Segmentation"
    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_channels: A number of convolution filters in ASPP module. Default is 256
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
            Default is **None**
        upsampling: Final upsampling factor. Default is 8 to preserve input-output spatial shape identity
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax" (could be **None** to return logits)
    Returns:
        ``torch.nn.Module``: **DeepLabV3**
    .. _DeeplabV3:
        https://arxiv.org/abs/1706.05587
    """

    def __init__(
            self,
            encoder=None,
            encoder_channels=None,
            decoder_channels: int = 256,
            num_classes: int = 1,
            activation: Optional[str] = None,
            upsampling: int = 8,
            aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = Registers.backbones.get("TIMM")(encoder)

        self.decoder = DeepLabV3Decoder(
            in_channels=encoder_channels[-1],
            out_channels=decoder_channels,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=num_classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=encoder_channels[-1], **aux_params
            )
        else:
            self.classification_head = None


@Registers.seg_models.register
class DeepLabV3Plus(SegmentationModel):
    """DeepLabV3+ implementation from "Encoder-Decoder with Atrous Separable
    Convolution for Semantic Image Segmentation"

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        encoder_output_stride: Downsampling factor for last encoder features (see original paper for explanation)
        decoder_atrous_rates: Dilation rates for ASPP module (should be a tuple of 3 integer values)
        decoder_channels: A number of convolution filters in ASPP module. Default is 256
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
            Default is **None**
        upsampling: Final upsampling factor. Default is 4 to preserve input-output spatial shape identity
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax" (could be **None** to return logits)
    Returns:
        ``torch.nn.Module``: **DeepLabV3Plus**

    Reference:
        https://arxiv.org/abs/1802.02611v3
    """

    def __init__(
            self,
            encoder=None,
            encoder_channels=None,
            encoder_output_stride: int = 32,
            decoder_channels: int = 256,
            decoder_atrous_rates: tuple = (12, 24, 36),
            num_classes: int = 1,
            activation: Optional[str] = None,
            upsampling: int = 4,
            aux_params: Optional[dict] = None,
    ):
        super().__init__()

        # if encoder_output_stride not in [8, 16]:
        #     raise ValueError(
        #         "Encoder output stride should be 8 or 16, got {}".format(encoder_output_stride)
        #     )

        self.encoder = Registers.backbones.get("TIMM")(encoder)

        self.decoder = DeepLabV3PlusDecoder(
            encoder_channels=encoder_channels,
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            output_stride=encoder_output_stride,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=num_classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=encoder_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

if __name__ == '__main__':
    import torch
    from dotmap import DotMap
    model_kwargs = DotMap({
        "type": "DeepLabV3",
        "summary_size": [1, 224, 224],
        "backbone": {
            "kwargs": {
                "model_name": "resnet34",
                "pretrained": True,
                "checkpoint_path": "",
                "exportable": True,
                "in_chans": 3,
                "features_only": True
            }
        },
        "kwargs": {
            "encoder_channels": [3, 64, 64, 128, 256, 512],
            "decoder_channels": 256,
            "num_classes": 21,
            "upsampling": 4
        }
    })
    x = torch.rand(8, 3, 256, 256)
    model = DeepLabV3Plus(model_kwargs.backbone, **model_kwargs.kwargs)

    output = model(x)
    print(output.shape)
    model.eval()
    torch.onnx.export(model,
                      x,
                      "/root/code/onnx_name.onnx",
                      opset_version=13,
                      input_names=["input"],
                      output_names=["output"])
# if __name__ == '__main__':
#     import torch
#     from dotmap import DotMap
#     model_kwargs = DotMap({
#         "type": "DeepLabV3",
#         "summary_size": [1, 224, 224],
#         "backbone": {
#             "kwargs": {
#                 "model_name": "tf_mobilenetv3_small_075",
#                 "pretrained": True,
#                 "checkpoint_path": "",
#                 "exportable": True,
#                 "in_chans": 3,
#                 "features_only": True
#             }
#         },
#         "kwargs": {
#             "encoder_channels": [3, 16, 16, 24, 40, 432],
#             "decoder_channels": 256,
#             "num_classes": 21,
#             "upsampling": 32
#         }
#     })
#     x = torch.rand(8, 3, 256, 256)
#     model = DeepLabV3(model_kwargs.backbone, **model_kwargs.kwargs)
#
#     output = model(x)
#     print(output.shape)
#     model.eval()
#     torch.onnx.export(model,
#                       x,
#                       "/root/code/onnx_name.onnx",
#                       opset_version=9,
#                       input_names=["input"],
#                       output_names=["output"])
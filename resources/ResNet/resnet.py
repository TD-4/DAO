# Original code and checkpoints by Hang Zhang
# https://github.com/zhanghang1989/PyTorch-Encoding

import math
import torch
import os
import sys
import zipfile
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve
from core.modules.register import Registers

model_urls = {
    'resnet18': 'resnet18-5c106cde.pth',
    'resnet34': 'resnet34-333f7ec4.pth',
    'resnet50': 'resnet50s-a75c83cf.pth',
    'resnet101': 'resnet101s-03a0f310.pth',
    'resnet152': 'resnet152s-36670e8b.pth'
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """ResNet BasicBlock"""
    expansion = 1  # 是否要膨胀输出通道，basicblock不需要膨胀，可忽略

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, previous_dilation=1,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=previous_dilation, dilation=previous_dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        # 此basicblock的第一个CBR
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 此basicblock的第二个CBR
        out = self.conv2(out)
        out = self.bn2(out)

        # 如果此basicblock缩小了特征图尺寸，需要将residual 下采样
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """ResNet Bottleneck
    """
    # pylint: disable=unused-argument
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1,
                 downsample=None, previous_dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride,
            padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def _sum_each(self, x, y):
        assert(len(x) == len(y))
        z = []
        for i in range(len(x)):
            z.append(x[i]+y[i])
        return z

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """Dilated Pre-trained ResNet Model, which preduces the stride of 8 featuremaps at conv5.

    Reference:
        - He, Kaiming, et al. "Deep residual learning for image recognition." CVPR. 2016.
        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """
    # pylint: disable=unused-variable
    def __init__(self, block, layers, num_classes=1000, in_channels=3, dilated=True, multi_grid=False,
                 deep_base=True, norm_layer=nn.BatchNorm2d, feature_only=False):
        self.feature_only = feature_only
        self.inplanes = 128 if deep_base else 64
        super(ResNet, self).__init__()
        if deep_base:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        if dilated:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                           dilation=2, norm_layer=norm_layer)
            if multi_grid:
                self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                               dilation=4, norm_layer=norm_layer,
                                               multi_grid=True)
            else:
                self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                               dilation=4, norm_layer=norm_layer)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None, multi_grid=False):
        downsample = None

        # --------------此layer中的 第一个basic/bottleblock
        # 判断此basic/bottleblock是否进行下采样，即如果步长不为1，且输入通道数不等于输出通道数（算上膨胀的）===>即判断basic/bottleblock
        # 是否是第一个basic/bottleblock
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        multi_dilations = [4, 8, 16]
        if multi_grid:  # 这个分支是干嘛的？好像是deeplab网络会用到
            layers.append(block(self.inplanes, planes, stride, dilation=multi_dilations[0],
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        elif dilation == 1 or dilation == 2:  # 分支正常走
            layers.append(block(self.inplanes, planes, stride, dilation=1,
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=2,
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))
        # --------------此layer中的 第一个basic/bottleblock END

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if multi_grid:
                layers.append(block(self.inplanes, planes, dilation=multi_dilations[i],
                                    previous_dilation=dilation, norm_layer=norm_layer))
            else:
                layers.append(block(self.inplanes, planes, dilation=dilation, previous_dilation=dilation,
                                    norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        outputs = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        outputs.append(x)

        x = self.layer2(x)
        outputs.append(x)

        x = self.layer3(x)
        outputs.append(x)

        x = self.layer4(x)
        outputs.append(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.feature_only:
            return outputs
        return x


@Registers.backbones.register
def resnet18(pretrained=False, feature_only=False, num_classes=1000, in_channels=3, **kwargs):

    model = ResNet(BasicBlock, [2, 2, 2, 2], feature_only=feature_only, in_channels=in_channels, num_classes=num_classes, deep_base=False, **kwargs)
    if pretrained:
        import core
        pth_path = os.path.dirname(os.path.dirname(os.path.abspath(core.__file__)))
        pth_path = os.path.join(pth_path, "pretrained", model_urls['resnet18'])
        model.load_state_dict(torch.load(pth_path))
    return model


@Registers.backbones.register
def resnet34(pretrained=False, feature_only=False, num_classes=1000, in_channels=3, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        feature_only(bool): If True, forward return feature only
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], feature_only=feature_only, in_channels=in_channels, num_classes=num_classes, deep_base=False, **kwargs)
    if pretrained:
        import core
        pth_path = os.path.dirname(os.path.dirname(os.path.abspath(core.__file__)))
        pth_path = os.path.join(pth_path, "pretrained", model_urls['resnet34'])
        model.load_state_dict(torch.load(pth_path))
    return model


@Registers.backbones.register
def resnet50(pretrained=False, feature_only=False, num_classes=1000, in_channels=3, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        feature_only(bool): If True, forward return feature only
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], feature_only=feature_only, in_channels=in_channels, num_classes=num_classes, **kwargs)
    if pretrained:
        import core
        pth_path = os.path.dirname(os.path.dirname(os.path.abspath(core.__file__)))
        pth_path = os.path.join(pth_path, "pretrained", model_urls['resnet50'])
        model.load_state_dict(torch.load(pth_path))
    return model


@Registers.backbones.register
def resnet101(pretrained=False, feature_only=False, num_classes=1000, in_channels=3, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        feature_only(bool): If True, forward return feature only
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], feature_only=feature_only, in_channels=in_channels, num_classes=num_classes, **kwargs)
    if pretrained:
        import core
        pth_path = os.path.dirname(os.path.dirname(os.path.abspath(core.__file__)))
        pth_path = os.path.join(pth_path, "pretrained", model_urls['resnet101'])
        model.load_state_dict(torch.load(pth_path))
    return model


@Registers.backbones.register
def resnet152(pretrained=False, feature_only=False, num_classes=1000, in_channels=3, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], feature_only=feature_only, in_channels=in_channels, num_classes=num_classes, **kwargs)
    if pretrained:
        import core
        pth_path = os.path.dirname(os.path.dirname(os.path.abspath(core.__file__)))
        pth_path = os.path.join(pth_path, "pretrained", model_urls['resnet152'])
        model.load_state_dict(torch.load(pth_path))
    return model


if __name__ == "__main__":
    from torchsummary import summary
    import torch
    from torchvision import transforms
    from PIL import Image

    img_path = "/root/data/DAO/005982.jpg"
    model = resnet152(pretrained=True, feature_only=True)
    print(model)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()]
    )

    img = Image.open(img_path)
    img = transform(img)

    x = torch.unsqueeze(img, dim=0)

    # model = resnet18(pretrained=True, feature_only=True)    # dilated设置为Fasle， 否则FC层不对应，dilated在deeplab上会用得到
    # model = resnet101(dilated=False)    # dilated设置为Fasle， 否则FC层不对应，dilated在deeplab上会用得到
    model_result = model(x)
    print(model_result)
    summary(model, (3,224,224), device="cpu")
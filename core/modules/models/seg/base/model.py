import torch
from . import initialization as init


class SegmentationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = None  # backbone提取的不同层次的特征
        self.decoder_output = None

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x):
        self.features = [x, ] + self.encoder(x)  # backbone提取的不同层次的特征
        self.decoder_output = self.decoder(*self.features)  # 解码后的特征，一般是[batch_size, c, height, width]，c是输出维度
        # 对解码后的特征做个卷积得到最后结果[batch_size, num_class, height, width]
        return self._forward_seg()

    def _forward_cls(self):
        return self.classification_head(self.decoder_output)

    def _forward_det(self):
        return self.detection_head(self.features[3:])

    def _forward_seg(self):
        return self.segmentation_head(self.decoder_output)

    def predict(self, x):
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x

from models.basic import BasicModule
from config import config

import torch
import torchvision
from torch import nn


class ResNet18(BasicModule):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model_name = "ResNet18"

        # 设置预训练模型下载路径
        torch.hub.set_dir(config.working_root)
        # 加载预训练模型
        # self.model = torchvision.models.resnet18(pretrained=True) # 旧
        self.model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)

        # 调整第一层卷积层以适应 CIFAR-10 的输入尺寸 (32x32)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # 替换最后的全连接层以适应 CIFAR-10 的类别数 (10)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    x = torch.randn(32, 3, 32, 32)
    model = ResNet18()
    y = model(x)
    print(y.shape)

from models.basic import BasicModule

import torch
from torch import nn


class AlexNet(BasicModule):
    def __init__(self):
        super().__init__()
        self.model_name = "AlexNet"

        self.module = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=64 * 4 * 4, out_features=64),
            nn.Linear(in_features=64, out_features=10),
        )

    def forward(self, x):
        x = self.module(x)
        return x


# 验证网络正确性
if __name__ == '__main__':
    classification = AlexNet()
    # 按照batch_size=64，channel=3，size=32*32输入
    inputs = torch.ones((64, 3, 32, 32))
    outputs = classification(inputs)
    print(outputs.shape)

from models.basic import BasicModule
import torch
from torch import nn
from torch.nn import functional as F


class Residual(BasicModule):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return F.relu(x)


def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


class ResNet(BasicModule):
    def __init__(self):
        super().__init__()
        self.model_name = "ResNet"

        b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                           nn.BatchNorm2d(64), nn.ReLU(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        b3 = nn.Sequential(*resnet_block(64, 128, 2))
        b4 = nn.Sequential(*resnet_block(128, 256, 2))
        b5 = nn.Sequential(*resnet_block(256, 512, 2))

        self.net = nn.Sequential(
            b1,
            b2,
            b3,
            b4,
            b5,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.net(x)
        return x


if __name__ == "__main__":
    rblk = Residual(3, 3)
    x = torch.randn(32, 3, 32, 32)
    y = rblk(x)
    print(y.shape)  # torch.Size([32, 3, 32, 32]) 相同

    rblk = Residual(3, 6, use_1x1conv=True, strides=2)
    x = torch.randn(32, 3, 32, 32)
    y = rblk(x)
    print(y.shape)  # torch.Size([32, 6, 16, 16]) 增加输出通道数的同时，减半输出的高和宽

    model = ResNet()
    print(model)
    x = torch.randn(32, 3, 32, 32)
    y = model(x)
    print(y.shape)

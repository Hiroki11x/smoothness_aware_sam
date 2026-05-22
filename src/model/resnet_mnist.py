# define resnet18 for MNIST

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet18_1_Mnist(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18_1_Mnist, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        # strides = [stride] + [1]*(num_blocks-1)
        strides = [1] * num_blocks
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        # out = F.avg_pool2d(out, 4)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out
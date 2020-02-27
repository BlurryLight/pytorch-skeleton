#! /usr/bin/env python
#! coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 128)  # 16 channels, 4 * 4 feature maps
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))  # 28×28 -> (28 -5 +1) = 24x24
        out = F.max_pool2d(out, 2)  # 24*24 -> 12*12
        out = F.relu(self.conv2(out))  # 12×12 -> (12 -5 +1) = 8*8
        out = F.max_pool2d(out, 2)  # 8×8 -> 4*4
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


# test
if __name__ == '__main__':
    test = torch.rand([8, 1, 28, 28])
    print(test.shape)
    net = LeNet()
    res = net(test)
    print(res.shape)
    print(res.max(1))

#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 09-May-2023
#
# ---------------------------------------------------------------------------
# Implementation of LeNet based classifier
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.basenet import BaseNet

class LeNet(BaseNet):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.rep_dim = 2304

        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04)

        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04)

        self.conv3 = nn.Conv2d(64, 128, 5, padding=2)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04)

        self.conv4 = nn.Conv2d(128, 256, 5, padding=2)
        self.bn2d4 = nn.BatchNorm2d(256, eps=1e-04)

        self.fc1 = nn.Linear(256 * 6 * 6, self.rep_dim)
        self.bn1d = nn.BatchNorm1d(self.rep_dim, eps=1e-04)

        self.fc2 = nn.Linear(self.rep_dim, 10)

    def forward(self, x):

        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))

        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))

        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))

        x = self.conv4(x)
        x = self.pool(F.leaky_relu(self.bn2d4(x)))

        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))

        x = self.fc2(x)

        # x = torch.log_softmax(self.fc2(x))

        return x
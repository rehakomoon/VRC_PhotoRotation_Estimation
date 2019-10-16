# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 18:09:14 2019

@author: rehakomoon
"""

import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3) # 256x256x3 -> 254x254x32
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2) # 254x254x32 -> 127x127x32
        self.dp1 = nn.Dropout2d(p=0.3)
        self.conv2 = nn.Conv2d(32, 32, 3) # 127x127x32 -> 125x125x32
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2) # 125x125x32 -> 62x62x32
        self.dp2 = nn.Dropout2d(p=0.3)
        self.conv3 = nn.Conv2d(32, 64, 3) # 62x62x32 -> 60x60x64
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2) # 60x60x64 -> 30x30x64
        self.dp3 = nn.Dropout2d(p=0.3)
        self.conv4 = nn.Conv2d(64, 64, 3) # 30x30x64 -> 28x28x64
        self.bn4 = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(2, 2) # 28x28x64 -> 14x14x64
        self.dp4 = nn.Dropout2d(p=0.3)
        self.conv5 = nn.Conv2d(64, 128, 3) # 14x14x64 -> 12x12x128
        self.bn5 = nn.BatchNorm2d(128)
        self.pool5 = nn.MaxPool2d(2, 2) # 12x12x128 -> 6x6x128
        self.dp5 = nn.Dropout2d(p=0.3)
        
        self.fc1 = nn.Linear(6*6*128, 512)
        self.dp6 = nn.Dropout2d(p=0.3)
        self.fc2 = nn.Linear(512, 128)
        self.dp7 = nn.Dropout2d(p=0.3)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dp1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dp2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.dp3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)
        x = self.dp4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.pool5(x)
        x = self.dp5(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dp6(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dp7(x)
        x = self.fc3(x)

        return x
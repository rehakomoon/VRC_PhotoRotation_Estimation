# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 18:09:14 2019

@author: rehakomoon
"""

import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        dp_prob = 0.3
        
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        #self.dp1 = nn.Dropout2d(p=dp_prob)
        self.conv2 = nn.Conv2d(16, 16, 3)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2)
        #self.dp2 = nn.Dropout2d(p=dp_prob)
        self.conv3 = nn.Conv2d(16, 64, 3)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dp3 = nn.Dropout2d(p=dp_prob)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dp4 = nn.Dropout2d(p=dp_prob)
        self.conv5 = nn.Conv2d(64, 128, 3)
        self.bn5 = nn.BatchNorm2d(128)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.dp5 = nn.Dropout2d(p=dp_prob)
        self.conv6 = nn.Conv2d(128, 128, 3)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool6 = nn.MaxPool2d(2, 2)
        self.dp6 = nn.Dropout2d(p=dp_prob)
        self.conv7 = nn.Conv2d(128, 256, 3)
        self.bn7 = nn.BatchNorm2d(256)
        self.pool7 = nn.MaxPool2d(2, 2)
        self.dp7 = nn.Dropout2d(p=dp_prob)
        """
        self.conv8 = nn.Conv2d(256, 256, 3)
        self.bn8 = nn.BatchNorm2d(256)
        self.pool8 = nn.MaxPool2d(2, 2)
        self.dp8 = nn.Dropout2d(p=dp_prob)
        self.conv9 = nn.Conv2d(256, 512, 3)
        self.bn9 = nn.BatchNorm2d(512)
        self.pool9 = nn.MaxPool2d(2, 2)
        self.dp9 = nn.Dropout2d(p=dp_prob)
        """
        self.gpool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(256, 256)
        self.fdp1 = nn.Dropout2d(p=dp_prob)
        self.fc2 = nn.Linear(256, 64)
        self.fdp2 = nn.Dropout2d(p=dp_prob)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        #x = self.dp1(x)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        #x = self.dp2(x)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dp3(x)
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.dp4(x)
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))
        x = self.dp5(x)
        x = self.pool6(F.relu(self.bn6(self.conv6(x))))
        x = self.dp6(x)
        x = self.pool7(F.relu(self.bn7(self.conv7(x))))
        x = self.dp7(x)
        #x = self.pool8(F.relu(self.bn8(self.conv8(x))))
        #x = self.dp8(x)
        #x = self.pool9(F.relu(self.bn9(self.conv9(x))))
        #x = self.dp9(x)
        x = self.gpool(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fdp1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fdp2(x)
        x = self.fc3(x)

        return x
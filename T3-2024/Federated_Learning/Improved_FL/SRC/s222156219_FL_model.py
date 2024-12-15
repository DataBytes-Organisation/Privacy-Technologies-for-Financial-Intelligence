# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/18 15:47
@Auth ： Chuang Liu
@Email ：LIUC0316@126.COM
@File ：model.py
@IDE ：PyCharm
"""
from torch import nn
class MyNet(nn.Module):
    def __init__(self, args, name):
        super(MyNet, self).__init__()
        self.name = name
        self.len = 0
        self.loss = 0
        self.fc1 = nn.Linear(args.input_dim, 48)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(48, 96)
        self.fc3 = nn.Linear(96, 24)
        self.fc4 = nn.Linear(24, 2)
        self.fc5 = nn.Linear(2, 1)

    def forward(self, data):
        x = self.fc1(data)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        x = self.fc5(x)
        x = self.sigmoid(x)

        return x
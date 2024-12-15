# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/18 15:49
@Auth ： Chuang Liu
@Email ：LIUC0316@126.COM
@File ：get_data.py
@IDE ：PyCharm
"""
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append('../')
def normalize_data(file_name):
    """
    parameters:
    param file_name: csv file name
    return:
    dataframe: normalized data
    """
    df = pd.read_csv('./data/Loan/' + file_name + '.csv', encoding='gbk')
    columns = df.columns
    # normalization
    for i in range(12):
        MAX = np.max(df[columns[i]])
        MIN = np.min(df[columns[i]])
        df[columns[i]] = (df[columns[i]] - MIN) / (MAX - MIN)

    return df

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

def load_client_data(file_name, B):
    """
    parameters:
    file_name: csv file name
    param B: batch size
    return:
    DataLoader: train data and val data
    """
    data = normalize_data(file_name)
    data = data.values.tolist()
    features = []
    for i in range(len(data)):
        train_features = []
        train_label = []
        train_features.append(data[i][:-1])
        train_label.append(data[i][-1:])
        train_features = torch.FloatTensor(train_features).view(-1)
        train_label = torch.FloatTensor(train_label).view(-1)
        features.append((train_features, train_label))

    Data_train = features[:int(len(features) * 0.8)]
    Data_val = features[int(len(features) * 0.8):]

    train_len = int(len(Data_train) / B) * B
    val_len = int(len(Data_val) / B) * B
    Data_train, Data_val = Data_train[:train_len], Data_val[:val_len]

    train = MyDataset(Data_train)
    val = MyDataset(Data_val)

    Data_train = DataLoader(dataset=train, batch_size=B, shuffle=True, num_workers=0)
    Data_val = DataLoader(dataset=val, batch_size=B, shuffle=False, num_workers=0)

    return Data_train, Data_val

def load_test_data(file_name, B):
    """
    parameters:
    file_name: csv file name
    param B: batch size
    return:
    DataLoader: test data
    """
    data = normalize_data(file_name)
    data = data.values.tolist()
    features = []
    for i in range(len(data)):
        train_features = []
        train_label = []
        train_features.append(data[i][:-1])
        train_label.append(data[i][-1:])
        train_features = torch.FloatTensor(train_features).view(-1)
        train_label = torch.FloatTensor(train_label).view(-1)
        features.append((train_features, train_label))

    Data_test = features[:]
    test_len = int(len(Data_test) / B) * B
    Data_test = Data_test[:test_len]
    test = MyDataset(Data_test)
    Data_test = DataLoader(dataset=test, batch_size=B, shuffle=False, num_workers=0)

    return Data_test
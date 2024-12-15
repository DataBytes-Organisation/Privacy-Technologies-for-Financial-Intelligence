# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/18 15:47
@Auth ： Chuang Liu
@Email ：LIUC0316@126.COM
@File ：client.py
@IDE ：PyCharm
"""

from collections import OrderedDict
import numpy as np
import torch
from torch import nn
import copy
from tqdm import tqdm

from dataset import load_client_data, load_test_data

def get_data_batch(args, data):
    """
    get a random batch of data.

    parameters:
    args: hyperparameters
    data: data of dataloader
    return:
    data: one batch data of dataloader
    """
    ind = np.random.randint(0, high=len(data), size=None, dtype=int)
    seq, label = data[ind]
    seq, label = seq.to(args.device), label.to(args.device)

    return seq, label

def compute_grad(args, model,
                 data_batch,
                 v=None,
                 second_order_grads=False):
    """
    calculate gradient of model.

    parameters:
    args: hyperparameters
    model: client model or server model
    v: gradient after one derivative
    second_order_grads: whether to calculate second order gradient
    return:
    gradient: second order gradient of model
    """
    criterion = nn.MSELoss().to(args.device)
    x, y = data_batch
    if second_order_grads:
        frz_model_params = copy.deepcopy(model.state_dict())
        delta = 1e-3
        dummy_model_params_1 = OrderedDict()
        dummy_model_params_2 = OrderedDict()
        with torch.no_grad():
            for (layer_name, param), grad in zip(model.named_parameters(), v):
                dummy_model_params_1.update({layer_name: param + delta * grad})
                dummy_model_params_2.update({layer_name: param - delta * grad})

        model.load_state_dict(dummy_model_params_1, strict=False)
        logit_1 = model(x)
        loss_1 = criterion(logit_1, y)
        grads_1 = torch.autograd.grad(loss_1, model.parameters())

        model.load_state_dict(dummy_model_params_2, strict=False)
        logit_2 = model(x)
        loss_2 = criterion(logit_2, y)
        grads_2 = torch.autograd.grad(loss_2, model.parameters())

        model.load_state_dict(frz_model_params)

        grads = []
        with torch.no_grad():
            for g1, g2 in zip(grads_1, grads_2):
                grads.append((g1 - g2) / (2 * delta))
        return grads

    else:
        logit = model(x)
        loss = criterion(logit, y)
        grads = torch.autograd.grad(loss, model.parameters())
        return grads

def train(args, model, client_id):
    """
    train.
    parameters:
    args: hyperparameters
    model: client or server model
    client id: client model name
    return:
    model: client model after training
    """
    model.train()
    Data_train, _ = load_client_data(model.name, args.B)
    model.len = len(Data_train)
    if args.meta_learning:
        """
        meta-learning train.
        return: 
        model: client model after meta-learning training
        """
        Data_train = [x for x in iter(Data_train)]
        # for epoch in loop:
        for epoch in range(args.local_epochs):
            # loop.set_description(f'Client {client_id}----Train----Epoch {epoch + 1}')
            temp_model = copy.deepcopy(model)
            # define meta-function
            data_batch_1 = get_data_batch(args, Data_train)
            grads = compute_grad(args, temp_model, data_batch_1)
            for param, grad in zip(temp_model.parameters(), grads):
                param.data.sub_(args.alpha * grad)
            # differentiate the meta-function
            data_batch_2 = get_data_batch(args, Data_train)
            grads_1st = compute_grad(args, temp_model, data_batch_2)
            data_batch_3 = get_data_batch(args, Data_train)
            grads_2nd = compute_grad(args, model, data_batch_3, v=grads_1st, second_order_grads=True)
            # update local model weight
            for param, grad1, grad2 in zip(model.parameters(), grads_1st, grads_2nd):
                param.data.sub_(args.beta * grad1 - args.beta * args.alpha * grad2)
    else:
        """
        federated learning training.
        return:
        model: client model after federated learning training
        """
        loop = tqdm(range(args.local_epochs), total=args.local_epochs)
        optimizer = torch.optim.Adam(model.parameters())
        loss_function = nn.MSELoss().to(args.device)
        # one step
        loss = 0.0
        accuracy = 0.0
        for epoch in loop:
            loop.set_description(f'Client {client_id}----Train----Epoch {epoch + 1}')
            for seq, label in Data_train:
                seq, label = seq.to(args.device), label.to(args.device)
                output = model(seq)
                loss_flag = loss_function(output, label)
                optimizer.zero_grad()
                loss_flag.backward()
                optimizer.step()
                loss += loss_flag.item()
                pred = torch.tensor([[1] if num[0] >= 0.5 else [0] for num in output]).to(args.device)
                accuracy += pred.eq(label.long()).sum().item()
        loss = loss / len(Data_train) / args.local_epochs
        accuracy = accuracy / (len(Data_train)*args.B) / args.local_epochs
        print(f'Client {client_id}----train----loss: {loss}   accuracy: {accuracy}')
    return model

def test(args, model):
    """
    test.
    parameters:
    args: hyperparameters
    model: client model or server model
    return:
    result: loss and accuracy of the model
    """
    model.eval()
    Data_test = load_test_data('test', args.B)
    if model.name != 'Server':
        Data_train, Data_val = load_client_data(model.name, args.B)
        loss_function = nn.MSELoss().to(args.device)
        loss_train, loss_val = 0.0, 0.0
        accuracy_val, accuracy_test = 0.0, 0.0
        for (seq, label) in Data_train:
            with torch.no_grad():
                seq = seq.to(args.device)
                label = label.to(args.device)
                output = model(seq)
                loss_flag = loss_function(output, label)
                loss_train += loss_flag.item()
        loss_train = loss_train / len(Data_train)
        for (seq, label) in Data_val:
            with torch.no_grad():
                seq = seq.to(args.device)
                label = label.to(args.device)
                output = model(seq)
                loss_flag = loss_function(output, label)
                loss_val += loss_flag.item()
                pred = torch.tensor([[1] if num[0] >= 0.5 else [0] for num in output]).to(args.device)
                accuracy_val += pred.eq(label.long()).sum().item()
        loss_val = loss_val / len(Data_val)
        accuracy_val = accuracy_val / (len(Data_val) * args.B)
        for (seq, label) in Data_test:
            with torch.no_grad():
                seq = seq.to(args.device)
                label = label.to(args.device)
                output = model(seq)
                pred = torch.tensor([[1] if num[0] >= 0.5 else [0] for num in output]).to(args.device)
                accuracy_test += pred.eq(label.long()).sum().item()
        accuracy_test = accuracy_test / (len(Data_test) * args.B)
        return loss_train, loss_val, accuracy_val, accuracy_test
    else:
        loss_function = nn.MSELoss().to(args.device)
        loss = 0.0
        accuracy = 0.0
        for (seq, label) in Data_test:
            with torch.no_grad():
                seq = seq.to(args.device)
                label = label.to(args.device)
                output = model(seq)
                loss_flag = loss_function(output, label)
                loss += loss_flag.item()
                pred = torch.tensor([[1] if num[0] >= 0.5 else [0] for num in output]).to(args.device)
                accuracy += pred.eq(label.long()).sum().item()
        loss = loss / len(Data_test)
        accuracy = accuracy / (len(Data_test) * args.B)
        return loss, accuracy

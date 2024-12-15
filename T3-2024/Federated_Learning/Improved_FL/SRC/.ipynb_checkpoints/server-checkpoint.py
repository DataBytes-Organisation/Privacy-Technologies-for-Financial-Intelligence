# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/18 15:48
@Auth ： Chuang Liu
@Email ：LIUC0316@126.COM
@File ：server.py
@IDE ：PyCharm
"""

import torch
import numpy as np
import random
from client import train, test
from model import MyNet
import copy
from tqdm import tqdm
import time

class PerFed:
    def __init__(self, args):
        self.args = args
        self.nn = MyNet(args=self.args, name='Server').to(args.device)
        self.nns = []
        # if meta-learning is true, model adopts FedAvg
        self.args.index = range(self.args.clients_num)
        # init clients' models
        for i in range(self.args.clients_num):
            temp = MyNet(args=self.args, name=self.args.clients[i]).to(args.device)
            if not self.args.meta_learning:
                temp.load_state_dict(torch.load(f'.\{self.args.parameters_path}\client{i}.pth', weights_only=True))
            self.nns.append(temp)
        if not self.args.meta_learning:
            self.nn.load_state_dict(torch.load(f'.\{self.args.parameters_path}\server.pth',weights_only=True))
            self.args.parameters_path = 'FL_parameters'
            self.args.rounds = 500

    def server(self):
        rounds, loss_train, loss_val, accuracy_val, accuracy_test, loss, accuracy = [], [], [], [], [], [], []
        # loop = tqdm(range(self.args.rounds), total=self.args.rounds)
        # for t in loop:
        for t in range(self.args.rounds):
            # print('round', t + 1, ':')
            # loop.set_description(f'Round [{(t + 1)}/{self.args.rounds}]')
            print("--------------------------------------------------------------------------------------------------------------------------------------")
            if self.args.meta_learning:
                print(f"Meta Learning----round {t+1}----")
                m = np.max([int(self.args.C * self.args.clients_num), 1])
                self.args.index = random.sample(range(0, self.args.clients_num), m)
            else:
                print(f"Federated Learning----round {t+1}----")
            # dispatch parameters
            self.dispatch()
            # local updating
            loss_flag_1, loss_flag_2, accuracy_flag_1, accuracy_flag_2 = self.client_update()
            print(f"Client----"
                  f"loss_train: {loss_flag_1}\t"
                  f"loss_val: {loss_flag_2}\t"
                  f"accuracy_val: {accuracy_flag_1}\t"
                  f"accuracy_test: {accuracy_flag_2}")
            loss_train.append(loss_flag_1)
            loss_val.append(loss_flag_2)
            accuracy_val.append(accuracy_flag_1)
            accuracy_test.append(accuracy_flag_2)
            rounds.append(t+1)
            # aggregation parameters
            loss_flag, accuracy_flag = self.aggregation()
            loss.append(loss_flag)
            accuracy.append(accuracy_flag)
            print(f"Server----"
                  f"loss: {loss_flag}\t\t"
                  f"accuracy: {accuracy_flag}")
        # save models' parameters after meta learning
        torch.save(self.nn.state_dict(), f'.\{self.args.parameters_path}\server.pth')
        for i in range(self.args.clients_num):
            torch.save(self.nns[i].state_dict(), f'.\{self.args.parameters_path}\client{i}.pth')
        return rounds, loss_train, loss_val, accuracy_val, accuracy_test, loss, accuracy


    def aggregation(self):
        nn_weight_sum = 0
        for p in self.nn.parameters():
            p.data.zero_()
        for i in self.args.index:
            # normal
            nn_weight_sum += self.nns[i].len
        for i in self.args.index:
            for x, y in zip(self.nns[i].parameters(), self.nn.parameters()):
                y.data += (x.data * (self.nns[i].len / nn_weight_sum))
        # calculate loss and accuracy of server model
        loss, accuracy = test(self.args, self.nn)
        return loss, accuracy

    def dispatch(self):
        if self.args.meta_learning:
            for i in self.args.index:
                for old_params, new_params in zip(self.nns[i].parameters(), self.nn.parameters()):
                  old_params.data = new_params.data.clone()
        else:
            for i in self.args.index:
                Kb_arg_num = 0
                for old_params, new_params in zip(self.nns[i].parameters(), self.nn.parameters()):
                    old_params.data = new_params.data.clone()
                    Kb_arg_num += 1
                    if Kb_arg_num == 2 * (self.args.total - self.args.Kp):
                        break

    def client_update(self):  # update clients' model parameters
        for i in self.args.index:
            self.nns[i] = train(self.args, self.nns[i], i)
        # calculate loss and accuracy of clients' model
        loss_train, loss_val, accuracy_val, accuracy_test = 0.0, 0.0, 0.0, 0.0
        for i in range(self.args.clients_num):
            loss_flag_1, loss_flag_2, accuracy_flag_1, accuracy_flag_2 = test(self.args,self.nns[i])
            loss_train += loss_flag_1
            loss_val += loss_flag_2
            accuracy_val += accuracy_flag_1
            accuracy_test += accuracy_flag_2
        loss_train /= self.args.clients_num
        loss_val /= self.args.clients_num
        accuracy_val /= self.args.clients_num
        accuracy_test /= self.args.clients_num
        return loss_train, loss_val, accuracy_val, accuracy_test

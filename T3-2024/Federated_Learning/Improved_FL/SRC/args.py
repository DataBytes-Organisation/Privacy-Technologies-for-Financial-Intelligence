# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/18 15:48
@Auth ： Chuang Liu
@Email ：LIUC0316@126.COM
@File ：args.py
@IDE ：PyCharm
"""
import argparse
import torch

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--meta_learning', type=bool, default=True, help='whether to use meta learning')
    parser.add_argument('--parameters_path', type=str, default='meta_parameters', help='path of saved parameters')
    parser.add_argument('--rounds', type=int, default=600, help='number of communication rounds')
    parser.add_argument('--clients_num', type=int, default=10, help='number of total clients')
    parser.add_argument('--local_epochs', type=int, default=10, help='number of local epochs')
    parser.add_argument('--input_dim', type=int, default=12, help='input dimension')
    parser.add_argument('--alpha', type=float, default=0.01, help='learning rate')
    parser.add_argument('--beta', type=float, default=0.001, help='learning rate')
    parser.add_argument('--Kp', type=int, default=2, help='number of personalized layers')
    parser.add_argument('--total', type=int, default=5, help='number of total layers')
    parser.add_argument('--C', type=float, default=0.5, help='sampling rate')
    parser.add_argument('--B', type=int, default=25, help='local batch size')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    clients = ['Client' + str(i) for i in range(10)]
    parser.add_argument('--clients', default=clients)

    args = parser.parse_args()

    return args
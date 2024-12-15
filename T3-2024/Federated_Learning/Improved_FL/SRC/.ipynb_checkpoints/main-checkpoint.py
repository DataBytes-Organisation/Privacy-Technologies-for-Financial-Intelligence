# -*- coding: utf-8 -*-
"""
@Time ： 2024/11/18 15:48
@Auth ： Chuang Liu
@Email ：LIUC0316@126.COM
@File ：main.py
@IDE ：PyCharm
"""

from args import args_parser
from server import PerFed
from visualization import visualization_client_train, visualization_client_accuracy, visualization_server_loss, visualization_server_accuracy
import torch

def main():
    args = args_parser()
    # for bool_value in [True, False]:
    #     torch.cuda.empty_cache()
    #     args.meta_learning = bool_value
    #     FL = PerFed(args)
    #     print(FL.args.meta_learning)
    #     rounds, loss_train, loss_val, accuracy_val, accuracy_test, loss, accuracy = FL.server()
    #     visualization_client_train(rounds, loss_train, loss_val)
    #     visualization_client_accuracy(rounds, accuracy_val, accuracy_test)
    #     visualization_server_loss(rounds, loss)
    #     visualization_server_accuracy(rounds, accuracy)

    torch.cuda.empty_cache()
    args.meta_learning = False
    FL = PerFed(args)
    print(FL.args.meta_learning)
    rounds, loss_train, loss_val, accuracy_val, accuracy_test, loss, accuracy = FL.server()
    visualization_client_train(rounds, loss_train, loss_val)
    visualization_client_accuracy(rounds, accuracy_val, accuracy_test)
    visualization_server_loss(rounds, loss)
    visualization_server_accuracy(rounds, accuracy)

if __name__ == '__main__':
    main()
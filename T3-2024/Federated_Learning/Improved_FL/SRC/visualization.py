# -*- coding: utf-8 -*-
"""
@Time ： 2024/12/7
@Auth ： Chuang Liu
@Email ：LIUC0316@126.COM
@File ：visualization
@IDE ：PyCharm
"""

import matplotlib.pyplot as plt

def visualization_client_train(rounds, loss_train, loss_val):
    plt.plot(rounds, loss_train, "b-", label="loss_train")
    plt.plot(rounds, loss_val, "r-", label="loss_val")

    plt.xlim(0,len(rounds))
    plt.title('Client Model Training Loss Change Diagram')
    plt.xlabel('Round')
    plt.ylabel('Loss')

    plt.legend()
    plt.show()

def visualization_client_accuracy(rounds, accuracy_val, accuracy_test):
    plt.plot(rounds, accuracy_val, "b-", label="accuracy_val")
    plt.plot(rounds, accuracy_test, "r-", label="accuracy_test")

    plt.xlim(0,len(rounds))
    plt.title('Client Model Training Accuracy Change Diagram')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')

    plt.legend()
    plt.show()

def visualization_server_loss(rounds, loss):
    plt.plot(rounds, loss, "b-", label="loss")

    plt.xlim(0,len(rounds))
    plt.title('Server Model Training Loss Change Diagram')
    plt.xlabel('Round')
    plt.ylabel('Loss')

    plt.legend()
    plt.show()

def visualization_server_accuracy(rounds, accuracy):
    plt.plot(rounds, accuracy, "r-", label="accuracy")

    plt.xlim(0,len(rounds))
    plt.title('Server Model Training Accuracy Change Diagram')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')

    plt.legend()
    plt.show()
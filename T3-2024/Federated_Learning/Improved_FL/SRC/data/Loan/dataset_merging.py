# -*- coding: utf-8 -*-
"""
@Time ： 2024/12/6
@Auth ： Chuang Liu
@Email ：LIUC0316@126.COM
@File ：dataset_merging
@IDE ：PyCharm
"""

import pandas as pd

# load all loan data
df = pd.DataFrame()
for i in range(22):
    df_flag = pd.read_csv(f'./Loan_fake_Task2_{i}.csv')
    df = pd.concat([df,df_flag], axis=0, ignore_index=True)

df.info()
df.head()

# scramble data
for i in range(10000):
    df = df.sample(frac=1).reset_index(drop=True)

df.head()

# save 11 loan data files including 10 client data files and 1 server data file
for i in range(10):
    df[i*10000:(i+1)*10000].to_csv(f'Client{i}.csv', index=False)
df[100000:].to_csv('Server.csv', index=False)



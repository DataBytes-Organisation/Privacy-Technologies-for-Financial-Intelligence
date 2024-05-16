import time
import numpy as np
import pandas as pd
from Pyfhel import Pyfhel

HE = Pyfhel()
HE.contextGen(scheme='bfv', n=2**14, t_bits=20)
HE.keyGen()

df0 = pd.read_csv("suspects10-target.csv")

df1 = pd.read_csv("suspects10-ml01.csv")
df2 = pd.read_csv("suspects10-ml02.csv")
df3 = pd.read_csv("suspects10-ml03.csv")
df4 = pd.read_csv("suspects10-ml04.csv")
df5 = pd.read_csv("suspects10-ml05.csv")
df6 = pd.read_csv("suspects10-ml06.csv")
df7 = pd.read_csv("suspects10-ml07.csv")
df8 = pd.read_csv("suspects10-ml08.csv")
df9 = pd.read_csv("suspects10-ml09.csv")
df10 = pd.read_csv("suspects10-ml10.csv")

targets = df0['ID'].to_list()

suspects = [[]]*10
suspects[0] = df1['ID'].to_list()
suspects[1] = df2['ID'].to_list()
suspects[2] = df3['ID'].to_list()
suspects[3] = df4['ID'].to_list()
suspects[4] = df5['ID'].to_list()
suspects[5] = df6['ID'].to_list()
suspects[6] = df7['ID'].to_list()
suspects[7] = df8['ID'].to_list()
suspects[8] = df9['ID'].to_list()
suspects[9] = df10['ID'].to_list()



def match(tar, sus):
	int1 = np.array([tar], dtype=np.int64)
	int2 = np.array([sus], dtype=np.int64)
	
	ctxt1 = HE.encryptInt(int1)
	ctxt2 = HE.encryptInt(int2)
	
	ctxtSub = ctxt1 - ctxt2
	resSub = HE.decryptInt(ctxtSub)
	
	if (resSub[0]) == 0:
		print("Target ID", tar ," has matched to suspect ID: ", HE.decryptInt(ctxt1))

def listmatch(tar, sus):
	for l in range(0, (len(sus))):
		for t in range(0, (len(tar))):
			for s in range(0, (len(sus[l]))):
				#print(tar[t],(sus[l][s]))
				match(tar[t],(sus[l][s]))



tic = time.perf_counter()

listmatch(targets,suspects)

toc = time.perf_counter()
print(f"{toc - tic:0.4f} seconds to process")

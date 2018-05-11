import os.path as path
import os
import collections
import numpy as np
import matplotlib.pyplot as plt
import jieba


numWords = []
with open('angry.txt',"r",encoding='UTF-8') as f:
	line = f.readline()
	while line:
		counter = len(list(jieba.cut(line,cut_all=False)))
		numWords.append(counter)
		line = f.readline()
with open('common.txt',"r",encoding='UTF-8') as f:
	line = f.readline()
	while line:
		counter = len(list(jieba.cut(line,cut_all=False)))
		numWords.append(counter)
		line = f.readline()
numFiles = len(numWords)
print(numFiles)


plt.hist(numWords, 50)
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.axis([0, 60, 0, 10000])
plt.show()
import os.path as path
import os
import pdb
import collections
import jieba
import numpy as np
import random
import tensorflow as tf
from six.moves import xrange
import math
import matplotlib.pyplot as plt


#读词和词向量
wordsList = np.load('word.npy')
wordsList = wordsList.tolist()#原始是numpy

print(len(wordsList))
# wordVectors = np.load('word_vec.npy')

#读训练数据
numWords = []
with open('angry.txt',"r",encoding='UTF-8') as f:
	line = f.readline()
	while line:
		# print(line)
		#pdb.set_trace()
		counter = len(list(jieba.cut(line,cut_all=False)))
		# print(counter)
		# pdb.set_trace()
		numWords.append(counter)
		line = f.readline()
with open('common.txt',"r",encoding='UTF-8') as f:
	line = f.readline()
	while line:
		counter =len(list(jieba.cut(line,cut_all=False)))
		numWords.append(counter)
		line = f.readline()
numFiles = len(numWords)



maxSeqLength= 29 #统计后得出

ids = np.zeros((numFiles, maxSeqLength), dtype='int32')
fileCounter = 0
with open('angry.txt',"r",encoding='UTF-8') as f:
	
	line = f.readline()
	while line:
		indexCounter = 0
		line_cut = list(jieba.cut(line,cut_all=False))
		for word in line_cut:
			try:
				# print(word)
				# pdb.set_trace()
				ids[fileCounter][indexCounter] = wordsList.index(word)
				# print(ids[fileCounter])
				# pdb.set_trace()
			except ValueError:
				if indexCounter + 1 == len(line_cut) :
					ids[fileCounter][indexCounter] = 0
				else :
					ids[fileCounter][indexCounter] = 199999 #Vector for unkown words
			indexCounter = indexCounter + 1
			# print(indexCounter)
			# pdb.set_trace()
			if indexCounter >= maxSeqLength:
				break
		fileCounter = fileCounter + 1 
		line = f.readline()
		
with open('common.txt',"r",encoding='UTF-8') as f:
	
	line = f.readline()
	while line:
		indexCounter = 0
		line_cut = list(jieba.cut(line,cut_all=False))
		for word in line_cut:
			try:
				ids[fileCounter][indexCounter] = wordsList.index(word)
			except ValueError:
				if indexCounter + 1 == len(line_cut) :
					ids[fileCounter][indexCounter] = 0
				else :
					ids[fileCounter][indexCounter] = 199999 #Vector for unkown words
			indexCounter = indexCounter + 1
			if indexCounter >= maxSeqLength:
				fileCounter = fileCounter + 1 
				line = f.readline()
				break
		fileCounter = fileCounter + 1 
		line = f.readline()
print(ids.shape)

np.save('modelMatrix', ids)
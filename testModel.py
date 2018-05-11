import os.path as path
import os
import sys
import pdb
import collections
import jieba
import numpy as np
import random
import tensorflow as tf
from six.moves import xrange
import math
from random import randint
import datetime

batchSize = 24
lstmUnits = 35
numClasses = 2
iterations = 100000
numDimensions = 128
maxSeqLength=29

testString = sys.argv[1]
# testString = '京东什么垃圾客服'
wordsList = np.load('word.npy')
wordsList = wordsList.tolist()#原始是numpy

wordVectors = np.load('word_vec.npy')





def getSentenceMatrix(sentence):
	arr = np.zeros([batchSize, maxSeqLength])
	sentenceMatrix = np.zeros([batchSize,maxSeqLength])
	split_sentence= list(jieba.cut(sentence,cut_all=False))
	# print(split_sentence)
	# pdb.set_trace()
	if len(split_sentence)<29:
		for indexCounter,word in enumerate(split_sentence):
			try:
				sentenceMatrix[0,indexCounter] = wordsList.index(word)
			except ValueError:
				if indexCounter + 1 == len(sentence) :
					sentenceMatrix[0,indexCounter] = 0 
				else:
					sentenceMatrix[0,indexCounter] = 199999 
	return sentenceMatrix



tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])
data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors,input_data)#24,12,128


lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)


correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)



sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('result'))




sentenceMatrix = getSentenceMatrix(testString)

predictedSentiment = sess.run(prediction, {input_data: sentenceMatrix})[0]


if predictedSentiment[0] > predictedSentiment[1] : 
	print('用户生气了')
else:
	print('正常交流')

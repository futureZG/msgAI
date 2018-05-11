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
from random import randint
import datetime

batchSize = 24
lstmUnits = 35
numClasses = 2
iterations = 100000
numDimensions = 128
maxSeqLength=29


wordVectors = np.load('word_vec.npy')
modelMatrix = np.load('modelMatrix.npy')


# print(wordVectors[0])
# print(modelMatrix[0:20])

#得到训练数据及标记
def getTrainBatch():
	labels = []
	arr = np.zeros([batchSize, maxSeqLength])
	for i in range(batchSize):
		if (i % 2 == 0): 
			num = randint(1,30000)
			labels.append([1,0])
		else:
			num = randint(31000,86000)
			labels.append([0,1])
		arr[i] = modelMatrix[num-1:num]
	return arr, labels



tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])
data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors,input_data)#24,12,128


lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.8)
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
sess.run(tf.global_variables_initializer())



for i in range(iterations):
	nextBatch, nextBatchLabels = getTrainBatch();
	sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})
	
save_path = saver.save(sess, "result/msgAI.ckpt", global_step=i)
print("saved to %s" % save_path)

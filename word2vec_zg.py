import os.path as path
import os
import jieba
import pdb
import collections
import numpy as np
import random
import tensorflow as tf
from six.moves import xrange
import math

def read_data():
	raw_word_list = []
	with open('train.txt',"r",encoding='UTF-8') as f:
		line = f.readline()
		while line : 
			# print(line)
			#pdb.set_trace()
			while '\n' in line:
				line = line.replace('\n','')
			while ' ' in line:
					line = line.replace(' ','')
			#去除脏数据
			if '加入咨询' in line or '订单编号' in line :
				line = f.readline() 
			else:
				if len(line)>0: # 如果句子非空
						raw_words = list(jieba.cut(line,cut_all=False))
						raw_word_list.extend(raw_words)
				#print(raw_words)
				#print(raw_word_list)
				line = f.readline()
			#pdb.set_trace()
	return raw_word_list
words = read_data()


# print(words)
#取20W的数据
vocabulary_size = 200000
def build_dataset(words):
	count = [['UNK', -1]]
	count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
	dictionary = dict()
	for word, _ in count:
		dictionary[word] = len(dictionary)
	data = list()
	unk_count = 0
	for word in words:
		if word in dictionary:
		 	index = dictionary[word]
		else:
			index = 0  
			unk_count += 1
		data.append(index)
	count[0][1] = unk_count
	reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)


result=[]
for i in reverse_dictionary:
	result.append(reverse_dictionary[i])
np.save("word",result)

del words  #删除words节省内存
del result

data_index = 0

#生成训练batch数据和标签
def generate_batch(batch_size, num_skips, skip_window):
	global data_index
	assert batch_size % num_skips == 0
	assert num_skips <= 2 * skip_window
	batch = np.ndarray(shape=(batch_size), dtype=np.int32)
	labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
	span = 2 * skip_window + 1  # 窗口
	buffer = collections.deque(maxlen=span)
	for _ in range(span):
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
	for i in range(batch_size // num_skips):
		target = skip_window  
		targets_to_avoid = [skip_window]
		for j in range(num_skips):
			while target in targets_to_avoid:
				target = random.randint(0, span - 1)
			targets_to_avoid.append(target)
			batch[i * num_skips + j] = buffer[skip_window]
			labels[i * num_skips + j, 0] = buffer[target]
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
	return batch, labels

#参数
batch_size = 128
embedding_size = 128  #词向量长度
skip_window = 1       
num_skips = 2          
num_sampled = 64 


graph = tf.Graph()
with graph.as_default():
	    # Input data.
	train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
	train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
	# valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

	# Ops and variables pinned to the CPU because of missing GPU implementation
	with tf.device('/cpu:0'):
		# Look up embeddings for inputs.
		embeddings = tf.Variable(
			tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
		embed = tf.nn.embedding_lookup(embeddings, train_inputs)

		# Construct the variables for the NCE loss
		nce_weights = tf.Variable(
			tf.truncated_normal([vocabulary_size, embedding_size],
		                        stddev=1.0 / math.sqrt(embedding_size)))
		nce_biases = tf.Variable(tf.zeros([vocabulary_size]),dtype=tf.float32)

	#NCE损失函数
	loss = tf.reduce_mean(
			tf.nn.nce_loss(weights=nce_weights,biases=nce_biases, inputs=embed, labels=train_labels,
	             num_sampled=num_sampled, num_classes=vocabulary_size))

	#梯度下降优化
	optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

	
	norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
	normalized_embeddings = embeddings / norm

	#初始化参数
	init = tf.global_variables_initializer()

#下面是训练过程
num_steps = 2000000
with tf.Session(graph=graph) as session:
	
	init.run()

	average_loss = 0
	for step in xrange(num_steps):
		batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
		feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

		_, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
		average_loss += loss_val

		if step % 2000 == 0:
			if step > 0:
				average_loss /= 2000
			print("Average loss at step ", step, ": ", average_loss)
			average_loss = 0

	final_embeddings = normalized_embeddings.eval()
	#保存向量集
	np.save("word_vec",final_embeddings)
import os.path as path
import os
import collections
import numpy as np


emotion_angry = []
emothon_common = []
with open('LSTM_data.txt',"r",encoding='UTF-8') as f:
	line = f.readline()
	while line:
		if '加入咨询' in line or '订单编号' in line :
			line = f.readline() 

		elif '狗屁' in line or '垃圾' in line or '退货' in line or '慢' in line or '妈逼' in line or '换货' in line or '什么意思' in line  or '快点' in line or '扯犊子' in line or '领导' in line or '投诉' in line or '特么' in line or '请回复' in line or '去你大爷的' in line  or '差评' in line or '搞笑' in line or  '不行' in line or '不处理' in line or '什么玩意' in line or '!' in line or '烂透了' in line or '不说话' in line or '急' in line or '争论' in line or '伤不起' in line or '不负责' in line or '生气' in line or '死' in line or '装傻' in line or '服了' in line or '他妈' in line or '杂种' in line or '坑' in line or '京东' in line or '回话' in line or '忽悠' in line or '郁闷' in line or  '卧槽' in line or '尼玛' in line or '容忍' in line or '搞毛线' in line or '变质' in line or '客服' in line  or '无语' in line or '服务' in line or '骂' in line or '气人' in line :
			emotion_angry.append(line)
			line = f.readline() 
		else :
			emothon_common.append(line)
			line = f.readline() 

f_angry = open('angry_origin.txt','w',encoding='utf-8')
f_common = open('common_origin.txt','w',encoding='utf-8')
# for i in range(len(emotion_angry)):
# 	f_angry.write(f_angry[i]+'\n')
# for j in range(len(emothon_common)):
# 	f_common.write(emothon_common[j]+'\n')
for word1 in emotion_angry:
	f_angry.write(word1)
for word2 in emothon_common:
	f_common.write(word2)
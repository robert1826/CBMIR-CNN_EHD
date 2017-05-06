from __future__ import print_function
import numpy as np
from six.moves import cPickle as pickle
from six.moves import range
from scipy import misc
import numpy as np
import random
import sys
import heapq as hq
import shutil
import os
import re

def load_descriptor(name):
	with open(name, 'rb') as f:
		save = pickle.load(f)
		desc = save['desc7']
		labels = save['labels']
		names = save['name']
		del save

	return desc, labels, names

def load_retrieval_result():
	with open('retrieval_result_fc7', 'rb') as f:
		save = pickle.load(f)
		ResultQ = save['query']
		ResultR = save['result']
		indexQ = save['query_index']
		indexR = save['result_index'] 
	del ResultR[0]
	del indexR[0]
	return ResultQ, ResultR, indexQ, indexR

def get_file_name(s):
	 num = re.search('(\d+)(?=(\.png|\.jpg))', s).group(1)
	 return num + '.png'

if __name__ == '__main__':
	desc, labels, names = load_descriptor('train_dataset.txt_desc')
	t_desc, t_labels,t_names = load_descriptor('test_dataset.txt_desc')
	ResultQ, ResultR, indexQ, indexR = load_retrieval_result()

	for i in indexQ:
		relevance_score = {}
		for j in indexR[i]:
			relevance_score[ labels[j] ] = 0
		
		for current_label in relevance_score.keys():
			current_score = 0
			for j in list(reversed(indexR[i])):
				if labels[j] == current_label:
					current_score += 1 - list(reversed(indexR[i])).index(j) // 20
			relevance_score[current_label] = current_score

		alpha = sum(1 - i // 20 for i in range(10))
		relevance_score = list(zip(list(relevance_score.keys()), list(relevance_score.values())))
		relevance_score = sorted(relevance_score, key=lambda x : -x[1])

		for j in range(len(relevance_score)):
			if sum( [ relevance_score[k][1] for k in range(j) ] ) >= alpha:
				relevance_score = relevance_score[:j]
				break
			
		# print(t_labels[i], relevance_score)

		# now use the classes that was obtained by the 'relevance score' & EHD to refine the result
		useful_classes = map(lambda x : x[0], relevance_score)
		useful_classes = list(useful_classes)
		useful_classes = set(useful_classes)
		
		useful_images = [j for j in indexR[i] if labels[j] in useful_classes]

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

def load_retrieval_result(fileName):
	with open(fileName, 'rb') as f:
		save = pickle.load(f)
		ResultQ = save['query']
		ResultR = save['result']
		indexQ = save['query_index']
		indexR = save['result_index'] 
		acc = save['acc']
	# del ResultR[0]
	# del indexR[0]
	return ResultQ, ResultR, indexQ, indexR, acc

def get_basename(s):
	 num = re.search('(\d+)(?=(\.png|\.jpg))', s).group(1)
	 return num

def get_file_name(s):
	num = re.search('(\d+)(?=(\.png|\.jpg))', s).group(1)
	return num + '.png'

def distance(a,b):
	dsum = 0.
	# print(len(a),len(b))
	for i in range(len(a)):
		dsum += (a[i]-b[i])**2
	return dsum

def read_EHD():
	ehd = {}
	with open(sys.argv[1], 'r') as f:
		for curLine in f.readlines():
			filename = curLine.strip().split(' ')[0]
			filename = get_file_name(filename)
			filename = get_basename(filename)
			ehd[filename] = list(map(int, curLine.split(' ')[1:]))
	return ehd


if __name__ == '__main__':
	if len(sys.argv) < 3:
		print('error : please input (EHD file, prev retrieval result)')
		exit(1)

	ehd = read_EHD()

	# name : 1000.png, result : ../IRMA/../../../1000.png
	desc, labels, names = load_descriptor('train_dataset.txt_desc')
	t_desc, t_labels,t_names = load_descriptor('test_dataset.txt_desc')
	ResultQ, ResultR, indexQ, indexR, acc = load_retrieval_result(sys.argv[2])
	print('[Phase 1 Mean Acc.]', acc, '\n')


	# filter test images
	good = [i for i in range(len(t_labels)) if int(t_labels[i]) < 45][:100]
	t_desc = [t_desc[i] for i in good]
	t_labels = [t_labels[i] for i in good]
	t_names = [t_names[i] for i in good]


	# all_retrievals[i] = list of sorted retrievals for test img i
	all_retrievals = {}

	top_n = 5
	for i in indexQ:
		relevance_score = {}
		indexR[i] = indexR[i][-25:]
		for j in indexR[i]:
			relevance_score[ labels[j] ] = 0
		
		for current_label in relevance_score.keys():
			current_score = 0
			for j in list(reversed(indexR[i])):
				if labels[j] == current_label:
					current_score += 1 - list(reversed(indexR[i])).index(j) // top_n
			relevance_score[current_label] = current_score

		alpha = sum(1 - i // top_n for i in range(10))
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
		# print(useful_images)
		
		ehd_retrieval = sorted(useful_images, key=lambda x: distance(ehd[get_basename(names[x])], ehd[get_basename(t_names[i])]))
		# print(ehd_retrieval)

		all_retrievals[i] = ehd_retrieval[:top_n]
	
	# evaluation using accuracy
	eval_res = []
	for i in all_retrievals.keys():
		correct = sum([ 1 for j in all_retrievals[i] if labels[j] == t_labels[i] ])
		eval_res += [correct / top_n]

	print('[Mean Accuracy]', sum(eval_res) / len(eval_res))
	print('[Min. Accuracy]', min(eval_res))
	print('[Max. Accuracy]', max(eval_res))

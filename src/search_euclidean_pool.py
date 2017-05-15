from __future__ import print_function
import numpy as np
from six.moves import cPickle as pickle
from six.moves import range
import numpy as np
import random
import sys
import heapq as hq
import shutil
import os
from multiprocessing import Pool
import scipy.spatial.distance as dis

#Load the data descriptors
labels_index = {}
labels_sum = {}

def load_data(name):
	with open(name, 'rb') as f:
		save = pickle.load(f)
		desc = save['desc7']
		labels = save['labels']
		names = save['name']
		print(len(desc))
		print(len(labels))
		del save
		print('loaded')

	return desc, labels, names

def distance(a,b):
	# return dis.braycurtis(a, b)  # 61.2
	# return dis.cosine(a, b) # 61
	return dis.euclidean(a, b) # 60.9
	# return dis.correlation(a, b) # 60.7
	# return dis.cityblock(a, b) # 60.4
	# return dis.canberra(a, b) # 59.5
	# return dis.chebyshev(a, b) # 52.3
	# return dis.chebyshev(a, b) # 52.3

def cbir(cbir_args):
	# args desc, labels, names, t_desc, t_labels, t_names, top_n, test_num, src_root, src_root2
	desc = cbir_args[0]
	labels = cbir_args[1]
	names = cbir_args[2]
	t_desc = cbir_args[3]
	t_labels = cbir_args[4]
	t_names = cbir_args[5]
	top_n = cbir_args[6]
	test_num = cbir_args[7]
	src_root = cbir_args[8]
	src_root2 = cbir_args[9]

	test_img = t_desc[test_num]
	retrievals = []
	temp_Retrieval = []

	for i in range(len(desc)):
		dist = distance(desc[i], test_img)
		hq.heappush(temp_Retrieval, (-1*dist, i))
		if len(temp_Retrieval) > top_n:
			hq.heappop(temp_Retrieval)
	for x in range(top_n):
		retrievals.append(hq.heappop(temp_Retrieval))
	correct = sum([1 for j in [labels[u[1]] for u in retrievals] if j == t_labels[test_num]])
	accuracy = correct / top_n

	print('(', test_num, '/', len(t_desc), ')', "==>", accuracy )
	
	src = src_root2 + t_names[test_num]

	tempR=[]
	tmpindex=[]
	for j in range(len(retrievals)):
		src = src_root + names[retrievals[j][1]]
		tempR.append(src)
		tmpindex.append(retrievals[j][1])
	
	# ResultQ[test_num] = src
	# indexQ[test_num] = test_num
	# ResultR[test_num] = tempR
	# indexR[test_num] = tmpindex
	return (src, test_num, tempR, tmpindex, accuracy)

if __name__ == "__main__":
	if len(sys.argv) < 5:
		print('error : no input files (train desc file, test desc file, train dataset root, test dataset root)')
		exit(1)
	
	# load data
	desc, labels, names = load_data(sys.argv[1])
	t_desc, t_labels,t_names = load_data(sys.argv[2])

	# initialize variables
	top_n = 100
	ResultQ=[None] * len(t_desc)
	indexQ=[None] * len(t_desc)
	ResultR=[[] for _ in range(len(t_desc))]
	indexR=[[] for _ in range(len(t_desc))]

	src_root = sys.argv[3]
	src_root2 = sys.argv[4]

	eval_res = [None] * len(t_desc)

	with Pool() as pool:
		# args desc, labels, names, t_desc, t_labels, t_names, top_n, test_num, src_root, src_root2
		cbir_res = pool.map(cbir, [ (desc, labels, names, t_desc, t_labels, t_names, top_n, i, src_root, src_root2) for i in range(len(t_desc)) ])
		for i in range(len(cbir_res)):
			# return (src, test_num, tempR, tmpindex, accuracy)
			ResultQ[i] = cbir_res[i][0]
			indexQ[i] = cbir_res[i][1]
			ResultR[i] = cbir_res[i][2]
			indexR[i] = cbir_res[i][3]
			eval_res[i] = cbir_res[i][4]

	print('[Mean Accuracy]', sum(eval_res) / len(eval_res))
	f = open("retrieval_result_fc7", 'wb')
	save = {
	    'query': ResultQ,
	   	'result': ResultR, 
	   	'query_index': indexQ,
	   	'result_index': indexR,
		'acc' : sum(eval_res) / len(eval_res)
	}
	pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
	f.close()
	print("saved file")
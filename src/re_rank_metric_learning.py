from __future__ import print_function
import numpy as np
from six.moves import cPickle as pickle
from six.moves import range
from scipy import misc
import random
import sys
import heapq as hq
import re
import scipy.spatial.distance as dis
from scipy import linalg
from multiprocessing import Pool
from metric_learn import LMNN

def load_descriptor(name):
	with open(name, 'rb') as f:
		save = pickle.load(f)
		desc = save['desc']
		labels = save['labels']
		names = save['name']
		del save

	return desc, labels, names

def distance(a, b, M_inv):
	return dis.mahalanobis(a, b, M_inv)

def get_basename(s):
	 num = re.search('(\d+)(?=(\.png|\.jpg))', s).group(1)
	 return num

def get_file_name(s):
	num = re.search('(\d+)(?=(\.png|\.jpg))', s).group(1)
	return num + '.png'

def read_EHD():
	ehd = {}
	with open('../data_files/ehd_out.txt', 'r') as f:
		for curLine in f.readlines():
			filename = curLine.strip().split(' ')[0]
			filename = get_file_name(filename)
			filename = get_basename(filename)
			ehd[filename] = list(map(int, curLine.split(' ')[1:]))
	return ehd


def one_retrieval(my_args):
	ehd, test_name, names, top_n, M_inv = my_args

	# cur_retrieval = sorted(range(len(desc)), key=lambda x: distance(t_desc[test_ind], desc[x], M))
	cur_retrieval = []
	for name in names:
		dist = distance(ehd[get_basename(test_name)], ehd[get_basename(name)], M_inv)
		hq.heappush(cur_retrieval, (-1 * dist, name))
		if len(cur_retrieval) > top_n:
			hq.heappop(cur_retrieval)

	print('Test img #', test_name, 'done')
	# return (test_ind, cur_retrieval[:top_n])
	return (test_name, [u[1] for u in cur_retrieval])

if __name__ == '__main__':
	# name : 1000.png, result : ../IRMA/../../../1000.png
	desc, labels, names = load_descriptor('train_dataset.txt_desc')
	t_desc, t_labels,t_names = load_descriptor('test_dataset.txt_desc')

	ehd = read_EHD()
	X = [ ehd[get_basename(u)] for u in names ] 
	X = np.array(X)
	
	lmnn = LMNN(max_iter=1)
	print('Begin Metric Learning...')
	lmnn.fit(X, labels)
	print('Finished Metric Learning')
	M = lmnn.metric()
	M_inv = linalg.inv(M)

	### multi-process retrieval
	# all_retrievals[i] = list of sorted retrievals for test img i
	all_retrievals = {}
	top_n = 5
	
	with Pool() as pool:
		ret_res = pool.map(one_retrieval, [ (ehd, t_names[i], names, top_n, M_inv) for i in range(len(t_desc)) ])
		for one_res in ret_res:
			all_retrievals[one_res[0]] = one_res[1]

	# ### evaluation using accuracy
	eval_res = []
	for i in all_retrievals.keys():
		correct = sum([ 1 for j in all_retrievals[i] if labels[names.index(j)] == t_labels[t_names.index(i)] ])
		eval_res += [correct / top_n]

	print('[Mean Accuracy]', sum(eval_res) / len(eval_res))
	print('[Min. Accuracy]', min(eval_res))
	print('[Max. Accuracy]', max(eval_res))
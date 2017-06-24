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

def one_retrieval(my_args):
	t_desc = my_args[0]
	test_ind = my_args[1]
	desc = my_args[2]
	top_n = my_args[3]
	M_inv = my_args[4]

	# cur_retrieval = sorted(range(len(desc)), key=lambda x: distance(t_desc[test_ind], desc[x], M))
	cur_retrieval = []
	for i in range(len(desc)):
		dist = distance(t_desc[test_ind], desc[i], M_inv)
		hq.heappush(cur_retrieval, (-1 * dist, i))
		if len(cur_retrieval) > top_n:
			hq.heappop(cur_retrieval)

	print('Test img #', test_ind, 'done')
	# return (test_ind, cur_retrieval[:top_n])
	return (test_ind, [u[1] for u in cur_retrieval])

if __name__ == '__main__':
	# name : 1000.png, result : ../IRMA/../../../1000.png
	desc, labels, names = load_descriptor('train_dataset.txt_desc')
	t_desc, t_labels,t_names = load_descriptor('test_dataset.txt_desc')

	print('One desc len', len(desc[0]))

	# chosen = [i for i in range(len(labels)) if int(labels[i]) in range(13, 17)]
	chosen = list(range(len(desc)))
	desc = [desc[i] for i in chosen]
	labels = [labels[i] for i in chosen]
	print('Using', len(desc), 'Images For training')


	lmnn = LMNN(max_iter=1)
	print('Begin Metric Learning...')
	lmnn.fit(np.array(desc), labels)
	print('Finished Metric Learning')
	M = lmnn.metric()
	M_inv = linalg.inv(M)

	### multi-process retrieval
	# all_retrievals[i] = list of sorted retrievals for test img i
	all_retrievals = {}
	top_n = 5
	
	with Pool() as pool:
		ret_res = pool.map(one_retrieval, [ (t_desc, i, desc, top_n, M_inv) for i in range(len(t_desc)) ])
		for one_res in ret_res:
			all_retrievals[one_res[0]] = one_res[1]

	### evaluation using accuracy
	eval_res = []
	for i in all_retrievals.keys():
		correct = sum([ 1 for j in all_retrievals[i] if labels[j] == t_labels[i] ])
		eval_res += [correct / top_n]

	print('[Mean Accuracy]', sum(eval_res) / len(eval_res))
	print('[Min. Accuracy]', min(eval_res))
	print('[Max. Accuracy]', max(eval_res))
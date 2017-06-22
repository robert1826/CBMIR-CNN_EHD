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
from metric_learn import LMNN

def load_descriptor(name):
	with open(name, 'rb') as f:
		save = pickle.load(f)
		desc = save['desc']
		labels = save['labels']
		names = save['name']
		del save

	return desc, labels, names

def distance(a, b, M):
	return dis.mahalanobis(a, b, linalg.inv(M))

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

	# all_retrievals[i] = list of sorted retrievals for test img i
	all_retrievals = {}
	top_n = 5
	for t in range(len(t_labels)):
		if t_labels[t] not in labels:
			continue
		 
		cur_retrieval = sorted(range(len(desc)), key=lambda x: distance(t_desc[0], desc[x], M))
		all_retrievals[t] = cur_retrieval[:top_n]

		if t % 10 == 0:
			print('Test Image #', t)

	# evaluation using accuracy
	eval_res = []
	for i in all_retrievals.keys():
		correct = sum([ 1 for j in all_retrievals[i] if labels[j] == t_labels[i] ])
		eval_res += [correct / top_n]

	print('[Mean Accuracy]', sum(eval_res) / len(eval_res))
	print('[Min. Accuracy]', min(eval_res))
	print('[Max. Accuracy]', max(eval_res))
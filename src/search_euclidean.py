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
	dsum = 0.
	# print(len(a),len(b))
	for i in range(len(a)):
		dsum += (a[i]-b[i])**2
	return dsum

if __name__ == "__main__":
	if len(sys.argv) < 5:
		print('error : no input files (train desc file, test desc file, train dataset root, test dataset root)')
		exit(1)

	desc, labels, names = load_data(sys.argv[1])
	t_desc, t_labels,t_names = load_data(sys.argv[2])

	eval_res = []
	top_n = 100
	### for visualization
	dst_root = 'Retrieval'
	shutil.rmtree(dst_root, True)
	os.makedirs(dst_root)
	src_root = sys.argv[3]
	src_root2 = sys.argv[4]

	test_num = -1
	
	ResultQ=[]
	indexQ=[]
	ResultR=[[]]
	indexR=[[]]
	# for t in range(500):
	for t in range(len(t_desc)):
		test_img = t_desc[t]
		test_num += 1
		retrievals = []
		temp_Retrieval = []
	
		for i in range(len(desc)):
			dist = distance(desc[i], test_img)
			hq.heappush(temp_Retrieval, (-1*dist, i))
			if len(temp_Retrieval) > top_n:
				hq.heappop(temp_Retrieval)
		for x in range(top_n):
			retrievals.append(hq.heappop(temp_Retrieval))
		correct = sum([1 for j in [labels[u[1]] for u in retrievals] if j == t_labels[t]])
		eval_res += [correct / top_n]

		# print('[query label] ', t_names[t],' ',t_labels[t])
		# # print('[sims]', retrievals)
		# print('[names]', [names[u[1]] for u in retrievals])
		# print('[labels]', correct, [labels[i[1]] for i in retrievals])
		# print()
		print('(', t, '/ 500)', "==>", correct / top_n, "==>", sum(eval_res) / len(eval_res))
		
		dst_cur = dst_root + '/test_' + str(test_num) + '_c' + str(correct)
		os.makedirs(dst_cur)

		src = src_root2 + t_names[t]
		dst = dst_cur + '/query_' + t_labels[t]
		shutil.copyfile(src, dst)
		
		ResultQ.append(src)
		indexQ.append(t)

		tempR=[]
		tmpindex=[]
		for j in range(len(retrievals)):
			src = src_root + names[retrievals[j][1]]

			dst = dst_cur + '/' + str(j) + '_' + labels[retrievals[j][1]] + '.png'
			shutil.copyfile(src, dst)
			tempR.append(src)
			tmpindex.append(retrievals[j][1])
		
		ResultR.append(tempR)
		indexR.append(tmpindex)

		# print(ResultQ)
		# print(ResultR)
		# print(indexQ)
		# print(indexR)
		if(t!=0 and t%25 == 0):
			f = open("retrieval_result_fc7", 'wb')
			save = {
			    'query': ResultQ,
			   	'result': ResultR, 
			   	'query_index': indexQ,
			   	'result_index': indexR
			}
			pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
			f.close()
			print("saved file")

	print('[Mean Accuracy]', sum(eval_res) / len(eval_res))
	f = open("retrieval_result_fc7", 'wb')
	save = {
	    'query': ResultQ,
	   	'result': ResultR, 
	   	'query_index': indexQ,
	   	'result_index': indexR
	}
	pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
	f.close()
	print("saved file")
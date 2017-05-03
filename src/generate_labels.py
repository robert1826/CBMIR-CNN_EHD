import numpy as np
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import sys

if __name__ == '__main__':
	
	if len(sys.argv) < 2:
		print('error : no input file (codes csv file), exiting')
		exit(1)
	
	file_name = sys.argv[1]
	in_file = open(file_name, 'r')

	labels = [[]] #image_id;irma_code;05_class;05_irma_code_submask;06_class;06_irma_code;07_irma_code;08_irma_code
	for str in in_file.read().split('\n'):
		labels.append(str.split(';'))
	del labels[1]
	del labels[0]
	in_file.close()

	d = {}
	for i in labels:
		d[i[0]] = i[2]

	# for viewing the dictionary
	# print([ (i, d[i]) for i in list(d.keys())[1:10] ])

	out_file = open(file_name + '_labels.pickle', 'wb')
	save = {
		'labels': d,
	}
	pickle.dump(save, out_file, pickle.HIGHEST_PROTOCOL)
	out_file.close()
	
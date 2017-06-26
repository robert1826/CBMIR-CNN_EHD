from six.moves import cPickle as pickle
from sklearn.neural_network import MLPClassifier
from multiprocessing import Pool
import sys
from sklearn.externals import joblib
from random import shuffle


def load_descriptor(name):
	with open(name, 'rb') as f:
		save = pickle.load(f)
		desc = save['desc']
		labels = save['labels']
		names = save['name']
		del save

	return desc, labels, names

def train_MLPClassifier(desc, labels):
	# pick 10 samples from each class
	chosen = []
	for label in set(labels):
		chosen += [i for i in range(len(labels)) if labels[i] == label][:10]
	shuffle(chosen)
	training_desc = [desc[i] for i in chosen]
	training_labels = [labels[i] for i in chosen]
	
	X, Y = [], []
	for i in range(len(training_desc)):
		for j in range(i + 1, len(training_desc)):
			X += [training_desc[i] + training_desc[j]]
			X += [training_desc[j] + training_desc[i]]

			Y += [training_labels[i] == training_labels[j]] * 2
	print('#training pairs', len(Y))
	
	classifier = MLPClassifier(verbose=True)
	classifier.fit(X, Y)
	return classifier

def retrieve(myargs):
	t_desc, t, desc, classifier = myargs

	potentials = []
	for i in range(len(desc)):
		prediction = classifier.predict([t_desc[t] + desc[i]])
		if prediction[0]:
			potentials += [i]
	print('Test Image #{} Done'.format(t))
	return potentials

def test_retrieval(t_desc, t_labels, desc, labels, classifier):
	eval_res = []
	top_n = 5
	with Pool() as pool:
		print('Begin Multiprocess Retrieval')
		all_retrievals = pool.map(retrieve, [(t_desc, t, desc, classifier) for t in range(len(t_desc))])

		print('Begin Evaluation')
		for t in range(len(all_retrievals)):
			retrievals = all_retrievals[t]

			# if len(retrievals) == 0:
			# 	print('#', t, 'PASS')
			# 	eval_res += [0]
			# 	continue
			correct = sum([1 for ret in retrievals if labels[ret] == t_labels[t]])
			print('#', t, 'precision', correct / max(len(retrievals), 0.1), 'recall', correct / sum([1 for u in range(len(labels)) if labels[u] == t_labels[t]]))

			eval_res += [min(correct, top_n) / top_n]
	print('Top-{} Accuracy is {}'.format(top_n, sum(eval_res) / len(eval_res)))

def test_classifier_acc(t_desc, t_labels, desc, labels, classifier):
	with Pool() as pool:
		measures = pool.map(__test_classifier_acc, [ (t, t_desc, t_labels, desc, labels, classifier) for t in range(len(t_desc))])
		
		tp = sum([u[0] for u in measures]) / len(measures)
		fp = sum([u[1] for u in measures]) / len(measures)
		tn = sum([u[2] for u in measures]) / len(measures)
		fn = sum([u[3] for u in measures]) / len(measures)
		
		print()
		acc = (tp + tn) / (tp + tn + fp + fn)
		precision = tp / (tp + fp)
		recall = tp / (tp + fn)
		f1 = 2 * precision * recall / (precision + recall)
		f2 = 5 * precision * recall / (4 * precision + recall)
		print('Classifier Accuracy', acc)
		print('Classifier Precision',precision)
		print('Classifier Recall', recall)
		print('Classifier F1 Score', f1)
		print('Classifier F2 Score', f2)

def __test_classifier_acc(myargs):
	t, t_desc, t_labels, desc, labels, classifier = myargs

	tp, fp, tn, fn = 0, 0, 0, 0

	for i in range(len(desc)):
		prediction = classifier.predict([t_desc[t] + desc[i]])
		true_prediction = t_labels[t] == labels[i]
		
		if prediction:
			if true_prediction:
				tp += 1
			else:
				fp += 1
		else:
			if not true_prediction:
				tn += 1
			else:
				fn += 1

	print('Test Image #{} done'.format(t))
	return (tp, fp, tn, fn)

if __name__ == '__main__':
	# load data
	desc, labels, names = load_descriptor('train_dataset.txt_desc')
	t_desc, t_labels,t_names = load_descriptor('test_dataset.txt_desc')
	
	if len(sys.argv) < 2:
		print('Wrong args, Please enter train or eval')

	elif sys.argv[1] == 'train':
		classifier = train_MLPClassifier(desc, labels)
		joblib.dump(classifier, 'clf.pkl') 

	elif sys.argv[1] == 'eval':
		classifier = joblib.load('clf.pkl')
		args = (t_desc, t_labels, desc, labels, classifier)
		# test_retrieval(*args)
		test_classifier_acc(*args)

	else:
		print('Wrong args, Please enter train or eval')



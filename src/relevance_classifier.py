from six.moves import cPickle as pickle
from sklearn.neural_network import MLPClassifier

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
	
	training_desc = [desc[i] for i in chosen]
	training_labels = [labels[i] for i in chosen]
	
	X, Y = [], []
	for i in range(len(training_desc)):
		for j in range(i + 1, len(training_desc)):
			X += [training_desc[i] + training_desc[j]]
			Y += [training_labels[i] == training_labels[j]]
	print('#training pairs', len(Y))
	
	classifier = MLPClassifier(alpha=1)
	classifier.fit(X, Y)
	return classifier

def retrieve(t_desc, t, desc, classifier):
	potentials = []
	for i in range(len(desc)):
		prediction = classifier.predict([t_desc[t] + desc[i]])
		if prediction[0]:
			potentials += [i]
	return potentials

if __name__ == '__main__':
	# Data loading and preprocessing
	desc, labels, names = load_descriptor('train_dataset.txt_desc')
	t_desc, t_labels,t_names = load_descriptor('test_dataset.txt_desc')

	classifier = train_MLPClassifier(desc, labels)

	eval_res = []
	top_n = 5
	for t in range(len(t_desc)):
		retrievals = retrieve(t_desc, t, desc, classifier)
		if len(retrievals) == 0:
			print('#', t, 'PASS')
			continue
		correct = sum([1 for ret in retrievals if labels[ret] == t_labels[t]])
		print('#', t, correct / len(retrievals))

		eval_res += [min(correct, top_n) / top_n]
	print('Top-{} Accuracy is {}'.format(top_n, sum(eval_res) / len(eval_res)))
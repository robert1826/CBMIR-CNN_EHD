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

# Data loading and preprocessing
desc, labels, names = load_descriptor('train_dataset.txt_desc')
t_desc, t_labels,t_names = load_descriptor('test_dataset.txt_desc')

chosen = []
for label in set(labels):
	chosen += [i for i in range(len(labels)) if labels[i] == label][:10]

desc = [desc[i] for i in chosen]
labels = [labels[i] for i in chosen]

X, Y = [], []

for i in range(len(desc)):
	for j in range(i + 1, len(desc)):
		X += [desc[i] + desc[j]]
		Y += [labels[i] == labels[j]]
print('#training pairs', len(Y))

classifier = MLPClassifier(alpha=1)
classifier.fit(X, Y)


correct = 0
tot = 0
for i in range(len(t_labels)):
	for j in range(i + 1, len(t_labels)):
		prediction = classifier.predict([t_desc[i] + t_desc[j]])
		true_prediction = t_labels[i] == t_labels[j]
		
		if prediction[0] == true_prediction:
			correct += 1
		tot += 1	
print('Accuracy', correct / tot)
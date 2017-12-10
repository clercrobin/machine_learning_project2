import numpy as np
import scipy
from helpers import *

def is_number(s):
	c = sum(c.isdigit() for c in s)
	return s[0].isdigit() or (c >= len(s) / 2)

def preprocess(s):
	tokens = s.strip().split()

	# Recreate smileys
	for i, t in enumerate(tokens):
		# Replace numbers by <number>
		if is_number(t):
			tokens[i] = '<number>'
		# Recreate smileys
		if t in [':', ';'] and i < len(tokens) - 1 and tokens[i+1] in [')', '(', 'p', 'd', 's', 'o']:
			tokens[i:i+2] = [t + tokens[i+1]]
		if t == '-' and i < len(tokens) - 2 and tokens[i+1] == '_' and tokens[i+2] in ['_', '__', '___']:
			tokens[i:i+3] = ['-_-']

	return ' '.join(tokens)

def hashtags(dataset):
	d = {}
	for s in dataset:
		for t in s.strip().split():
			if t[0] == '#':
				if not t in d:
					d[t] = 1
				else:
					d[t] += 1
	print(sorted(d.items(), key=lambda x: x[1]))

def preprocess_dataset(destination_folder):
	def write_file(filename, s):
		f = open(destination_folder + filename, 'w')
		f.write(s)
		f.close()

	# Small dataset
	train, y = load_dataset(False)
	train = [preprocess(s) for s in train]
	train_pos = '\n'.join(train[:100000])
	train_neg = '\n'.join(train[100000:])
	write_file('train_pos.txt', train_pos)
	write_file('train_neg.txt', train_neg)

	# Full dataset
	train, y = load_dataset(True)
	train = [preprocess(s) for s in train]
	train_pos = '\n'.join(train[:1250000])
	train_neg = '\n'.join(train[1250000:])
	write_file('train_pos_full.txt', train_pos)
	write_file('train_neg_full.txt', train_neg)

	ids, test = load_csv_data('dataset/test_data.txt')
	test = '\n'.join([str(id) + ',' + preprocess(s) for (id, s) in zip(ids, test)])
	write_file('test_data.txt', test)

if __name__ == '__main__':
	preprocess_dataset('preprocessed_dataset/')

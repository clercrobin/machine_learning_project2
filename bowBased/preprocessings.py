import re
from helpers import *

eyes = r'[8:=;]\s*?'
eyes_with_x = r'([8:=;]|\bx)\s*?'
noses = r"(['`\-]\s*?)?"

smile = re.compile(r'{}{}([)]+|d+\b)|\^\s*?\^'.format(eyes_with_x, noses))
lol = re.compile(r'{}{}p+\b'.format(eyes_with_x, noses))
sad = re.compile(r'{}{}[(]+|-\s*?(_\s*?)+-'.format(eyes_with_x, noses))
neutral = re.compile(r'{}{}([\\\/*]+|l+\b)'.format(eyes_with_x, noses))
surprised = re.compile(r'{}{}o+\b|\bo\s*?(_\s*?)+o\b'.format(eyes, noses))
heart = re.compile(r'<\s*?3')
number = re.compile(r'\b[-+]?[.\d]*[\d]+[:,.\d]*\b')
hashtag = re.compile(r'(#\S+)')
repeat = re.compile(r'(([!?.])\s*?){2,}')
elong = re.compile(r'\b(\S*?)(?P<l>.)(?P=l){2,}\b')

def preprocess(s):
	""" Apply the preprocessings on a string. """
	s = smile.sub('<smile>', s)
	s = lol.sub('<lolface>', s)
	s = sad.sub('<sadface>', s)
	s = neutral.sub('<neutralface>', s)
	s = surprised.sub('<surprisedface>', s)
	s = heart.sub('<heart>', s)
	s = number.sub('<number>', s)
	s = hashtag.sub('<hashtag> \g<1>', s)
	s = repeat.sub('\g<1> <repeat>', s)
	s = elong.sub('\g<1>\g<2> <elong>', s)
	return s

def preprocess_dataset(destination_folder):
	""" Preprocess the dataset and write the output in destination_folder. """
	def write_file(filename, s):
		f = open(destination_folder + filename, 'w')
		f.write(s)
		f.close()

	# The small dataset was not added in the final release
	"""# Small dataset
	train, _ = load_dataset('dataset/', False)
	train = [preprocess(s) for s in train]
	train_pos = ''.join(train[:100000])
	train_neg = ''.join(train[100000:])
	write_file('train_pos.txt', train_pos)
	write_file('train_neg.txt', train_neg)"""

	# Full dataset
	train, _ = load_dataset('dataset/', True)
	train = [preprocess(s) for s in train]
	train_pos = ''.join(train[:1250000])
	train_neg = ''.join(train[1250000:])
	write_file('train_pos_full.txt', train_pos)
	write_file('train_neg_full.txt', train_neg)

	ids, test = load_csv_data('../data/test_data.txt')
	test = '\n'.join([str(id) + ',' + preprocess(s) for (id, s) in zip(ids, test)])
	write_file('test_data.txt', test)

if __name__ == '__main__':
	preprocess_dataset('preprocessed_dataset/')
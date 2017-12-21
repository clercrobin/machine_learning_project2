import numpy as np
import scipy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from keras.models import *
from keras.layers import *
from keras import regularizers
from helpers import *
from preprocessings import *

def separate_dataset(X, y, ratio):
    print('Separate dataset')
    nb_train = int(ratio * X.shape[0])
    return  X[:nb_train], y[:nb_train], X[nb_train:], y[nb_train:]

def train_vectorizer(dataset, tfidf=False, k=3):
	print('Vectorize')
	if tfidf:
		vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, k), max_features=10**6)
	else:
		vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, k), max_features=10**6)
	vectorizer.fit(dataset)
	return vectorizer

# Logistic regression

def train_lr(X, y):
	print('Train LR')
	model = LogisticRegression()
	model.fit(X_train, y_train)
	return model

def accuracy_lr(X, y, model):
	return model.score(X_valid, y_valid)

def predict_lr(X, model):
	y = model.predict(X_test)
	return (y * 2 - 1).astype(int) # Transform {0, 1} -> {-1, 1}

# Feedforward neural network

def train_nn(X, y):
	print('Train NN')
	# Architecture
	inputs = Input(shape=(X.shape[1],), sparse=True)
	x = Dense(16, activation='relu')(inputs)
	x = Dropout(0.5)(x)
	predictions = Dense(1, activation='sigmoid')(x)
	model = Model(inputs=inputs, outputs=predictions)

	# Optimization
	model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

	# Training
	for _ in range(10):
		model.fit(X, y, epochs=1, batch_size=256)
		print('Validation: ', model.evaluate(X_valid, y_valid, batch_size=256))

	return model

def accuracy_nn(X, y, model):
	return model.evaluate(X, y, batch_size=256)

def predict_nn(X, model):
	return ((model.predict(X) > 0.5) * 2 - 1).astype(int)

if __name__ == '__main__':
	# Load dataset
	train, y = load_dataset('preprocessed_dataset/', True)
	ids, test = load_csv_data('preprocessed_dataset/test_data.txt')

	# Create bags of words
	vectorizer = train_vectorizer(train + test, True, 3)
	X = vectorizer.transform(train)
	print(X.shape)
	X_test = vectorizer.transform(test)

	# Create a training and a validation set
	X, y = shuffle(X, y)
	X_train, y_train, X_valid, y_valid = separate_dataset(X, y, 0.95)

	# Train, evaluate and predict
	model_type = 'lr'
	if model_type == 'lr':
		model = train_lr(X_train, y_train)
		print('Accuracy: ', accuracy_lr(X_valid, y_valid, model))
		y = predict_lr(X_test, model)
	elif model_type == 'nn':
		model = train_nn(X_train, y_train)
		print('Accuracy: ', accuracy_nn(X_valid, y_valid, model))
		y = predict_nn(X_test, model)
	
	# Create the submission
	create_csv_submission(ids, y, 'submission_bow.csv')

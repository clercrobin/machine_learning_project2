# -*- coding: utf-8 -*-

import numpy as np
import _pickle as cPickle
from sklearn.linear_model import LogisticRegression
from helpers import create_csv_submission
from keras.models import load_model
import pickle

def trainLogisticRegression(train, y):
	""" We train our linear regression with the outputs of our models as features for each tweet """
	classifier = LogisticRegression() 
	model = classifier.fit(train, y) 
	return model

def predictAdapted(model,test):
	""" Predicts thanks to the trained logistic regression and adapts the labels from [0 1] to [-1 1] """
	yPred = model.predict(test)
	yPred = 1 - 2*yPred
	return yPred

def predictModels(model, xTrain, xTest):
	""" Generates the data used to train and predict via the logistic regression """
	train = model.predict_proba(xTrain, batch_size=256)
	test = model.predict_proba(xTest)
	return train,test



def loadTrainTest(interval, xTrain, xTest):
	"""This function merges the results for all train and test performance over each model in interval so that we can train a classifier with it"""
	y_train1,y_test1 = pickle.load(open('gloveBased/pickles/cnn_embeddings_1_output.pickle','rb'))
	y_train2,y_test2 = pickle.load(open('gloveBased/pickles/cnn_embeddings_2_output.pickle','rb'))
	i=0
	for x in interval:
		model = load_model("gloveBased/pickles/model"+str(x)+".h5")
		newTrain, newTest =  predictModels(model, xTrain, xTest)
		if i==0:
			train=newTrain
			test=newTest
		else:
			train = np.concatenate([train, newTrain],axis=1)
			test = np.concatenate([test, newTest],axis=1)
		i+=1
	train = np.concatenate([train, y_train1, y_train2],axis=1)
	test = np.concatenate([test, y_test1, y_test2], axis=1)
	return train, test

# Load all the tweets
# [xTrain, y, xTest, inputWords, weights] = cPickle.load(open("gloveBased/tweetRepresentations.dat", "rb"))

# Merge all the pickles
#train, test = loadTrainTest([21,22,23,24],xTrain,xTest)

#cPickle.dump([train, test, y],open('finalRepresentation.dat', 'wb'))

[train, test, y] = cPickle.load(open('finalRepresentation.dat', 'rb'))

model = trainLogisticRegression(train,y)
yPred = predictAdapted(model,test)

create_csv_submission(np.arange(1,10001),yPred,'submission.csv')
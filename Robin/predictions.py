# -*- coding: utf-8 -*-

import numpy as np
import _pickle as cPickle
from sklearn.ensemble import AdaBoostClassifier
from helpers import create_csv_submission
from keras.models import load_model

def trainAdaBoostClassifier(train, y):
	adaBoostClassifier = AdaBoostClassifier() 
	model = adaBoostClassifier.fit(train, y) # For each 
	return model

def predictAdapted(model,test):
	yPred = model.predict(test)
	yPred = 1 - 2*yPred
	return yPred

def predictModels(model, xTrain, xTest):
	train = model.predict_proba(xTrain, batch_size=256)
	test = model.predict_proba(xTest)
	return train,test



def loadTrainTest(interval, xTrain, xTest):
	"""This function merges the results for all train and test performance over each model in interval so that we can train a classifier with it"""

	i=0
	for x in interval:
		model = load_model("pickles/model"+str(x)+".h5")
		newTrain, newTest =  predictModels(model, xTrain, xTest)
		if i==0:
			train=newTrain
			test=newTest
		else:
			train = np.concatenate([train, newTrain],axis=1)
			test = np.concatenate([test, newTest],axis=1)
		i+=1
	return train, test

# Load all the tweets
[xTrain, y, xTest, inputWords, weights] = cPickle.load(open("tweetRepresentations.dat", "rb"))

# Merge all the pickles
train, test = loadTrainTest([21,22,23],xTrain,xTest)


model = trainAdaBoostClassifier(train,y)
yPred = predictAdapted(model,test)

create_csv_submission(np.arange(1,10001),yPred,'submission.csv')
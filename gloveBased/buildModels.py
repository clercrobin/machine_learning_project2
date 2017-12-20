import _pickle as cPickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
from keras.models import load_model


from tweetRepresentation import buildFeatures

# Take the online error to converge easily.
# Decrease the S shape : last layer relu, could help for the proba of the adaboost classifier
# derivative cross entropy + sigmoid = linear, easier, then finalize thanks to least mean square
# https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/


# Load the results
[xTrain, y, xTest, embeddingLayerInit, nbWords] = cPickle.load(open("tweetRepresentations.dat", "rb"))

#Model 21
model = Sequential()
model.add(Embedding(nbWords+1, 25, input_length=xTrain.shape[1], weights=[embeddingLayerInit]))
# Like in the lectures
model.add(Convolution1D(nb_filter=128, filter_length=5, border_mode='same', activation='sigmoid'))
model.add(MaxPooling1D(pool_length=3))
model.add(Flatten())
# We add a dense layer
model.add(Dense(256, activation='sigmoid'))
# The actual classification
model.add(Dense(1, activation='sigmoid'))
# Binary crossentropy because it is a binary classification and derivative of binary crossentropy + sigmoid is linear 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

#Fitting
model.fit(xTrain, y, validation_split=0.1, nb_epoch=3, batch_size=256, verbose=1)

#Save the model
model.save('pickles/model21.h5')

del model


#Model 22
model = Sequential()
model.add(Embedding(nbWords+1, 25, input_length=xTrain.shape[1], weights=[embeddingLayerInit]))
# Like in the lectures
model.add(Convolution1D(nb_filter=256, filter_length=7, border_mode='same', activation='sigmoid'))
model.add(MaxPooling1D(pool_length=5))
model.add(Flatten())
# We add a dense layer
model.add(Dense(256, activation='sigmoid'))
# The actual classification
model.add(Dense(1, activation='sigmoid'))
# Binary crossentropy because it is a binary classification and derivative of binary crossentropy + sigmoid is linear 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

#Fitting
model.fit(xTrain, y, validation_split=0.1, nb_epoch=3, batch_size=256, verbose=1)

#Save the model
model.save('pickles/model22.h5')


del model
#Model 23
model = Sequential()
model.add(Embedding(nbWords+1, 25, input_length=xTrain.shape[1], weights=[embeddingLayerInit]))
# Like in the lectures
model.add(Convolution1D(nb_filter=256, filter_length=7, border_mode='same', activation='sigmoid'))
model.add(MaxPooling1D(pool_length=5))

model.add(Convolution1D(nb_filter=64, filter_length=5, border_mode='same', activation='sigmoid'))
model.add(MaxPooling1D(pool_length=3))
model.add(Flatten())
# We add a dense layer
model.add(Dense(256, activation='sigmoid'))
# The actual classification
model.add(Dense(1, activation='sigmoid'))
# Binary crossentropy because it is a binary classification and derivative of binary crossentropy + sigmoid is linear 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

#Fitting
model.fit(xTrain, y, validation_split=0.1, nb_epoch=3, batch_size=256, verbose=1)

#Save the model
model.save('pickles/model23.h5')

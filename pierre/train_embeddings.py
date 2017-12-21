import pickle
import numpy as np
import scipy
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from keras.models import *
from keras.layers import *
from keras.preprocessing import text, sequence
from keras import regularizers
from helpers import *

def separate_dataset(X, y, ratio):
    print('Separate dataset')
    nb_train = int(ratio * len(X))
    return  X[:nb_train], y[:nb_train], X[nb_train:], y[nb_train:]

# Train with CNN

def train_cnn(X, y):
    print('Train CNN')
    # Architecture
    # CONV(5) - MAX(3) - CONV(5) - MAX(3) - DENSE(128)

    model = Sequential()
    model.add(Embedding(vocab_size, output_dim=200, input_length=X.shape[1]))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(3)) # global max pooling
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    # Optimization
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # Training
    for epoch in range(10):
        model.fit(X, y, epochs=1, batch_size=128)
        print('Validation: ', model.evaluate(X_valid, y_valid, batch_size=128))
        model.save('models/cnn_embeddings_{}'.format(epoch+1))

    return model

def accuracy_cnn(X, y, model):
    return model.evaluate(X, y, batch_size=128)

def predict_cnn(X, model):
    return ((model.predict(X) > 0.5) * 2 - 1).astype(int)

if __name__ == '__main__':
    # Load dataset
    train, y = load_dataset('preprocessed_dataset/', True)
    ids, test = load_csv_data('preprocessed_dataset/test_data.txt')

    # Transform tweets to sequences
    print('Create sequences')
    vocab_size = 200000 # Fix the size of the vocabulary
    tokenizer = text.Tokenizer(vocab_size, filters='"#$%&()*+,-/:;<=>@[\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(train + test)
    print('vocab size: {}'.format(vocab_size))
    train = tokenizer.texts_to_sequences(train)
    X = sequence.pad_sequences(train, maxlen=52) # Fix the number of words
    print(X.shape)

    # Create a training and a validation set
    X, y = shuffle(X, y)
    X_train, y_train, X_valid, y_valid = separate_dataset(X, y, 0.95)

    # Train, evaluate and predict
    model = train_cnn(X_train, y_train)
    print('Accuracy: ', accuracy_cnn(X_valid, y_valid, model))
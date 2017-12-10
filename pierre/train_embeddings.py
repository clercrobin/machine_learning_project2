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

# Train with LSTM

def train_lstm(X, y):
    print('Train LSTM')
    # Architecture
    model = Sequential()
    model.add(Embedding(embeddings.shape[0], output_dim=embeddings.shape[1], input_length=X.shape[1], \
        weights=[embeddings], trainable=False))
    #model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128))
    #model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # Optimization
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # Training
    model.fit(X, y, batch_size=128, epochs=2)

    return model

def accuracy_lstm(X, y, model):
    return model.evaluate(X, y, batch_size=128)

# Train with CNN

def train_cnn(X, y):
    print('Train CNN')
    # Architecture
    model = Sequential()
    model.add(Embedding(vocab_size, output_dim=100, input_length=X.shape[1]))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    #model.add(GlobalAveragePooling1D())
    model.add(MaxPooling1D(14)) # global max pooling
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    print(model.summary())

    """model.add(Conv1D(64, 3, activation='relu', input_shape=(seq_length, 100)))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))"""

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

if __name__ == '__main__':
    # Load dataset
    train, y = load_dataset('preprocessed_dataset/', False)

    # Transform tweets to sequences
    print('Create sequences')
    vocab_size = 100000 # Fix the size of the vocabulary
    tokenizer = text.Tokenizer(vocab_size)
    tokenizer.fit_on_texts(train)
    print('vocab size: {}'.format(vocab_size))
    train = tokenizer.texts_to_sequences(train)
    X = sequence.pad_sequences(train, maxlen=50) # Fix the number of words
    print(X.shape)

    # Create a training and a validation set
    X, y = shuffle(X, y)
    X_train, y_train, X_valid, y_valid = separate_dataset(X, y, 0.8)

    """# Load and prepare test dataset
    ids, test = load_csv_data('dataset/test_data.txt')
    X_test = vectorizer.transform(test)"""

    # Train, evaluate and predict
    model = 'cnn'
    if model == 'lstm':
        model = train_lstm(X_train, y_train)
        print('Accuracy: ', accuracy_lstm(X_valid, y_valid, model))
        #y = predict_lstm(X_test, model)
    elif model == 'cnn':
        model = train_cnn(X_train, y_train)
        print('Accuracy: ', accuracy_cnn(X_valid, y_valid, model))
        #y = predict_cnn(X_test, model)

    # Create the submission
    #create_csv_submission(ids, y, 'submission_bow.csv')"""

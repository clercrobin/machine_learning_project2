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

def text_to_sequence(vocab, s):
    seq = [vocab.get(t, -1) + 1 for t in s.strip().split()]
    return [x for x in seq if x > 0]

# Train with LR

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
    model.add(Embedding(embeddings.shape[0], output_dim=embeddings.shape[1], input_length=X.shape[1], \
        weights=[embeddings], trainable=False))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(128, 5, activation='relu'))
    #model.add(MaxPooling1D(5))
    #model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(5))  # global max pooling
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

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
    model.fit(X, y, batch_size=128, epochs=10)

    return model

def accuracy_cnn(X, y, model):
    return model.evaluate(X, y, batch_size=128)

if __name__ == '__main__':
    # Load dataset
    train, y = load_dataset(False)

    # Transform tweets to sequences
    print('Create sequences')
    vocab = pickle.load(open('glove/vocab.pkl', 'rb'))
    embeddings = np.load('glove/embeddings50_99.npy')[0]
    embeddings = np.vstack((np.zeros((1, embeddings.shape[1])), embeddings))

    train = [text_to_sequence(vocab, s) for s in train]
    X = sequence.pad_sequences(train)
    print(X.shape)

    # Create a training and a validation set
    X, y = shuffle(X, y)
    X_train, y_train, X_valid, y_valid = separate_dataset(X, y, 0.8)

    """# Load and prepare test dataset
    ids, test = load_csv_data('dataset/test_data.txt')
    X_test = vectorizer.transform(test)"""

    # Train, evaluate and predict
    model = 'cnn'
    if model == 'lr':
        print(X_train.shape)
        X_train = np.sum(embeddings[X_train], axis=1)
        print(X_train.shape)
        X_valid = np.sum(embeddings[X_valid], axis=1)
        model = train_lr(X_train, y_train)
        print('Accuracy: ', accuracy_lr(X_valid, y_valid, model))
    elif model == 'lstm':
        model = train_lstm(X_train, y_train)
        print('Accuracy: ', accuracy_lstm(X_valid, y_valid, model))
        #y = predict_lstm(X_test, model)
    elif model == 'cnn':
        model = train_cnn(X_train, y_train)
        print('Accuracy: ', accuracy_cnn(X_valid, y_valid, model))
        #y = predict_cnn(X_test, model)

    # Create the submission
    #create_csv_submission(ids, y, 'submission_bow.csv')"""

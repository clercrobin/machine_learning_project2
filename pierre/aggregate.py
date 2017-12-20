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

if __name__ == '__main__':
    # Load dataset
    train, y = load_dataset('preprocessed_stanford_dataset/', True)
    ids, test = load_csv_data('preprocessed_stanford_dataset/test_data.txt')

    # Transform tweets to sequences
    print('Create sequences')
    vocab_size = 200000 # Fix the size of the vocabulary
    tokenizer = text.Tokenizer(vocab_size, filters='"#$%&()*+,-/:;<=>@[\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(train + test)
    print('vocab size: {}'.format(vocab_size))
    train = tokenizer.texts_to_sequences(train)
    X = sequence.pad_sequences(train, maxlen=52) # Fix the number of words
    print(X.shape)
    test = tokenizer.texts_to_sequences(test)
    X_test = sequence.pad_sequences(test, maxlen=52)

    # Shuffle as Robin
    np.random.seed(0) # We set a particular seed to be able to do it again
    shuffled = np.arange(X.shape[0])
    np.random.shuffle(shuffled)
    X = X[shuffled]
    y = y[shuffled]

    # Output predictions
    model = load_model('models/cnn_embeddings_1')
    model.summary()
    y_train = model.predict(X)
    y_test = model.predict(X_test)
    pickle.dump((y_train, y_test), open('models/cnn_embeddings_1_output.pickle', 'wb'))

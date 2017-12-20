# -*- coding: utf-8 -*-

import sys
from preprocessing import wholeClean
from preprocessing import mergeDictionaries
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import numpy as np
import re
import _pickle as cPickle

def buildFeatures():
    
    
    dictionary = mergeDictionaries(3)
    contractions_dict = { 
        "ain't": "is not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he had",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "I'd": "I would",
        "I'd've": "I would have",
        "I'll": "I will",
        "I'll've": "I will have",
        "I'm": "I am",
        "I've": "I have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you had",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have"
        }
    contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

    positives = open("train_pos_full.txt", 'rb')
    negatives = open("train_neg_full.txt", 'rb')
    testfile = open('test_data.txt' ,'rb')
    glovetwitter25d = open('./embeddings/glovetwitter25d.txt','rb')

    # Build the 
    xTrain = []
    currentIndex = 0
    total = 2500000
    for tweet in positives:
        tweet = tweet.decode('utf8')
        tweet = wholeClean(tweet,dictionary, contractions_re,contractions_dict)
        xTrain.append(tweet)
        if currentIndex%1000 == 0:
            print(str(currentIndex)+ '/' + str(total) + ' processed tweets.')
            print(tweet)
        currentIndex = currentIndex + 1
    for tweet in negatives:
        tweet = tweet.decode('utf8')
        tweet = wholeClean(tweet,dictionary, contractions_re,contractions_dict)
        xTrain.append(tweet)
        if currentIndex%1000 == 0:
            print(str(currentIndex)+ '/' + str(total) + ' processed tweets.')
            print(tweet)
        currentIndex = currentIndex + 1
    negatives.close()
    positives.close()

    testSize = 10000
    xTest = []

    currentIndex = 0
    for tweet in testfile:
        tweet = tweet.decode('utf8')
        tweet = tweet[tweet.find(',')+1:]
        tweet = wholeClean(tweet,dictionary, contractions_re,contractions_dict)
        xTest.append(tweet)
        if currentIndex%1000 == 0:
            print(str(currentIndex)+ '/' + str(testSize) + ' processed tweets.')
            print(tweet)
        currentIndex = currentIndex + 1
    testfile.close()
    
    # Tokenize all the tweets thanks to the keras way
    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts(xTrain)
    wordIndex = tokenizer.word_index
    nbUniqueTokens = len(wordIndex)
    print('Found' + str(nbUniqueTokens) +' unique tokens')

    # Splits to a list of words
    trainSequences = tokenizer.texts_to_sequences(xTrain) 
    testSequences = tokenizer.texts_to_sequences(xTest)

    maxlen = 35;
    print('Pad sequences to get a fixed size input')
    trainSequences = sequence.pad_sequences(trainSequences, maxlen=maxlen)
    testSequences = sequence.pad_sequences(testSequences, maxlen=maxlen)


    # 0 for positive because in the end between [-1,1] => 1-2y
    yTrain = np.array(int(total/2) * [0] + int(total/2) * [1])

    # Shuffle all because it is firstly sorted
    np.random.seed(0) # We set a particular seed to be able to do it again
    shuffled = np.arange(trainSequences.shape[0])
    np.random.shuffle(shuffled)
    trainSequences = trainSequences[shuffled]
    yTrain = yTrain[shuffled]


    # We have to find the initialized values of the embedded layer
    print('Extracting glove')
    gloveDictionary = {}

    for line in glovetwitter25d:
        coordinates = line.split()
        word = coordinates[0]
        coords = np.asarray(coordinates[1:], dtype='float32')
        gloveDictionary[word] = coords
    glovetwitter25d.close()

    # We initialize what will be the embedding layer of our network
    embeddingLayerInitialization = np.zeros((nbUniqueTokens + 1, 25))
    for word, i in wordIndex.items():
        if i < nbUniqueTokens:
            embeddingCoords = gloveDictionary.get(word)
            if embeddingCoords: # If the word is not found the corresponding coefficients will be zeros and learned through training
                embeddingLayerInitialization[i] = embeddingCoords
    cPickle.dump([trainSequences, yTrain, testSequences, embeddingLayerInitialization, nbUniqueTokens],open('tweetRepresentations.dat', 'wb'))
    return None

# Generate all the pickles
buildFeatures()
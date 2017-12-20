# -*- coding: utf-8 -*-

import sys
import re
import itertools
from nltk.corpus import words


def removeMultipleLetters(tweet):
    tweet=tweet.split()
    for i in range(len(tweet)):
        tweet[i]=''.join(''.join(s)[:2] for _, s in itertools.groupby(tweet[i]))
    tweet=' '.join(tweet) # Puts back the space between each word
    return tweet

def removeHashtags(tweet):
    """This function removes the hashtag of the tweets"""
    return tweet.replace('#','')

def checkDictionary(tweet, dictionary):
    """Corrects the usual typos and abbreviations"""
    tweet = tweet.split()
    for i in range(len(tweet)):
        if tweet[i] in dictionary.keys():
            tweet[i] = dictionary[tweet[i]]
    tweet = ' '.join(tweet) # Puts back the space between each word
    return tweet


def expand_contractions(s, contractions_re, contractions_dict):
    """ This function expands the usual contractions"""
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)

def mergeDictionaries(number = 3):
    """ This function builds a merge of all the dictionay normalizations we found
    The dictionaries are from : A concatenation of several websites proposing a few most used abbreviations
    http://luululu.com/tweet/, the lexical normalization dictionary from http://people.eng.unimelb.edu.au/tbaldwin/, 
      """
    dictionary = {}
    for i in range(1,number+1):
        tempdictionary = open('dictionaries/textNormDict'+ str(i) +'.txt', 'rb')
        if i==1:
            l=[]
            row = 0
            for word in tempdictionary:
                word = word.decode('utf8')
                word = word.strip() 
                if row%2==0:
                    l.append(word)
                else:
                    l[(row-1)//2] = l[(row-1)//2]+" "+word
                row+=1
        else:
            for word in tempdictionary:
                word = word.decode('utf8')
                word = word.split()
                dictionary[word[0]] = word[1]
        tempdictionary.close()
    return dictionary


def wholeClean(tweet, dictionary,contractions_re,contractions_dict):
    """Cleans the whole tweet """
    tweet = removeHashtags(tweet) # We begin by removing the hashtags
    tweet = expand_contractions(tweet, contractions_re, contractions_dict)
    tweet = removeMultipleLetters(tweet)
    tweet = checkDictionary(tweet, dictionary)
    tweet = tweet.strip()
    
    return tweet

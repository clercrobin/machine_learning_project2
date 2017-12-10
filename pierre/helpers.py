# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np

def load_csv_data(data_path):
    """Loads data and returns ids and tweets"""
    file = open(data_path)
    ids = []
    tweets = []
    for line in file:
        i_comma = line.find(',')
        ids.append(int(line[:i_comma]))
        tweets.append(line[i_comma+1:].strip())
    return ids, tweets

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    print('Submit')
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

def load_dataset(folder_path='dataset/', full=False):
    print('Load dataset')
    if full:
        train_pos_file = open(folder_path + 'train_pos_full.txt')
        train_neg_file = open(folder_path + 'train_neg_full.txt')
    else:
        train_pos_file = open(folder_path + 'train_pos.txt')
        train_neg_file = open(folder_path + 'train_neg.txt')
    train_pos = [line for line in train_pos_file]
    train_neg = [line for line in train_neg_file]
    train = train_pos + train_neg
    y = np.concatenate((np.ones(len(train_pos)), np.zeros(len(train_neg))), axis=0)
    return train, y
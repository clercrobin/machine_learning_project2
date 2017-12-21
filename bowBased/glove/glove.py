#!/usr/bin/env python3
from scipy.sparse import *
import numpy as np
import pickle
import random


def main():
    print("loading cooccurrence matrix")
    with open('cooc.pkl', 'rb') as f:
        cooc = pickle.load(f)
    print("{} nonzero entries".format(cooc.nnz))

    nmax = 100
    print("using nmax =", nmax, ", cooc.max() =", cooc.max())

    # Parameters
    embedding_dim = 50
    alpha = 3 / 4
    epochs = 100
    eta = 0.001

    print("initializing embeddings")
    X = coo_matrix((np.log(cooc.data), (cooc.row, cooc.col)))
    F = coo_matrix((np.minimum(1, (cooc.data/nmax)**alpha), (cooc.row, cooc.col)))
    W = np.random.normal(size=(cooc.shape[0], embedding_dim))
    Z = np.random.normal(size=(cooc.shape[1], embedding_dim))

    for epoch in range(epochs):
        print("epoch {}".format(epoch))
        cost = 0
        for i, j, x, f in zip(cooc.row, cooc.col, X.data, F.data):
            error = (x - np.dot(W[i], Z[j]))
            scale = f * error
            W[i] += eta * Z[j] * scale
            Z[j] += eta * W[i] * scale
            cost += f * error * error
        print('approx cost: {}'.format(0.5*cost / cooc.data.size))
        if (epoch + 1) % 10 == 0:
            np.save('embeddings{}_{}'.format(embedding_dim, epoch+1), (W, Z))


if __name__ == '__main__':
    main()

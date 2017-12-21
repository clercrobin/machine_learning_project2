# gloveBased method Here must be saved tweetRepresentations.dat from the cloud


1. Add the dataset and the pre trained glove textfile in the data folder


2. Run buildModels.py Or add the models from the cloud in pickles and the tweetReprentations.dat in this folder

It will generate the tweet representations in a pickle via the importation of tweetRepresentation.py

and load it to train several models, each of them saved via h5


## File organization

preprocessing.py hosts the methods described in our report called to clean each tweet to build a better representation

tweetRepresentation.py Builds a pickle of the cleaned tweets aggregated to be used in neural networks

buildModels.py Build and train and save convolution neural networks.
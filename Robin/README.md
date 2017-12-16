# Project Text Sentiment Classification


1. Add the dataset and the glove

Add the kaggle files directly in the main directory, and the glove25 dimensions in the embeddings folder

2. Run tweetRepresentation

It will generate the tweet representations via a pickle

3. Run buildModels

Those representation will train several models, each of them saved via h5

4. Run predictions

It will load the previous models, apply them on the data so that you can train an AdaboostClassifier on to mean the results of those models
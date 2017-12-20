# Project 2

## Dependencies

The code uses :
* numpy
* scipy
* scikit-learn
* keras (with h5py to save the models)

## Folder organization

The dataset should be placed in "dataset/".

The preprocessed dataset will be placed in "preprocessed_dataset".

The GloVe embeddings and the files that generate them are located in "glove/".

Trained models are in the folder "models".

The main files for training are in the root folder.

## Helpers

In "helpers.py", there are three useful functions :
* `load_dataset(folder_path='dataset/', full=False)` to load the training set.
* `load_csv_data(data_path)` to load the test set.
* `create_csv_submission(ids, y_pred, name)` to create a submission.

## Preprocessings

The processings are inspired from the ones done by Stanford for GloVe. Regex are used to :
* replace smileys by tags : "<smile>", "<sadface>", "<surprisedface>", etc. ;
* replace numbers by "<number>". ;
* remove leters repeated at the end and signal it by a tag "<elong>" ;
* remove the repeated punctuation and signal it by a tag "<repeat>" ;
* add a tag "<hashtag>" before hashtags.

To preprocess the datasets, you just have to launch "preprocessings.py".

## Bag of words

The first approach which serves as baseline is using bag of words. 

In the file "train_bow.py", you will find code to create bag of words using tdf-if coefficients or not. Then, you can train a simple logistic regression model or feedforward neural networks.

## Glove

In the folder, you will find the code to generate GloVe embeddings.

## Embeddings

Finally, in "train_embeddings.py", you will find models that trained an embedding layer. A CNN model is available.

## Aggregate

In the file "aggregate.py", there are functions that allows to compute the output of a model on the training set and the test set. We can then use these files as input for another classifier that aggregates the results of several models.
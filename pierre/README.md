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

Currently, I "recreate" some smileys that have been tokenized incorrectly. Moreover, I replace tokens that contain a lot of digits by "\<number\>".

To preprocess the datasets, you just have to launch "preprocessings.py".

## Bag of words

The first approach which serves as baseline is using bag of words. 

In the file "train_bow.py", you will find code to create bag of words using tdf-if coefficients or not. Then, you can train a simple logistic regression model or feedforward neural networks.

Currently, neural networks do not improve the results. The best result on Kaggle test set is 82.34% using tdf-idf coefficients, logistic regression and the full dataset.

## Glove

In the folder, you will find the code to generate GloVe embeddings.

In "train_glove.py", there are the code to load GloVe embeddings and train models with. Logistic regression, LSTM and CNN models are available.

The results are very bad with embeddings trained on the dataset: the accuracy is aroung 60%. I will try with embeddings pretrained by Stanford.

## Embeddings

Finally, in "train_embeddings.py", you will find models that trained an embedding layer. LSTM (not tested) and CNN models are available.

Currently, they are the best models.
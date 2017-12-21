#!/bin/bash

echo 'Build vocabulary'
./build_vocab.sh
echo 'Cut vocabulary'
./cut_vocab.sh
echo 'Pickle vocabulary'
python3 pickle_vocab.py
echo 'Create matrix'
python3 cooc.py
echo 'Compute glove'
python3 glove.py
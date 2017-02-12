#!/bin/bash

source /shome/phwindis/setup_env

MODEL_DIR="$1"

python ./evaluateRNNClassifier-HDF-ext.py $MODEL_DIR

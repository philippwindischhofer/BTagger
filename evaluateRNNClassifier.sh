#!/bin/bash

source /shome/phwindis/setup_env

MODEL_DIR="$1"

python /shome/phwindis/DeepTagger/evaluateRNNClassifier-HDF.py $MODEL_DIR/model.h5 $MODEL_DIR/ROC
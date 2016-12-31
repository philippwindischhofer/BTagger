#!/bin/bash

source /shome/phwindis/setup_env

MODEL_DIR="$1"

python ./evaluateRNNClassifier-HDF.py $MODEL_DIR/model-final.h5 $MODEL_DIR/ROC

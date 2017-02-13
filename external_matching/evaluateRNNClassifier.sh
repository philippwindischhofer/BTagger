#!/bin/bash

source /shome/phwindis/setup_env

# only use one core for theano
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MKL_NUM_THREADS=1
export THEANO_FLAGS="base_compiledir=$HOME/theano.NOBACKUP"

MODEL_DIR="$1"

python ./evaluateRNNClassifier-HDF-ext.py $MODEL_DIR > $MODEL_DIR/eval.log

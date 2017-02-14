#!/bin/bash

# set up the environment
source /shome/phwindis/setup_env

# only use one core for training
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MKL_NUM_THREADS=1
export THEANO_FLAGS="base_compiledir=$HOME/theano.NOBACKUP"

./run.sh

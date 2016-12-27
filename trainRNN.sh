#!/bin/bash

# set up the environment
source /shome/phwindis/setup_env

# only use one core for training
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MKL_NUM_THREADS=1

# for local testing
JOB_ID=local_test_hdf

# for saving the output
OUTDIR=$HOME/DeepTagger/RNN_out/${JOB_ID}
echo $OUTDIR
mkdir -p $OUTDIR

# write all output files back here
python -m cProfile /shome/phwindis/DeepTagger/RNNClassifier-HDF.py `echo "$OUTDIR"`
#!/bin/bash

# set up the environment
source /shome/phwindis/setup_env

# only use one core for training
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MKL_NUM_THREADS=1
export THEANO_FLAGS="base_compiledir=$HOME/theano.NOBACKUP"

# for local testing
JOB_ID=LSTM32_1layer
JOB_DESC="training with /data/matched/1.h5, training batch size = 1000, 20 epochs, LSTM with 32 nodes, 1 layer"

# for saving the output
OUTDIR=$HOME/DeepTagger/RNN_out_external_matching/${JOB_ID}
echo $OUTDIR
mkdir -p $OUTDIR
echo $JOB_DESC
echo $JOB_DESC > $OUTDIR/desc.txt

# write all output files back here
python /shome/phwindis/DeepTagger/external_matching/RNNClassifier-HDF-ext.py `echo "$OUTDIR"` &> $OUTDIR/log.txt
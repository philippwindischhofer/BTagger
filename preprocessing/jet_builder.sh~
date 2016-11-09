#!/bin/bash

# set up the environment
source /shome/phwindis/setup_env

# only use one core for training
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MKL_NUM_THREADS=1

# for saving the output
OUTDIR=$HOME/DeepTagger/shallow_out/${JOB_ID}
echo $OUTDIR
mkdir -p $OUTDIR

# write all output files back here
python /shome/phwindis/DeepTagger/shallow/SoftmaxClassifierTheano.py `echo "$OUTDIR"`
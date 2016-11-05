#!/bin/bash

# set up the environment
source /shome/phwindis/setup_env

# only use one core for training
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MKL_NUM_THREADS=1

# for saving the output
OUTDIR=$HOME/DeepTagger/shallow/output/${JOB_ID}
echo $OUTDIR
mkdir -p $OUTDIR

# write all output files back here




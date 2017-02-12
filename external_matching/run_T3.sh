#!/bin/bash

# set up the environment
source /shome/phwindis/setup_env

# only use one core for training
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MKL_NUM_THREADS=1
export THEANO_FLAGS="base_compiledir=$HOME/theano.NOBACKUP"

# for local testing
JOB_ID=testing
JOB_DESC="pure test"

# for saving the output
OUTDIR=$HOME/DeepTagger/RNN_out_external_matching/${JOB_ID}
echo $OUTDIR
mkdir -p $OUTDIR
echo $JOB_DESC
echo $JOB_DESC > $OUTDIR/desc.txt

# to specify the model topology and the jet / track parameters accessible to the network
# number of nodes per layer
NODES=32
LAYERS=2
EPOCHS=10
TRAINING_LENGTH=1000

# number of tracks (-1 ... use all that are available)
TRACKS=10

# which track parameters are passed on to the network (as their indices)
# [ "Track_pt", "Track_eta", "Track_phi", "Track_dxy", "Track_dz", "Track_IP", "Track_IP2D", "Track_length" ]
PARAMETERS="0 1 2 3 4 5"

echo "starting"

# write all output files back here
python /shome/phwindis/DeepTagger/external_matching/RNNClassifier-HDF-ext.py `echo "$OUTDIR $EPOCHS $TRAINING_LENGTH $NODES $LAYERS $TRACKS $PARAMETERS"` > $OUTDIR/log.txt
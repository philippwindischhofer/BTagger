#!/bin/bash

# set up the environment
source /shome/phwindis/setup_env

# only use one core for training
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MKL_NUM_THREADS=1

# for manual running
JOB_ID=1

# for saving the output
OUTDIR=$HOME/DeepTagger/preprocessing_out/${JOB_ID}
echo $OUTDIR
mkdir -p $OUTDIR

JETS_IN="/mnt/t3nfs01/data01/shome/jpata/btv/gc/TagVarExtractor/GCa08e5e237323/TT_TuneCUETP8M1_13TeV-powheg-pythia8/job_0_out.root"

# here, tracks reside in the same file as the jets do
TRACKS_IN=$JETS_IN

# write all output files back here
python /shome/phwindis/DeepTagger/preprocessing/jet_builder.py `echo "$JETS_IN"` `echo "$TRACKS_IN"` `echo "$OUTDIR"`
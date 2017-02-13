#!/bin/bash
cd ~
source setup_env
cd DeepTagger

for i in {3..29}
do
    python root2hdf.py "/mnt/t3nfs01/data01/shome/jpata/btv/gc/TagVarExtractor/GCa08e5e237323/TT_TuneCUETP8M1_13TeV-powheg-pythia8/job_"$i"_out.root" "../"$i".h5"
done
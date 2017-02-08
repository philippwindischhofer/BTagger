#!/bin/bash
cd /shome/phwindis
source setup_env
cd /shome/phwindis/DeepTagger/external_matching

#for i in {3..29}
for i in {1..2}
do
    python jet_track_matcher.py "/mnt/t3nfs01/data01/shome/jpata/btv/gc/TagVarExtractor/GCa08e5e237323/TT_TuneCUETP8M1_13TeV-powheg-pythia8/job_"$i"_out.root" "/shome/phwindis/data/matched/"$i".h5"

# > /shome/phwindis/DeepTagger/external_matching/log.txt
done
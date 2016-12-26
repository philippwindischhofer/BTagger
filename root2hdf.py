from __future__ import division
import sys
import ROOT
import root_numpy as rnpy
import pandas as pd
import numpy as np

def main(argv):

    batch_size_jets = 1200
    batch_size_tracks = 5200
    read_pos_jets = 0
    read_pos_tracks = 0
    number_chunks = 0
    chunks_limit = 50

    print("importing jet list")
    # read in new chunk of jet and track data
    d1 = pd.DataFrame(rnpy.root2array("/mnt/t3nfs01/data01/shome/jpata/btv/gc/TagVarExtractor/GCa08e5e237323/TT_TuneCUETP8M1_13TeV-powheg-pythia8/job_0_out.root",
                                          treename = "tagVars/ttree"))
    
    print("importing tracks list")
    d2 = pd.DataFrame(rnpy.root2array("/mnt/t3nfs01/data01/shome/jpata/btv/gc/TagVarExtractor/GCa08e5e237323/TT_TuneCUETP8M1_13TeV-powheg-pythia8/job_0_out.root",
                                          treename = "tagVars/ttree_track"))

    print("creating HDF5")
    store = pd.HDFStore("/shome/phwindis/0.h5")
    print("writing jets")
    store.put('jets', d1, format = 'table')
    print("writing tracks")
    store.put('tracks', d2, format = 'table')
    print("closing HDF")
    store.close()
    
if __name__ == "__main__":
    main(sys.argv[1:])


from __future__ import division
import sys
import ROOT
import root_numpy as rnpy
import pandas as pd
import numpy as np
import seaborn
import pickle

def main(argv):
    jets_in = argv[0]
    tracks_in = argv[1]
    jets_out = argv[2]

    print "reading jets from " + jets_in
    d1 = pd.DataFrame(rnpy.root2array(jets_in, treename = "tagVars/ttree"))

    print "reading tracks from" + tracks_in
    d2 = pd.DataFrame(rnpy.root2array(tracks_in, treename = "tagVars/ttree_track"))

    d1['track_data'] = pd.np.empty((len(d1.index), 0)).tolist()

    print "joining tables"
    # iterate over the track list to join jets with the tracks belonging to them
    for irow, row in d2.iterrows():
        # these are the track data of the current track:
        tracks = row[["Track_pt", "Track_eta", "Track_phi", "Track_dxy", "Track_dz", "Track_IP", "Track_IP2D", "Track_length"]].as_matrix()
        jet_index = int(row["Track_jetIndex"])
        table_index = d1.loc[d1['Jet_jetIndex'] == jet_index].index[0]
    
        # append the tracks data to the matching jet in the main table
        d1['track_data'][table_index].append(tracks)
        
    print "sorting jets"
    # now divide the jets and put them in separate lists, according to their flavour
    jets_b = []
    jets_l = []
    jets_c = []

    # iterate over the jet list, with already matched tracks
    for irow, row in d1.iterrows():
        flavour = int(row["Jet_flavour"])
    
        # select the right list this jet belongs to
        if abs(flavour) == 5:
            jets = jets_b
        elif abs(flavour) == 4:
            jets = jets_c
        else:
            jets = jets_l
        
        # add the new jet to the list
        jets += [((row["Jet_pt"], row["Jet_eta"], row["Jet_phi"], row["Jet_mass"]), flavour, row["track_data"])]

    print "saving output"
    # save the thus processed data
    with open(jets_out + '/jets_b.dat', 'wb') as f:
        pickle.dump(jets_b, f)
    with open(jets_out + '/jets_l.dat', 'wb') as f:
        pickle.dump(jets_b, f)
    with open(jets_out + '/jets_c.dat', 'wb') as f:
        pickle.dump(jets_b, f)

if __name__ == "__main__":
    main(sys.argv[1:])

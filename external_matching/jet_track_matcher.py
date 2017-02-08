from __future__ import division
from __future__ import print_function
import sys
import ROOT
import root_numpy as rnpy
import pandas as pd
import numpy as np

def get_max_tracks(data):
    retval = 0
    for cur in data['track_data']:
        if len(cur) > retval:
            retval = len(cur)
    return retval

def equalize_tracks(data, set_tracks):
    empty = np.full(8, 0, float)
    for idx, cur in enumerate(data['track_data']):
        # take only these that are non-empty track lists
        if(len(cur) > 0):
            for i in range(set_tracks - len(cur)):
                data['track_data'][idx].append(empty)

def create_track_columns(set_tracks, number_parameters):
    colnames = []
    for i in range(set_tracks * number_parameters):
        colnames.append('T' + str(i))
    return colnames

def create_track_table(data):
    set_tracks = len(data['track_data'][0])
    number_parameters = len(data['track_data'][0][0])
    
    tracks = []
    colnames = create_track_columns(set_tracks, number_parameters)
    
    for cur in data['track_data']:
        arr = np.array(cur)
        tracks.append(arr.flatten())
        
    return pd.DataFrame(tracks, columns=colnames)

def save_dataset(file, data, **metadata):
    store = pd.HDFStore(file)
    store.put('data', data, format = 'table')
    store.get_storer('data').attrs.metadata = metadata
    store.close()

def main(argv):

    #batch_size_jets = 2400000
    #batch_size_tracks = 10400000

    batch_size_jets = 250000
    batch_size_tracks = 1100000

    #batch_size_jets = 2500
    #batch_size_tracks = 11000

    #batch_size_jets = 24000
    #batch_size_tracks = 104000

    print("importing jet list")
    # read in new chunk of jet and track data
    d1 = pd.DataFrame(rnpy.root2array(argv[0], treename = "tagVars/ttree", start = 0, stop = batch_size_jets))
    
    print("importing tracks list")
    d2 = pd.DataFrame(rnpy.root2array(argv[0], treename = "tagVars/ttree_track", start = 0, stop = batch_size_tracks))
    
    # figure out where the next chunk should start so that we don't count any jets multiple times
    last_tracks = (int)(d2.tail(1)['Track_jetIndex'].iloc[0]-1)
    last_jet = (int)(d1.tail(1)['Jet_jetIndex'].iloc[0]-1)
    
    # find the latest jet index thatis fully contained in this chunk
    while(len(d2.loc[d2['Track_jetIndex'] == last_tracks]) == 0):
        last_tracks -= 1
        
    if last_tracks > last_jet:
        print("Error: have more tracks than jets! Choose different chunk sizes!")

    # add the track data to the jet list
    d1['track_data'] = pd.np.empty((len(d1.index),0)).tolist()
    
    # iterate over the track list to join jets with the tracks belonging to them
    for irow, row in d2.iterrows():
        # these are the track data of the current track:
        tracks = row[["Track_pt", "Track_eta", "Track_phi", "Track_dxy", "Track_dz", "Track_IP", "Track_IP2D", "Track_length"]].as_matrix()
        jet_index = int(row["Track_jetIndex"])
        if jet_index > last_tracks:
            break
        table_index = d1.loc[d1['Jet_jetIndex'] == jet_index].index[0]

        # append the tracks data to the matching jet in the main table
        d1['track_data'][table_index].append(tracks)
        print(str(jet_index) + "/" + str(last_tracks), end = '\r')
        sys.stdout.flush()

    # extract only the fully matched rows for futher processing
    matched = d1.loc[d1['Jet_jetIndex'] < last_tracks - 1]
    
    # retain only the ones with a non-zero number of tracks:
    track_lengths = np.array(map(len, matched['track_data']))
    matched = matched.loc[track_lengths > 0]
    matched = matched.reset_index(drop=True)

    # specify number of jet parameters (always eight)
    number_parameters = 8
    set_tracks = get_max_tracks(matched)

    print("max number of tracks in this chunk: " + str(set_tracks))

    equalize_tracks(matched, set_tracks)
    track_table = create_track_table(matched)

    final = pd.concat([matched.ix[:,0:-1], track_table], axis = 1)
    
    print("number_tracks = " + str(set_tracks))
    print("number_jets = " + str(len(final)))
    save_dataset(argv[1], final, number_tracks = set_tracks, number_jets = len(final))
    
if __name__ == "__main__":
    print("this is the jet-track matching script")
    sys.stdout.flush()
    main(sys.argv[1:])




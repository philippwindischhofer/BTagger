import numpy as np
import pandas as pd

def read_metadata(store):
    return store.get_storer('data').attrs.metadata

def create_track_columns(set_tracks, number_parameters):
    colnames = []
    for i in range(set_tracks * number_parameters):
        colnames.append('T' + str(i))
    return colnames

def create_track_list(table, set_tracks, parameters_requested, tracks_requested, ordered = False):
    # number of jet parameters: always eight
    number_parameters = 8
    number_jets = len(table)

    # extract ALL the jet track columns present in the datafile
    cols = create_track_columns(set_tracks, number_parameters)

    # extract raw matrix
    tracks = table.ix[:,cols].as_matrix()
    tracks = tracks.reshape(number_jets, -1, number_parameters)

    # pt-order the tracks here, if demanded
    if ordered:
        tracks = np.array([sorted(cur, key = lambda tracks: tracks[0], reverse = True) for cur in tracks])

    # return only first n tracks / parameters here
    return tracks[:, 0:tracks_requested, 0:parameters_requested]

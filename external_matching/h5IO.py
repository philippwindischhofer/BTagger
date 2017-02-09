import numpy as np
import pandas as pd

def read_metadata(store):
    return store.get_storer('data').attrs.metadata

def create_track_columns(set_tracks, number_parameters):
    colnames = []
    for i in range(set_tracks * number_parameters):
        colnames.append('T' + str(i))
    return colnames

def create_track_list(table, set_tracks, number_parameters, ordered = False):
    number_jets = len(table)
    cols = create_track_columns(set_tracks, number_parameters)

    # extract raw matrix
    tracks = table.ix[:,cols].as_matrix()
    tracks = tracks.reshape(number_jets, -1, number_parameters)

    # TODO: return only first n tracks / parameters here

    # pt-order the tracks here, if demanded
    if ordered:
        return np.array([sorted(cur, key = lambda tracks: tracks[0], reverse = True) for cur in tracks])

    return tracks

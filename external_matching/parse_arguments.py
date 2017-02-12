import numpy as np
import pickle

def save_arguments(path, data):
    output = open(path, 'ab+')
    pickle.dump(data, output)
    output.close()

def read_arguments(path):
    output = open(path, 'rb')
    retval = pickle.load(output)
    output.close()
    return retval

def parse_arguments(argv):
    outdir = argv[0]
    epochs = int(argv[1])
    training_length = int(argv[2])
    nodes = int(argv[3])
    layers = int(argv[4])
    number_tracks = int(argv[5])
    parameters = np.array([])
    
    print("track parameters")
    for cur in argv[6:]:
        parameters = np.append(parameters, int(cur))
    parameters = parameters.astype(int)

    return dict([('outdir', outdir), ('number_epochs', epochs), ('training_length', training_length), ('number_nodes', nodes), ('number_layers', layers), ('number_tracks',  number_tracks), ('track_parameters', parameters)])

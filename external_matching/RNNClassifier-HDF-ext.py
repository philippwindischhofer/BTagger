from __future__ import division
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.optimizers import SGD
from keras.layers import Dense, Activation
from keras.layers import LSTM

# build here the keras model
def RNN_classifier():
    model = Sequential()
    
    model.add(LSTM(32, return_sequences = False, input_shape = (None, 8)))
    #model.add(LSTM(64, return_sequences = True))
    #model.add(LSTM(64))
    
    # make an output layer with just 1 output -> for a binary classification problem: b-jet / not b-jet
    model.add(Dense(1, activation='sigmoid'))

    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer = sgd)
  
    return model

def read_metadata(store):
    return store.get_storer('data').attrs.metadata

def create_track_columns(set_tracks, number_parameters):
    colnames = []
    for i in range(set_tracks * number_parameters):
        colnames.append('T' + str(i))
    return colnames

def create_track_list(table, set_tracks, number_parameters):
    number_jets = len(table)
    cols = create_track_columns(set_tracks, number_parameters)

    # extract raw matrix
    tracks = table.ix[:,cols].as_matrix()
    return tracks.reshape(number_jets, -1, number_parameters)

def main(argv):

    loss_history = []
    loss_val_history = []

    number_epochs = 5
    number_jet_parameters = 8
    training_dataset_length = 10000
    datafile = '/shome/phwindis/data/matched/1.h5'
    #datafile = '/scratch/snx3000/phwindis/0.h5'

    # create model
    model = RNN_classifier()

    # read in training data
    with pd.HDFStore(datafile) as store:
        metadata = read_metadata(store)
    number_tracks = metadata['number_tracks']

    print("reading training data")
    raw_data = pd.read_hdf(datafile, start = 0, stop = training_dataset_length)

    # build training input and output:
    x_train = create_track_list(raw_data, number_tracks, number_jet_parameters)
    y_train = np.array(abs(raw_data['Jet_flavour']) == 5)

    # converts boolean array into int!
    y_train = y_train * 1
    y_train = y_train.reshape((len(y_train), 1))

    print(x_train.shape)
    print(y_train.shape)

    #print(x_train)
    #print(y_train)

    print("start training")
    epoch_history = model.fit(x_train, y_train, validation_split = 0.20, nb_epoch = number_epochs)

    # update loss histories:
    loss_history.append(epoch_history.history['loss'])
    loss_val_history.append(epoch_history.history['val_loss'])

    #MODEL_OUT = argv[0] + '/model-' + str(number_chunks) + '.h5'
    #print("saving fitted model to " + MODEL_OUT)
    #model.save(MODEL_OUT)

    # process the training and validation loss histories
    #fig = plt.figure(figsize=(10,6))
    #plt.plot(np.reshape(loss_history, np.size(loss_history)), label = "training loss")
    #plt.plot(np.reshape(loss_val_history, np.size(loss_val_history)), label = "validation loss")
    #plt.xlabel("epoch")
    #plt.ylabel("loss")
    #plt.legend()
    #plt.savefig(argv[0] + '/loss-history.pdf')

    # save the model after training here
    #MODEL_OUT = argv[0] + '/model-final.h5'
    #print("saving fitted model to " + MODEL_OUT)
    #model.save(MODEL_OUT)

if __name__ == "__main__":
    main(sys.argv[1:])


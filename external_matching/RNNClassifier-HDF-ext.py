from __future__ import division
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.optimizers import SGD
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from h5IO import *

# build here the keras model
def RNN_classifier():
    model = Sequential()
    
    model.add(LSTM(64, return_sequences = False, input_shape = (None, 8)))
    #model.add(LSTM(128, return_sequences = True))
    #model.add(LSTM(128))
    
    # make an output layer with just 1 output -> for a binary classification problem: b-jet / not b-jet
    #model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))

    # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
  
    return model

def create_truth_output(raw_data):
    y_train = np.array(abs(raw_data['Jet_flavour']) == 5)

    # converts boolean array into int!
    y_train = y_train * 1
    y_train = y_train.reshape((len(y_train), 1))

    return y_train

def main(argv):

    loss_history = []
    loss_val_history = []

    number_epochs = 10

    training_dataset_length = 100000
    datafile = '/shome/phwindis/data/matched/1.h5'
    #datafile = '/scratch/snx3000/phwindis/0.h5'

    # create model
    model = RNN_classifier()

    # read in training data
    with pd.HDFStore(datafile) as store:
        metadata = read_metadata(store)
    number_tracks = metadata['number_tracks']

    jet_parameters_requested = 8 # number of parameters used for the classification
    tracks_requested = number_tracks # max. number of tracks used for each jet

    print("max number of tracks is " + str(number_tracks))

    print("reading training data")
    raw_data = pd.read_hdf(datafile, start = 0, stop = training_dataset_length)

    # build training input and output:
    x_train = create_track_list(raw_data, number_tracks, jet_parameters_requested, tracks_requested, ordered = False)
    y_train = create_truth_output(raw_data)

    print(x_train.shape)
    print(y_train.shape)

    print("start training")
    history = model.fit(x_train, y_train, validation_split = 0.20, nb_epoch = number_epochs, batch_size = 1, shuffle = True)

    # process the training and validation loss histories
    fig = plt.figure(figsize=(10,6))
    plt.plot(history.history['loss'], label = "training loss")
    plt.plot(history.history['val_loss'], label = "validation loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(argv[0] + '/loss-history.pdf')

    # process the training and validation loss histories
    fig = plt.figure(figsize=(10,6))
    plt.plot(history.history['acc'], label = "training accuracy")
    plt.plot(history.history['val_acc'], label = "validation accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(argv[0] + '/accuracy-history.pdf')

    # save the model after training here
    MODEL_OUT = argv[0] + '/model-final.h5'
    print("saving fitted model to " + MODEL_OUT)
    model.save(MODEL_OUT)

if __name__ == "__main__":
    main(sys.argv[1:])


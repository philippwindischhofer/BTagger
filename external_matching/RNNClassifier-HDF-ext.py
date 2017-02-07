from __future__ import division
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM

# build here the keras model
def RNN_classifier():
    model = Sequential()
    
    model.add(LSTM(64, return_sequences = True, input_shape = (None, 6)))
    model.add(LSTM(64, return_sequences = True))
    model.add(LSTM(64))
    
    # make an output layer with just 1 output -> for a binary classification problem: b-jet / not b-jet
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  
    return model

def shuffle_synchronized(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def prepare_training_data(jet_list, label, set_tracks):
    # extract the tracks and put them in pt-order, hardest tracks first
    jet_tracks = [cur[-1] for cur in jet_list]
    jet_tracks = [sorted(cur, key = lambda tracks: tracks[0], reverse = True) for cur in jet_tracks]
    
    # zero-pad the track dimension, to make sure all jets fed into the network during training have the same length
    max_tracks = max([len(cur) for cur in jet_tracks])
    padded = [np.vstack([cur, np.full((set_tracks - len(cur), 6), 0, float)]) for cur in jet_tracks]
    
    batch_size = len(padded)
    timestep_size = set_tracks
    jet_dim = 8
    x_train = np.array(padded).reshape(batch_size, timestep_size, jet_dim)
    y_train = np.full((batch_size, 1), label, float) # all are b-jets!
    
    return x_train, y_train, batch_size

def main(argv):

    number_epochs = 50
    model = RNN_classifier()

    loss_history = []
    loss_val_history = []

    epoch_cnt = 0
    while epoch_cnt < number_epochs:
        print("chunk number " + str(number_chunks))
        number_chunks += 1
    
        datafile = '/shome/phwindis/0.h5'
        #datafile = '/scratch/snx3000/phwindis/0.h5'

        # read in new chunk of jet and track data
    
            # select the right list this jet belongs to
            if abs(flavour) == 5:
                jets = jets_b
            elif abs(flavour) == 4:
                jets = jets_c
            else:
                jets = jets_l

        print("prepare data")
        max_tracks = get_max_tracks(jets_b, jets_c, jets_l)
        # now, have sorted jets in three lists, can use them directly for training!
        x_train_b, y_train_b, batch_size_b = prepare_training_data(jets_b, 1, max_tracks)
        x_train_c, y_train_c, batch_size_c = prepare_training_data(jets_c, 0, max_tracks)
        x_train_l, y_train_l, batch_size_l = prepare_training_data(jets_l, 0, max_tracks)
        x_train = np.vstack([x_train_b, x_train_c, x_train_l])
        y_train = np.vstack([y_train_b, y_train_c, y_train_l])

        x_train, y_train = shuffle_synchronized(x_train, y_train)

        print("start training")
        epoch_history = model.fit(x_train, y_train, validation_split = 0.20, batch_size = batch_size_b + batch_size_l + batch_size_c, nb_epoch = 40)

        # update loss histories:
        loss_history.append(epoch_history.history['loss'])
        loss_val_history.append(epoch_history.history['val_loss'])

        MODEL_OUT = argv[0] + '/model-' + str(number_chunks) + '.h5'
        print("saving fitted model to " + MODEL_OUT)
        model.save(MODEL_OUT)

    # process the training and validation loss histories
    fig = plt.figure(figsize=(10,6))
    plt.plot(np.reshape(loss_history, np.size(loss_history)), label = "training loss")
    plt.plot(np.reshape(loss_val_history, np.size(loss_val_history)), label = "validation loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(argv[0] + '/loss-history.pdf')

    # save the model after training here
    MODEL_OUT = argv[0] + '/model-final.h5'
    print("saving fitted model to " + MODEL_OUT)
    model.save(MODEL_OUT)

if __name__ == "__main__":
    main(sys.argv[1:])


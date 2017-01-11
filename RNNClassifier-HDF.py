
# coding: utf-8

# In[1]:

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

# In[2]:

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

# In[22]:

def prepare_training_data(jet_list, label, set_tracks):
    # extract the tracks and put them in pt-order, hardest tracks first
    jet_tracks = [cur[-1] for cur in jet_list]
    jet_tracks = [sorted(cur, key = lambda tracks: tracks[0], reverse = True) for cur in jet_tracks]
    
    # zero-pad the track dimension, to make sure all jets fed into the network during training have the same length
    max_tracks = max([len(cur) for cur in jet_tracks])
    padded = [np.vstack([cur, np.full((set_tracks - len(cur), 6), 0, float)]) for cur in jet_tracks]
    
    batch_size = len(padded)
    timestep_size = set_tracks
    jet_dim = 6
    x_train = np.array(padded).reshape(batch_size, timestep_size, jet_dim)
    y_train = np.full((batch_size, 1), label, float) # all are b-jets!
    
    return x_train, y_train, batch_size

def get_max_tracks(jets_b, jets_l, jets_c):
    b_tracks = [cur[-1] for cur in jets_b]
    l_tracks = [cur[-1] for cur in jets_l]
    c_tracks = [cur[-1] for cur in jets_c]
    
    max_tracks_b = max([len(cur) for cur in b_tracks])
    max_tracks_l = max([len(cur) for cur in l_tracks])
    max_tracks_c = max([len(cur) for cur in c_tracks])
    
    return np.max([max_tracks_b, max_tracks_l, max_tracks_c])

def main(argv):
    # In[23]:

    batch_size_jets = 12000
    batch_size_tracks = 52000
    read_pos_jets = 0
    read_pos_tracks = 0
    number_chunks = 0
    chunks_limit = 50

    # In[24]:

    model = RNN_classifier()
    #print("loading model back for further training")
    #model = load_model('/users/phwindis/BTagger/RNN_out/lstm64_3layers_singlestep_250/model-final.h5')

    loss_history = []
    loss_val_history = []

    # In[25]:

    while number_chunks < chunks_limit:
        print("chunk number " + str(number_chunks))
        number_chunks += 1
    
        #datafile = '/shome/phwindis/0.h5'
        datafile = '/scratch/snx3000/phwindis/0.h5'

        # read in new chunk of jet and track data
        d1 = pd.read_hdf(datafile, key = 'jets', start = read_pos_jets, stop = read_pos_jets + batch_size_jets)
        d1 = d1.reset_index(drop=True)
    
        d2 = pd.read_hdf(datafile, key = 'tracks', start = read_pos_tracks, stop = read_pos_tracks + batch_size_tracks)
        d2 = d2.reset_index(drop=True)        

        # break if we reached the end of the file
        if len(d1) < batch_size_jets:
            print("reached end of jet tree")
            break;
        
        if len(d2) < batch_size_tracks:
            print("reached end of track tree")
            break;
    
        # figure out where the next chunk should start so that we don't count any jets multiple times
        last_tracks = (int)(d2.tail(1)['Track_jetIndex'].iloc[0]-1)
        last_jet = (int)(d1.tail(1)['Jet_jetIndex'].iloc[0]-1)
    
        # find the latest jet index thatis fully contained in this chunk
        while(len(d2.loc[d2['Track_jetIndex'] == last_tracks]) == 0):
            last_tracks -= 1
        
        if last_tracks > last_jet:
            print("Error: have more tracks than jets! Choose different chunk sizes!")
    
        read_pos_jets += (d1.loc[d1['Jet_jetIndex'] == last_tracks].index[-1] + 1)
        read_pos_tracks += (d2.loc[d2['Track_jetIndex'] == last_tracks].index[-1] + 1)

        # add the track data to the jet list
        d1['track_data'] = pd.np.empty((len(d1.index),0)).tolist()
    
        # iterate over the track list to join jets with the tracks belonging to them
        for irow, row in d2.iterrows():
            # these are the track data of the current track:
            tracks = row[["Track_pt", "Track_dxy", "Track_dz", "Track_IP", "Track_IP2D", "Track_length"]].as_matrix()
            #tracks = row[["Track_pt", "Track_eta", "Track_phi", "Track_dxy", "Track_dz", "Track_IP", "Track_IP2D", "Track_length"]].as_matrix()
            jet_index = int(row["Track_jetIndex"])
            if jet_index > last_tracks:
                break
            table_index = d1.loc[d1['Jet_jetIndex'] == jet_index].index[0]

            # append the tracks data to the matching jet in the main table
            d1['track_data'][table_index].append(tracks)
    
        # now divide the jets and put them in separate lists, according to their flavour
        jets_b = []
        jets_l = []
        jets_c = []

        # iterate over the jet list, with already matched tracks
        for irow, row in d1.iterrows():
            jet_index = int(row["Jet_jetIndex"])
            if jet_index > last_tracks:
                break

            flavour = int(row["Jet_flavour"])

            # select the right list this jet belongs to
            if abs(flavour) == 5:
                jets = jets_b
            elif abs(flavour) == 4:
                jets = jets_c
            else:
                jets = jets_l

            # add the new jet to the list, if it contains any resolved tracks
            if len(row["track_data"]) > 0:
                jets += [(row["Jet_pt"], row["Jet_eta"], row["Jet_phi"], row["Jet_mass"], flavour, row["track_data"])]

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

    # In[26]:
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


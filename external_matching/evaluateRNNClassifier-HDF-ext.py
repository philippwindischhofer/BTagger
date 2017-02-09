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
from sklearn import metrics
from h5IO import *

def build_roc(response_b, response_nb):
    # produce the ROC plot and save it so that it can be combined with the others
    efficiency = np.array([])
    misid_prob = np.array([])

    # find the optimal threshold values for the plotting
    minval = np.min(np.concatenate([response_nb, response_b]))
    maxval = np.max(np.concatenate([response_nb, response_b]))
    for threshold in np.arange(minval, maxval, (maxval - minval) / 1000):
        correct_b = (response_b >= threshold).sum()
        misid_b = (response_nb >= threshold).sum()
            
        if(correct_b > 1 and misid_b > 1):
            efficiency = np.append(efficiency, correct_b / len(response_b))
            misid_prob = np.append(misid_prob, misid_b / len(response_nb))

    # compute AUC
    auc = np.trapz(y = efficiency, x = misid_prob)

    return misid_prob, efficiency, abs(auc)

def create_truth_output(raw_data):
    y_train = np.array(abs(raw_data['Jet_flavour']) == 5)

    # converts boolean array into int!                                                                             
    y_train = y_train * 1
    y_train = y_train.reshape((len(y_train), 1))

    return y_train

def main(argv):

    evaluation_dataset_length = 200000
    number_jet_parameters = 8

    # load back the keras model
    print("loading model from " + argv[0])
    model = load_model(argv[0])

    # loading the validation dataset
    datafile = '/shome/phwindis/data/matched/1.h5'

    with pd.HDFStore(datafile) as store:
        metadata = read_metadata(store)
    number_tracks = metadata['number_tracks']

    print("reading evaluation data")
    raw_data = pd.read_hdf(datafile, start = 0, stop = evaluation_dataset_length)

    # build validation input and output
    x_validation = create_track_list(raw_data, number_tracks, number_jet_parameters, ordered = True)
    y_validation = create_truth_output(raw_data).flatten()

    # obtain the model's response for the evaluation data
    response_rnn = model.predict(x_validation, batch_size = len(x_validation)).flatten()

    # get the CMVA response for the same data
    response_cmva = np.array(raw_data['Jet_cMVA']).flatten()

    # produce the ROC curves here
    misid_rnn, efficiency_rnn, _ = metrics.roc_curve(y_validation, response_rnn, pos_label = 1)
    misid_cmva, efficiency_cmva, _ = metrics.roc_curve(y_validation, response_cmva, pos_label = 1)

    print("plotting...")
    fig = plt.figure(figsize=(10,6))
    plt.plot(efficiency_rnn, misid_rnn, label = "LSTM")
    plt.plot(efficiency_cmva, misid_cmva, label = "cMVA")
    plt.yscale('log')
    axes = plt.gca()
    axes.set_ylim([1e-2,1])
    plt.xlabel('b jet efficiency')
    plt.ylabel('misidentification prob.')
    plt.legend(loc = "upper left")
    fig.savefig(argv[1] + '-plot.pdf')

if __name__ == "__main__":
    main(sys.argv[1:])



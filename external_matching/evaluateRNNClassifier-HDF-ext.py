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
from parse_arguments import *

def create_truth_output(raw_data):
    y_train = np.array(abs(raw_data['Jet_flavour']) == 5)

    # converts boolean array into int!                                                                             
    y_train = y_train * 1
    y_train = y_train.reshape((len(y_train), 1))

    return y_train

def main(argv):

    evaluation_dataset_length = 200000
    number_jet_parameters = 8
    model_path = argv[0] + '/model-final.h5'
    params_path = argv[0] + '/params.dat'

    # load back the keras model
    print("loading model from " + model_path)
    model = load_model(model_path)

    # load back the parameters
    args = read_arguments(params_path)
    print(args)

    # loading the validation dataset
    datafile = '/shome/phwindis/data/matched/3.h5'

    with pd.HDFStore(datafile) as store:
        metadata = read_metadata(store)
    number_tracks = metadata['number_tracks']

    jet_parameters_requested = args['track_parameters'] # request all jet parameters

    if args['number_tracks'] < 0:
        tracks_requested = np.arange(number_tracks) # request all tracks for each jet
    else:
        tracks_requested = np.arange(args['number_tracks']) # request the specified ones

    print("reading evaluation data")
    raw_data = pd.read_hdf(datafile, start = 0, stop = evaluation_dataset_length)

    print("building evaluation jetlist")
    # build validation input and output
    x_validation = create_track_list(raw_data, number_tracks, jet_parameters_requested, tracks_requested, ordered = True)
    y_validation = create_truth_output(raw_data).flatten()
    print(x_validation.shape)

    print("obtaining the model's predictions")
    # obtain the model's response for the evaluation data
    response_rnn = model.predict(x_validation, batch_size = len(x_validation)).flatten()

    print("producing the curves")
    # get the CMVA response for the same data
    response_cmva = np.array(raw_data['Jet_cMVA']).flatten()

    # produce the ROC curves here
    misid_rnn, efficiency_rnn, _ = metrics.roc_curve(y_validation, response_rnn, pos_label = 1)
    misid_cmva, efficiency_cmva, _ = metrics.roc_curve(y_validation, response_cmva, pos_label = 1)

    rocdata_rnn = np.vstack([misid_rnn, efficiency_rnn])
    rocdata_cmva = np.vstack([misid_cmva, efficiency_cmva])
    np.save(argv[0] + '/rocdata_rnn', rocdata_rnn)
    np.save(argv[0] + '/rocdata_cmva', rocdata_cmva)

    # compute AUC
    rnn_auc = metrics.roc_auc_score(y_validation, response_rnn)
    cmva_auc = metrics.roc_auc_score(y_validation, response_cmva)
    print("AUC(RNN) = " + str(rnn_auc))
    print("AUC(cMVA) = " + str(cmva_auc))

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
    fig.savefig(argv[0] + '/ROC-plot.pdf')

    print(len(response_rnn[y_validation == 1]))
    print(len(response_cmva[y_validation == 1]))

    print(len(response_rnn[y_validation == 0]))
    print(len(response_cmva[y_validation == 0]))

    # plot the RNN output vs. the CMVA output
    fig = plt.figure(figsize=(10,6))
    plt.hexbin(response_rnn[y_validation == 1], response_cmva[y_validation == 1], gridsize = 30, mincnt = 1, bins = 'log')
    plt.title('b jets')
    plt.xlabel('RNN output')
    plt.ylabel('cMVA output')
    cb = plt.colorbar()
    cb.set_label('log10(N)')
    fig.savefig(argv[0] + '/corrplot_b.pdf')

    fig = plt.figure(figsize=(10,6))
    plt.hexbin(response_rnn[y_validation == 0], response_cmva[y_validation == 0], gridsize = 30, mincnt = 1, bins = 'log')
    plt.title('non-b jets')
    plt.xlabel('RNN output')
    plt.ylabel('cMVA output')
    cb = plt.colorbar()
    cb.set_label('log10(N)')
    fig.savefig(argv[0] + '/corrplot_non_b.pdf')

if __name__ == "__main__":
    main(sys.argv[1:])



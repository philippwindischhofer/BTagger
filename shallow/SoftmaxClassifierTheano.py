from __future__ import division
import sys
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# build here the keras model (4 inputs, 4 outputs)
def simple_classifier():
    model = Sequential()
    model.add(Dense(24, input_dim = 4, init = 'normal', activation = 'relu'))
    model.add(Dense(4, init = 'normal', activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

def makeOneHot(data_out, data_in):
    data_out.loc[data_in['Jet_flavour_class'] == 0, 'light'] = 1
    data_out.loc[data_in['Jet_flavour_class'] == 1, 'charm'] = 1
    data_out.loc[data_in['Jet_flavour_class'] == 2, 'bhad'] = 1
    data_out.loc[data_in['Jet_flavour_class'] == 3, 'gluon'] = 1

def main(argv):
    # start loading the data:
    TRAINING_DATASET = "/shome/phwindis/data/training_data.csv"
    VALIDATION_DATASET = "/shome/phwindis/data/validation_data.csv"

    print("loading training dataset from " + TRAINING_DATASET)
    raw_data = pd.read_csv(TRAINING_DATASET, names = ['Jet_CSV', 'Jet_CSVIVF', 'Jet_JP', 'Jet_JBP','Jet_cMVA', 'Jet_flavour_class'])

    print("loading validation dataset from " + VALIDATION_DATASET)
    raw_data_validation = pd.read_csv(VALIDATION_DATASET, names = ['Jet_CSV', 'Jet_CSVIVF', 'Jet_JP', 'Jet_JBP','Jet_cMVA', 'Jet_flavour_class'])

    print("processing data")
    # get only the feature columns
    training_in = raw_data.loc[:,['Jet_CSV', 'Jet_CSVIVF', 'Jet_JP', 'Jet_JBP']].copy()
    validation_in = raw_data_validation.loc[:,['Jet_CSV', 'Jet_CSVIVF', 'Jet_JP', 'Jet_JBP']].copy()

    cols = ['light', 'charm', 'bhad', 'gluon']
    training_out = pd.DataFrame(0, index = np.arange(0, len(raw_data.index)), columns = cols)
    validation_out = pd.DataFrame(0, index = np.arange(0, len(raw_data_validation.index)), columns = cols)
    
    makeOneHot(training_out, raw_data)
    makeOneHot(validation_out, raw_data_validation)

    print("building model")
    model = simple_classifier()
    early_stop = EarlyStopping(monitor = 'val_loss', patience = 10, verbose = 0, mode = 'auto')

    print("start training")
    model.fit(training_in.as_matrix(), training_out.as_matrix(), nb_epoch = 1000, batch_size = 10, validation_data=(validation_in.as_matrix(), validation_out.as_matrix()), callbacks = [early_stop])

    # save the trained model:
    MODEL_OUT = argv[0] + '/model.h5'
    print("saving fitted model to " + MODEL_OUT)
    model.save(MODEL_OUT)

if __name__ == "__main__":
    main(sys.argv[1:])




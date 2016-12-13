from __future__ import division
import sys
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main(argv):

    # load back the model, and the validation data
    VALIDATION_DATASET = "/shome/phwindis/data/validation_data.csv"

    print("loading validation dataset from " + VALIDATION_DATASET)
    raw_data_validation = pd.read_csv(VALIDATION_DATASET, names = ['Jet_CSV', 'Jet_CSVIVF', 'Jet_JP', 'Jet_JBP','Jet_cMVA', 'Jet_flavour_class'])
    validation_in = raw_data_validation.loc[:,['Jet_CSV', 'Jet_CSVIVF', 'Jet_JP', 'Jet_JBP']].copy()

    print("loading model from " + argv[0])
    model = load_model(argv[0])

    classifier_out = model.predict(validation_in.as_matrix(), batch_size = 32)

    raw_data_validation['NN_classifier'] = classifier_out[:,2]
    raw_data_validation['random_classifier'] = np.random.rand(len(raw_data_validation.index))

    jet_algorithms = ["Jet_CSV", "Jet_CSVIVF", "Jet_JP", "random_classifier","Jet_cMVA", "NN_classifier"]

    # plot the ROC curves
    fig = plt.figure(figsize=(10,6))

    b_jets = raw_data_validation.loc[raw_data_validation['Jet_flavour_class'] == 2]
    non_b_jets = raw_data_validation.loc[(raw_data_validation['Jet_flavour_class'] != 2)]

    for algorithm in jet_algorithms:
        efficiency = np.array([])
        misid_prob = np.array([])
    
        minval = np.min(raw_data_validation[algorithm])
        maxval = np.max(raw_data_validation[algorithm])
        for threshold in np.arange(minval, maxval, (maxval - minval) / 1000):    
            correct_b = b_jets.loc[b_jets[algorithm] >= threshold]
            misid_b = non_b_jets.loc[non_b_jets[algorithm] >= threshold]
        
            if(len(correct_b) > 1 and len(misid_b) > 1):
                efficiency = np.append(efficiency, len(correct_b) / len(b_jets))
                misid_prob = np.append(misid_prob, len(misid_b) / len(non_b_jets))


        plt.plot(efficiency, misid_prob, label = algorithm)
        plotdata = np.vstack([misid_prob, efficiency])
        path = argv[1] + algorithm + ".np"
        np.save(path, plotdata)

    plt.yscale('log')
    axes = plt.gca()
    axes.set_ylim([1e-2,1])
    plt.legend(loc = 'upper left')
    plt.xlabel('b jet efficiency')
    plt.ylabel('misidentification prob.')
    plt.show()
    fig.savefig(argv[1])

if __name__ == "__main__":
    main(sys.argv[1:])

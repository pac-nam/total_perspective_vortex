import numpy as np
import pickle
import time

from mne.decoding import CSP
from mne.io import concatenate_raws, read_raw_edf
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne import Epochs, pick_types, events_from_annotations

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, train_test_split

import sys

class vortex :
    def __init__(self, args):
        self.__args = args

    def __preprocessing(self):
        """
        Doc eegbci : https://mne.tools/dev/generated/mne.datasets.eegbci.load_data.html
        Imagerie intéressante :
            6, 10, 14 Motor imagery: hands vs feet
            5, 9, 13 Motor execution: hands vs feet
        """
        event_ids=dict(hands=2, feet=3)
        # subject = 1
        # runs = 10
        # i = 1
        raw_fnames = list()
        raw_fnames = eegbci.load_data(1,  [5, 6, 9, 10, 13, 14], path=self.__args.path)
        raw = concatenate_raws([read_raw_edf(f, preload=True, stim_channel='auto') for f in raw_fnames])
        eegbci.standardize(raw)  # set channel names
        montage = make_standard_montage('standard_1020')
        raw.set_montage(montage)
        raw.rename_channels(lambda x: x.strip('.'))
        events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3))
        picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude='bads')
        epochs = Epochs(raw, events, event_ids, tmin=-0.2, tmax=3.0, picks=picks, preload=True)
        if self.__args.visualize is True :
            raw.plot(block=True, scalings='auto', title='Before filter')
            raw.plot_psd()
        raw.filter(15., 30., method='fir')
        if self.__args.visualize is True :
            raw.plot(block=True, scalings='auto', title='After filter')
            raw.plot_psd()
        return(epochs) 

    def predict(self) :
        try :
            model = pickle.load(open(".model.pickle", "rb"))
        except FileNotFoundError :
            print("Run training please")
            return(-1)
        epochs = self.__preprocessing()
        epochs = epochs.copy().crop(tmin=1., tmax=2.)
        epochs_get_data = epochs.get_data()
        labels = epochs.events[:, -1] - 2
        epoch = 0
        correct = 0
        for feature, target in zip(epochs_get_data, labels):
            feature = feature.reshape(1, feature.shape[0], feature.shape[1])
            y_pred = model.predict(feature)
            print("\nepoch [", epoch,"] :")
            print("event predicted : [", y_pred[0], "]")
            print("true event      : [", target, "]")
            print("predict is", target==y_pred[0],"\n")
            if y_pred[0] == target :
                correct += 1
            epoch += 1
            # time.sleep(1)
        correct_percent = (correct / epoch) * 100
        print("Performance predict : {}%".format(round(correct_percent, 2)))

    def training(self) :
        epochs = self.__preprocessing()
        epochs = epochs.copy().crop(tmin=1., tmax=2.)
        epochs_get_data = epochs.get_data()
        labels = epochs.events[:, -1] - 2

        # Create split
        epochs_get_data_train, epochs_get_data_test, labels_train, labels_test = train_test_split(epochs_get_data, labels, test_size=0.33)

        lda = LinearDiscriminantAnalysis()
        csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

        pipeline = Pipeline(
            [
                ('csp', csp),
                ('model', lda)
            ])

        params_grid = {
            'model__solver' : ['svd', 'lsqr', 'eigen'],
            'model__n_components' : [None, 0, 1]
        }

        cross_validation=5
        grid = GridSearchCV(pipeline, param_grid=params_grid, cv=cross_validation, n_jobs=-1)
        
        grid.fit(epochs_get_data_train, labels_train)

        print("best score :" , grid.best_score_, ", best params", grid.best_params_)
        model = grid.best_estimator_
        print("test score :", model.score(epochs_get_data_test, labels_test))
        pickle.dump(model, open(".model.pickle", 'wb'))
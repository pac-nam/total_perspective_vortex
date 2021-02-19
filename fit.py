from mne.decoding import CSP
from mne.io import concatenate_raws, read_raw_edf
from scipy import fftpack
import numpy as np
import matplotlib.pyplot as plt
from mne.channels import make_standard_montage
from mne.datasets import eegbci

from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit, cross_val_score
from mne import Epochs, pick_types, events_from_annotations

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

class fitTotal :
    def __init__(self):
        self.__raw = self.__preprocessing()
        self.fourrier()
        self.csp = CSP()

    def __preprocessing(self):
        """
        Doc eegbci : https://mne.tools/dev/generated/mne.datasets.eegbci.load_data.html
        Imagerie intéressante :
            6, 10, 14 Motor imagery: hands vs feet
            5, 9, 13 Motor execution: hands vs feet
        """
        runs = 10
        i = 1
        raw_fnames = list()
        subjects = [1]
        raw_fnames = list()

        for subject in subjects :
            tmp_raw_fnames = eegbci.load_data(subject, [5,6, 9,10,13,14], path='/Users/ayguillo/../../sgoinfre/goinfre/Perso/ayguillo/')
            raw_fnames += tmp_raw_fnames
        raw = concatenate_raws([read_raw_edf(f, preload=True, stim_channel='auto') for f in raw_fnames])
        eegbci.standardize(raw) 
        montage = make_standard_montage('standard_1005')
        raw.set_montage(montage)
        raw.rename_channels(lambda x: x.strip('.'))
        # raw.plot(block=True, scalings='auto', title='Before filter')
        raw.filter(7., 30., method='iir')
        # raw.plot(block=True, scalings='auto', title='After filter')
        return(raw)

    def fit(self) :

        names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA", "LDA"]

        classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel="linear", C=0.025),
            SVC(gamma=2, C=1),
            GaussianProcessClassifier(1.0 * RBF(1.0)),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            MLPClassifier(alpha=1, max_iter=1000),
            AdaBoostClassifier(),
            GaussianNB(),
            QuadraticDiscriminantAnalysis(),
            LinearDiscriminantAnalysis()]

        event_id = dict(hands=2, feet=3)
        events, _ = events_from_annotations(self.__raw, event_id=dict(T1=2, T2=3))

        picks = pick_types(self.__raw.info, meg=False, eeg=True, stim=False, eog=False,
                        exclude='bads')
        epochs = Epochs(self.__raw, events, event_id, -1, 4, proj=True, picks=picks,
                baseline=None, preload=True)
        epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
        labels = epochs.events[:, -1] - 2
        scores = []
        epochs_data = epochs.get_data()
        epochs_data_train = epochs_train.get_data()
        i = 0
        for classifier in classifiers :
            cv = ShuffleSplit(10, test_size=0.2, random_state=42)
            cv_split = cv.split(epochs_data_train)

            # Assemble a classifier
            # lda = LinearDiscriminantAnalysis()
            csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

            # Use scikit-learn Pipeline with cross_val_score function
            clf = Pipeline([('CSP', csp),('model',  classifier)])
            scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=-1)

            class_balance = np.mean(labels == labels[0])
            class_balance = max(class_balance, 1. - class_balance)
            print("Classification %s accuracy: %f / Chance level: %f" % (names[i], np.mean(scores),
                                                                    class_balance))
            i += 1

        # plot CSP patterns estimated on full data for visualization
        # csp.fit_transform(epochs_data, labels)

        # csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)
        # plt.show()


    def __channels(self, ann, data):
        start, end = int(ann["onset"] * 160), int((ann["onset"] + ann["duration"]) * 160)
        res = []
        for i in range(64):
            res.append(data[i][start:end])
        return res
    

    def __sort_data(self, raw, data):
        res = [[],[],[]]
        dictionnaire = {"T0": 0, "T1": 1, "T2": 2}
        for ann in raw.annotations:
            res[dictionnaire[ann["description"]]].append(self.__channels(ann, data))
        return res

    def fourrier(self, plot = True) :
        """
        Args :
            data : np.array to get_data method
            plot : default False. Plot reverse fourrier if True
        Return :
            fourrier : np.array. Result to fourrier method with real numbers.
            fourrier_sort : np.array. Result to fourrier method with real numbers, with reverse sort.
        """
        get_data = self.__raw.get_data()
        # res_data = self.__sort_data(self.__raw, get_data)
        data = get_data[0]
        fourrier = fftpack.fft(data)
        abs_fourrier = np.abs(fourrier)
        freq = fftpack.fftfreq(len(data))
        fourrier_sort = np.sort(fourrier.real)
        val_filter = fourrier_sort[::-1][10]
        print(val_filter)
        fourrier[abs_fourrier < val_filter] = 0
        if plot is True :
            ifourrier = fftpack.ifft(fourrier)
            # plt.plot(data)
            # self.__raw.plot(block=True, scalings='auto')
            plt.plot(ifourrier.real)
            plt.show()
        return(fourrier.real, fourrier_sort[::-1])

# fourrier = fourrier()
# pipe = Pipeline([('CSP', csp), ('model', classification())])
# pipe.fit(X, y)

# pickle(pipe)

# pipe.predict(X)

if __name__ == '__main__':
    fit = fitTotal()
    fit.fit()
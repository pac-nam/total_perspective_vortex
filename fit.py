from scipy import fftpack
import numpy as np
import matplotlib.pyplot as plt

from mne.decoding import CSP
from mne.io import concatenate_raws, read_raw_edf
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne import Epochs, pick_types, events_from_annotations

import sys

class fitTotal :
    def __init__(self):
        self.__raw = self.__filter()
        self.csp = CSP()

    def __filter(self):
        """
        Doc eegbci : https://mne.tools/dev/generated/mne.datasets.eegbci.load_data.html
        Imagerie intéressante :
            6, 10, 14 Motor imagery: hands vs feet
            5, 9, 13 Motor execution: hands vs feet
        """
        event_ids=dict(hands=2, feet=3)
        subject = 1
        runs = 10
        i = 1
        raw_fnames = list()
        # for i in range(1,11) :
        raw_fnames = eegbci.load_data(1, [5,6,9,10,13,14])
        raw = concatenate_raws([read_raw_edf(f, preload=True, stim_channel='auto') for f in raw_fnames])
        eegbci.standardize(raw)  # set channel names
        montage = make_standard_montage('standard_1020')
        raw.set_montage(montage)
        raw.rename_channels(lambda x: x.strip('.'))
        events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3))
        picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude='bads')
        print(events)
        # raw.plot(block=True, scalings='auto', title='Before filter')
        # raw.plot_psd()
        # raw.filter(8., 20., method='fir')
        # raw.plot(block=True, scalings='auto', title='After filter')
        # raw.plot_psd()
        return(raw)

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

if __name__ == '__main__':
    fit = fitTotal()
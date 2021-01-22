from scipy import fftpack
import sys
from mne.io import read_raw_edf
import numpy as np
import matplotlib.pyplot as plt

def channels(ann, data):
    start, end = int(ann["onset"] * 160), int((ann["onset"] + ann["duration"]) * 160)
    res = []
    for i in range(64):
        res.append(data[i][start:end])
    return res

def sort_data(raw, data):
    res = [[],[],[]]
    dictionnaire = {"T0": 0, "T1": 1, "T2": 2}
    for ann in raw.annotations:
        res[dictionnaire[ann["description"]]].append(channels(ann, data))
    return res

def fourrier(data, plot = False) :
    """
    Args :
        data : np.array to get_data method
        plot : default False. Plot reverse fourrier if True
    Return :
        fourrier : np.array. Result to fourrier method with real numbers.
        fourrier_sort : np.array. Result to fourrier method with real numbers, with reverse sort.
    """
    fourrier = fftpack.fft(data)
    abs_fourrier = np.abs(fourrier)
    freq = fftpack.fftfreq(len(data))
    fourrier_sort = np.sort(fourrier.real)
    val_filter = fourrier_sort[::-1][4]
    fourrier[abs_fourrier < val_filter] = 0
    ifourrier = fftpack.ifft(fourrier)
    if plot is True :
        plt.plot(data)
        plt.plot(ifourrier.real)
        plt.show()
    return(fourrier.real, fourrier_sort[::-1])

if __name__ == '__main__' :
    if len(sys.argv) > 1 :
        raw = read_raw_edf(sys.argv[1], preload=True, stim_channel='auto', verbose=False)
        get_data = raw.get_data()
        res_data = sort_data(raw, get_data)
        data = res_data[1][2][4]
        s_fourrier, fourrier_sort = fourrier(data, True)
        for iter in range(len(res_data[1][2])) :
            data = res_data[1][2][iter]
            ediff = np.ediff1d(data)
            abs_ediff = np.abs(ediff)
            mean = np.mean(abs_ediff)
            if mean > 3e-5 :
                # C'est du bruit. Ne pas envoyer à l'algo.
                plt.plot(data)
                plt.show()
    else :
        print("Argument not found")
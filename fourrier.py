from scipy import fftpack
import sys
from mne.io import read_raw_edf
import numpy as np
import matplotlib.pyplot as plt
def channels(ann, data):
    start, end = int(ann["onset"] * 160), int((ann["onset"] + ann["duration"]) * 160)
    res = []
    for i in range(64):
        # print(len(data), len(res))
        res.append(data[i][start:end])
    return res

def sort_data(raw, data):
    res = [[],[],[]]
    dictionnaire = {"T0": 0, "T1": 1, "T2": 2}
    for ann in raw.annotations:
        res[dictionnaire[ann["description"]]].append(channels(ann, data))
    return res

if __name__ == '__main__' :
    if len(sys.argv) > 1 :
        raw = read_raw_edf(sys.argv[1], preload=True, stim_channel='auto', verbose=False)
        get_data = raw.get_data()
        res_data = sort_data(raw, get_data)
        data = res_data[1][5][6]
        fourrier = fftpack.fft(data)
        abs_fourrier = np.abs(fourrier)
        freq = fftpack.fftfreq(len(data))
        fourrier_sort = np.sort(fourrier.real)
        val_filter = fourrier_sort[::-1][2]
        fourrier[abs_fourrier < val_filter] = 0
        ifourrier = fftpack.ifft(fourrier)
        plt.plot(data)
        plt.plot(ifourrier.real)
        plt.show()
    else :
        print("Argument not found")
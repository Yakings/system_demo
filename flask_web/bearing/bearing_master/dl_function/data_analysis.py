import numpy as np



def fft(x_vibration):
    x_freq = np.fft.fft(x_vibration)
    x_freqabs = abs(x_freq)
    return x_freqabs

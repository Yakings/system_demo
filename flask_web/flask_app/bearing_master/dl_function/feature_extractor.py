import numpy as np
import math
def get_mean(signal):
    return np.mean(signal)
def get_var(signal):
    return np.var(signal)

def get_abs_mean(signal):
    return np.mean(np.fabs(signal))
def get_max(signal):
    return np.max(signal)
def get_min(signal):
    return np.min(signal)
def get_qiaodu(signal):
    signal = np.array(signal)
    smean = np.mean(signal)

    hold = signal-smean
    hold = np.square(hold)
    hold = np.square(hold)
    numerator = np.mean(hold)

    denominator =np.square(np.mean(np.square(signal-smean)))
    return float(numerator/denominator)
if __name__ == '__main__':
    get_qiaodu([1,13.1,2,2,1,1,5,6,7])
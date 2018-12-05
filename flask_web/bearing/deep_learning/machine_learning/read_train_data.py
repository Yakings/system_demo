import os
import numpy as np
def get_all_file(filepath):
    files = os.listdir(filepath)
    file_list = []
    for fi in files:
        fi_d = os.path.join(filepath, fi)
        if os.path.isdir(fi_d):
            get_all_file(fi_d)
        elif 'acc_' in fi_d:
            # file_list.append(os.path.join(filepath, fi_d))
            file_list.append(fi_d)
    file_list.sort()
    # print(file_list)
    return file_list
    pass
def add_label(file_list):
    max_val = len(file_list)
    # print(max_val)
    y = [i for i in range(max_val-1, -1, -1)]
    return y
def get_one_bearing(path):
    file_list = get_all_file(path)
    y = add_label(file_list)
    # print(np.array(file_list))
    # print(np.array(y))
    # return np.array(file_list),np.array(y)
    return file_list,y

def get_all_bearings_list(path_list):
    x_all = []
    y_all = []
    for i in range(len(path_list)):
        path = path_list[i]
        files = os.listdir(path)
        for fi in files:
            fi_d = os.path.join(path, fi)
            if os.path.isdir(fi_d):
                # one_bearing_file = os.path.join(fi_d, fi_d)
                one_bearing_file = fi_d
                one_bearig_x, one_bearing_y = get_one_bearing(one_bearing_file)
                x_all.append(one_bearig_x),
                y_all.append(one_bearing_y)
    try:
        x_all = np.concatenate(x_all)
        y_all = np.concatenate(y_all)
    except:
        print('just one bearing!')

    x_all = np.reshape(x_all,(-1,))
    # y_all = y_all.reshape(-1)
    y_all = np.reshape(y_all,(-1))
    # print(x_all,y_all)
    return x_all,y_all
def get_all_bearings(path):
    files = os.listdir(path)
    file_list = []
    x_all = []
    y_all = []
    for fi in files:
        fi_d = os.path.join(path, fi)
        if os.path.isdir(fi_d):
            one_bearing_file = os.path.join(fi_d, fi_d)
            one_bearig_x, one_bearing_y = get_one_bearing(one_bearing_file)
            x_all.append(one_bearig_x),
            y_all.append(one_bearing_y)

    x_all = np.concatenate(x_all)
    x_all = np.reshape(x_all,(-1,))
    y_all = np.concatenate(y_all)
    y_all = y_all.reshape(-1)
    # print(x_all,y_all)
    return x_all,y_all


VIB_SIZE = 2000
step = 2
IMAGE_SIZE = VIB_SIZE//step

import csv
import random

#fft
def fft(x_vibration):
    x_freq = np.fft.fft(x_vibration)
    x_freqabs = abs(x_freq)
    return x_freqabs
# 高斯噪声
def wgn(length, mu=0, sigma=0.1):
    noise = [random.gauss(mu, sigma) for i in range(length)]
    return np.array(noise)
def read_from_string_func(name,input_dim):
    # 旋转角度范围
    name = np.reshape(name, [1,])
    # print('name',name)
    csv_reader = csv.reader(open(name[0]))
    vib = []
    for row in csv_reader:
        vib.append(float(row[4]))
    # print('vib:',vib)
    max_size = len(vib)
    set_max_size = (max_size//input_dim)*input_dim
    div_size = max_size-set_max_size
    first = random.randint(0,div_size)
    vib = np.array(vib)
    vib = vib[first:first+set_max_size]

    noise = wgn(set_max_size)
    vib += noise
    # vib = vib.reshape([VIB_SIZE//step,step])
    # vib = np.max(vib,axis=1)

    freq = fft(vib)
    step = set_max_size//input_dim
    freq = freq.reshape([set_max_size//step,step])
    freq = np.max(freq,axis=1)

    # freq = np.expand_dims(freq, axis=1)
    # print(vib.dtype.name)
    freq = freq.astype(np.float32)
    # print(vib.dtype.name)
    return freq

if __name__=="__main__":
    # path = r'G:\Bearing\bearing\data_set\FEMTOBearingDataSet\Training_set\Learning_set'
    path = r'/home/sunyaqiang/Myfile/bearing/data_set/FEMTOBearingDataSet/Test_set/Test_set'
    x_all, y_all = get_all_bearings(path)
    print(x_all,y_all)
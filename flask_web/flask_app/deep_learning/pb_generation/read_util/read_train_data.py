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
            file_list.append(os.path.join(filepath, fi_d))
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

if __name__=="__main__":
    # path = r'G:\Bearing\bearing\data_set\FEMTOBearingDataSet\Training_set\Learning_set'
    path = r'/home/sunyaqiang/Myfile/bearing/data_set/FEMTOBearingDataSet/Test_set/Test_set'
    x_all, y_all = get_all_bearings(path)
    print(x_all,y_all)
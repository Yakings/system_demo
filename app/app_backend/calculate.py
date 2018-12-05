import numpy as np


from app.app_backend.fileRead import get_file, get_file_list


# 判断轴承是否更换为新的ß
def judge_new(previous_rul, predict_rul):
    '''
    judge whether put a new bearing.
    :return:
    '''

    err = predict_rul - previous_rul
    if err > 100:
        return 1
    else:
        return 0
def get_num(path):
    filename_list = get_file_list(path)

def cnn(x):
    return x
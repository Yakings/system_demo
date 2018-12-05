from .ml_models import NuSVR_train,SVM_train,LinearRegression_train,\
    DecisionTreeClassifier_train,KNeighborsRegressor_train,\
    save_model,resotre_model,inference
from .read_train_data import get_all_bearings_list,read_from_string_func
import numpy as np
import os

def train_machine_learning(todolist,DATA_NAME_LIST,trained_model_path,train_setting_list):
    filename_list, label_list = get_all_bearings_list(DATA_NAME_LIST)
    signal_feature_list = []
    input_dim = train_setting_list.input_dim
    for i in range(len(filename_list)):
        print('reading:',filename_list[i])
        signal_feature_list.append(read_from_string_func(filename_list[i],input_dim))
    x = np.array(signal_feature_list)
    y = label_list
    select_flag=train_setting_list.ml_model_class
    print('begin trian machine learning model')
    clf = 0
    if select_flag==0:
        clf = NuSVR_train(x,y)
    elif select_flag == 1:
        clf = SVM_train(x,y)
    elif select_flag == 2:
        clf = LinearRegression_train(x,y)
    elif select_flag == 3:
        clf = DecisionTreeClassifier_train(x,y)
    elif select_flag == 4:
        clf = KNeighborsRegressor_train(x,y)
    print('trian machine learning model finished!')

    save_model_func(clf, file_path=trained_model_path,input_dim=input_dim)
    pass
def save_model_func(clf, file_path,input_dim):
    file_path = os.path.join(file_path,'output')
    print(file_path)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    save_model(clf, file_path,name='mist_binary_clas_ml__DM%sDM_quaintized'%(input_dim))
    print('save machine learning model finished!')
def restore_model_func(path, name):
    resotre_model(path, name)
def inference_func(x,clf):
    inference(x,clf)

import os
import csv
import numpy as np


import json

def get_str_csv(file_name):
    try:
        with open(file_name) as f:
            reader = csv.reader(f)

            column = [row[0] for row in reader]

    except:
        column = []
    return column
    pass
def get_csv(file_name):
    with open(file_name) as f:
        reader = csv.reader(f)

        column = [float(row[0]) for row in reader]
        return column
    pass
def get_signal_csv(file_name):
    # try:
    #     with open(file_name,'r') as f:
    #         reader = csv.reader(f)
    #         column = [float(row[0]) for row in reader]
    #         # column = []
    #         # for row in reader:
    #         #     try:
    #         #         column.append(float(row[0]))
    #         #     except:
    #         #         pass
    # except:
    #     print('unable to read!')
    #     column = []
    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            reader = csv.reader(f)
            column = [float(row[0]) for row in reader]
    else:
        column = []
    return column
    pass
def get_addition_signal_csv(file_name,y):
    try:
        with open(file_name,'r') as f:
            reader = list(csv.reader(f))[len(y):]
            column = [float(row[0]) for row in reader]
            y += column
            # column = []
            # for row in reader:
            #     try:
            #         column.append(float(row[0]))
            #     except:
            #         pass
    except:
        print('unable to read!')
        column = []
    # with open(file_name, 'r') as f:
    #     reader = csv.reader(f)
    #     column = [float(row[0]) for row in reader]
    return column
    pass
def read_json(json_file_name):
    with open(json_file_name, 'r') as load_f:
        load_dict = json.load(load_f)
        # print(load_dict)
        return load_dict

def get_now_user():
    user_now = read_json('./data/user_info/user_now.json')
    user_now = user_now[0]
    # print(user_now)
    return user_now

def creat_dir(path):
    exit = os.path.exists(path)
    if not exit:
        os.makedirs(path)
        return 1
    else:
        return 0



##########################################


def read_selected_task(name):
    if os.path.exists(name):
        selected_task_list = read_json(name)
        try:
            selected_task_list.remove('')
        except:
            pass
            # print('remove finished!')
    else:
        selected_task_list = []
    # print('selected_task_list',selected_task_list)
    return selected_task_list
    pass


def get_file_list_local(path='app/data/model/'):
    filename = os.listdir(path)
    try:
        filename.remove('select_setting.json')
    except:
        pass
        # print('remove finished!')
    try:
        filename.remove('view_setting.json')
    except:
        pass
        # print('remove finished!')
    try:
        filename.remove('model_setting.json')
    except:
        pass
        # print('remove finished!')

    # print(filename)
    return filename
    pass



def get_file(file_name):
    with open(file_name) as f:
            reader = csv.reader(f)
            loss = [row[0] for row in reader]
            loss = np.array(loss, np.float)

    pass


def get_signal(file_name):
    with open(file_name) as f:
            reader = csv.reader(f)
            loss = [row[4] for row in reader]
            loss = np.array(loss, np.float)

    pass


if __name__=='__main__':
    # get_file_list('../data/model/')
    read_json('../data/user_info/user_now.json')
    get_now_user()


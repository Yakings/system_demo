import os
import csv
import numpy as np
def get_signal_csv(file_name):
    with open(file_name) as f:
        reader = csv.reader(f)

        column = [float(row[4]) for row in reader]
        return column
    pass
def get_all_csv(path):
    files = os.listdir(path)
    if len(files) < 10:
        new_path = os.path.join(path, files[0])
        if os.path.isdir(new_path):
            path = new_path

    files = os.listdir(path)
    all_datas = []
    for file in files:
        all_name = os.path.join(path, file)
        if not os.path.isdir(all_name):
            data = get_signal_csv(all_name)
            # all_datas.append(data)
            all_datas += data
    return all_datas

def get_csv_path(path):
    files = os.listdir(path)
    if len(files) < 10:
        for file in files:
            new_path = os.path.join(path, file)
            if os.path.isdir(new_path):
                path = new_path
                break
    return path

def get_loss_csv(file_name):
    with open(file_name) as f:
            reader = csv.reader(f)
            loss = [row[0:2] for row in reader]
            loss = np.array(loss, np.float)
            if len(loss) == 0:
                trainloss = []
                testloss = []
            else:
                trainloss = loss[:,0].tolist()
                testloss = loss[:,1].tolist()
            return [trainloss,testloss]
    pass


# def gci(filepath):
#遍历filepath下所有文件，包括子目录
  # files = os.listdir(filepath)
  # for fi in files:
  #   fi_d = os.path.join(filepath,fi)
  #   if os.path.isdir(fi_d):
  #     gci(fi_d)
  #   else:
  #     print(os.path.join(filepath,fi_d))


if __name__=="__main__":
    get_signal_csv('../data/acc_00001.csv')
    # get_all_csv('../data/aaaa/Bearing1_1/')
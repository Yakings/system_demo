import json
import os
def modifi_user(new_name='admin'):
    name = './data/user_info/user_now.json'
    write_json([new_name], name)
    pass

def modifi_select_set(new_name_list=[''],path=''):
    name = path
    if len(new_name_list) == 0:
        new_name_list=['']
    write_json(new_name_list, name)
    pass
def modifi_view_set(new_name_list=[''], path=''):
    name = path
    if len(new_name_list) == 0:
        new_name_list=['']
    write_json(new_name_list, name)
    pass

def write_json(new_thing,json_file_name):
    # print(os.getcwd())
    with open(json_file_name, "w") as f:
        json.dump(new_thing, f)
    # print("写入文件完成...")
import csv
def write_add_csv(filename, contex = [0.1, 0.09]):
    with open(filename, "a+",newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(contex)
        # print('log loss')
    pass
if __name__=='__main__':
    write_json(['admin'],'../data/user_info/user_now.json')
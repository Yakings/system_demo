import requests

def url_get_string(url = 'http://192.168.0.100:5000/return_list'):
    import requests
    r = requests.post(url)
    # print(r.text)
    # print(0)


def get_file_list_from_net(url ='http://0.0.0.0:5000/train_task/admin'):
    r = requests.post(url)
    get_str = r.text
    # print(get_str)
    name_num = get_str.split('@@')
    names = list(filter(None,name_num[0].split('///')))
    nums_str = list(filter(None,name_num[1].split('///')))

    name_num_list = [names[i]+'--'+nums_str[i] for i in range(len(names))]
    # print(names,nums_str)
    return name_num_list
    pass


# 上传文件
def url_send_file(file_name,url='http://192.168.0.100:5000/edge_uploading/0'):
    # with open(file_name, 'rb') as f:
    #     data = f.read()
    files = {'image': (file_name, open(file_name, 'rb'))}

    resp = requests.post(url, data=None,files=files)  # post提交数据
    return resp

    pass

# 下载文件
# def url_download(url = 'http://0.0.0.0:5000/download/loss.csv'):
#     from urllib.request import urlopen
#     file_bytes = urlopen(url).read()
#     with open('./aaaa.csv','wb') as f:
#         f.write(file_bytes)
# 下载文件
from urllib.request import urlopen
def url_download(url = 'http://192.168.0.100:5000/download/loss.csv',dist_path='./aaa.csv'):
    # import io
    # import csv
    # print('url:',url)
    file_bytes = urlopen(url).read()
    with open(dist_path,'wb') as f:
        f.write(file_bytes)
def url_get_input_dim(url='http://192.168.0.100:5000/inputdim/admin/1539683968'):
    ile_int = int(urlopen(url).read())
    # print(ile_int)
    return ile_int
    pass
if __name__=="__main__":
    # url_get_string()
    a = url_send_file('./center.py')
    # print(a.text)
    # url_download()
    # get_file_list()
    pass
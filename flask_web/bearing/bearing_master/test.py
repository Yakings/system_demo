
import numpy as np

def arr_to_str():
    a =  '/1535197933/1535288283'
    b = a.split('/')

    b = list(filter(None,b))
    print(b)


def get_gpu():
    import pynvml
    pynvml.nvmlInit()
    # 这里的0是GPU id
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(meminfo.used)
    pass


def get_cpu_core_number():
    from multiprocessing import cpu_count
    print(cpu_count())





import psutil
def get_cpu_memo():
# 定义一个进程列表
    process_lst = []

    def getProcess(pName):
        # 获取当前系统所有进程id列表
        all_pids  = psutil.pids()

        # 遍历所有进程，名称匹配的加入process_lst
        for pid in all_pids:
            p = psutil.Process(pid)
            if (p.name() == pName):
                process_lst.append(p)

        return process_lst

    # 获取进程名位Python的进程对象列表
    process_lst = getProcess("Python")

    # 获取内存利用率：
    for process_instance in process_lst:
        print(process_instance.memory_percent())

    # 获取cpu利用率：
    for process_instance in process_lst:
        process_instance.cpu_percent(None)

    for process_instance in process_lst:
        print(process_instance.cpu_percent(None))


    # 系统的内存利用率
    print(psutil.virtual_memory().percent)

    # 系统的CPU利用率
    psutil.cpu_percent(None)

    print(psutil.cpu_percent(None))



import psutil
import os,datetime,time

def getMemCpu():
    data = psutil.virtual_memory()
    total = data.total #总内存,单位为byte
    free = data.available #可以内存
    memory =  "Memory usage:%d"%(int(round(data.percent)))+"%"+"  "
    cpu = "CPU:%0.2f"%psutil.cpu_percent(interval=1)+"%"
    return memory+cpu

def main():
    while(True):
        info = getMemCpu()
        time.sleep(0.2)
        print(info)


def get_Memo_rate_mome():
    data = psutil.virtual_memory()
    total = data.total  # 总内存,单位为byte
    free = data.available  # 可以内存
    memory_rate = int(round(data.percent))
    return  memory_rate, total
def cpu_core_rate_num():
    from multiprocessing import cpu_count
    num_core = cpu_count()
    cpu_rate = psutil.cpu_percent(interval=1)
    return cpu_rate, num_core


def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    gpus_list =  [x.name for x in local_device_protos if x.device_type == 'GPU']
    return len(gpus_list)



def url_download():
    url = 'http://0.0.0.0:5000/download/loss.csv'
    from urllib.request import urlopen
    import io

    import csv

    file_bytes = urlopen(url).read()
    with open('./aaaa.csv','wb') as f:
        f.write(file_bytes)
    # data_stream = io.BytesIO(file_bytes)
    # f = csv.reader(data_stream)
    # for i in f:
    #     a = i

def url_get_string():
    import requests
    url = 'http://0.0.0.0:5000/return_list'

    r = requests.post(url)
    print(r.text)
    print(0)



if __name__=="__main__":
    # arr_to_str()
    # get_gpu()
    # get_cpu_core_number()
    # get_cpu_memo()
    # main()
    # print(cpu_core_rate_num())
    # print(get_available_gpus())
    url_get_string()
    # url_download()
    pass

import psutil

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
    from tensorflow.python.client import device_lib as device_lib
    # from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    print(local_device_protos)
    gpus_list =  [x.name for x in local_device_protos if x.device_type == 'GPU']
    print('gpu list:',gpus_list)
    return len(gpus_list)


def get_gpu_used(i=0):
    print('gpu memoinfo')
    import pynvml
    pynvml.nvmlInit()
    # 这里的0是GPU id
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print('gpu memoinfo',meminfo.used)
    return meminfo.used,meminfo.total
    pass
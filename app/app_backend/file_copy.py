import os,shutil
def copyfile_func(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        pass
        # print("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.copyfile(srcfile,dstfile)      #复制文件
        # print("copy %s -> %s"%( srcfile,dstfile))
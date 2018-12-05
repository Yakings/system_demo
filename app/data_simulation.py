#!/usr/bin/env python
# coding:utf-8
import sys
try:
    # Python2
    import Tkinter as tk
    from Tkinter import Menu
    import tkMessageBox as messagebox
    import tkFileDialog as filedialog
except ImportError:
    # Python3
    import tkinter as tk
    from tkinter import *
    import tkinter.messagebox as messagebox
    import tkinter.filedialog as filedialog
    from tkinter.filedialog import askdirectory
from PIL import Image, ImageTk

import threading,cv2

from tkinter import ttk

# import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os,shutil
import time

from app_backend.fileRead import get_file_list_local, get_now_user, \
    creat_dir, read_selected_task,get_signal_csv,get_csv,get_str_csv,read_json,get_addition_signal_csv
from app_backend.fileWriter import modifi_user, modifi_select_set, \
    modifi_view_set, write_json,write_add_csv
from app_backend.net_search import url_get_string,url_send_file
from models_tools.inference_pb import predict_rul_pb,ClassifierGraph
from models_tools.inference_ml import ClassifierMachine
from app_backend.file_copy import copyfile_func
from queue import Queue  # Queue在3.x中改成了queue
from numpy import arange, sin, pi
import os
t = arange(0.0, 3000, 0.01)
s = sin(2 * pi * t)


SIGNAL_PATH = './data/signal/'
# SIGNAL_PATH = '.\\data\signal\\'

# 从Frame派生一个Application类，这是所有Widget的父容器
class Application():
    def __init__(self, master=None):
        self.master = master
        # 设置窗口标题:
        self.master.title('数据模拟系统')
        #设置窗口尺寸：‘宽*长+ 横坐标+纵坐标’
        self.master.geometry('1000x400+500+5')
        self.creatMenu()
        self.createWidgets()
        # self.listbox(root=master)
        self.user_now = ''
        self.running = False
        self.predict_file_queue_list = []


        self.data_origin_path=''
        self.data_dest_path=''
        self.select_origin_path()
        self.select_dest_path()

        self.feeding = 0

        self.start_Button = Button(self.master, text='开始', command=self.begin_copy)
        self.start_Button.grid(row=6, column=0)
        self.stop_Button = Button(self.master, text='停止', command=self.stop_copy)
        self.stop_Button.grid(row=6, column=1)

    def listbox(self,root):
        mylist = tk.Listbox(root,width=50)  # 列表框
        mylist.pack()

        for item in ["1", '1','1','1','1','1','1','1','1',"asdsa", "asdsadsa", "asdsadsad"]:  # 插入内容
            mylist.insert(tk.END, item)  # 从尾部插入

    ###########################################



    def creatTitle(self, name,position=[1,1]):
        # self.plottitle = Label(self.master,text='A1 Label',background='red', foreground='red',bg='brown')
        self.plottitle = Label(self.master,text=name,font=1)
        self.plottitle.grid(row=position[0], column=position[1],rowspan=2,columnspan=2)
        pass
    def creatMenu(self):
        self.menubar = Menu(self.master)

        self.filemenu = Menu(self.menubar, tearoff=False)
        # self.filemenu.add_command(label="打开", command=self.openpicture, accelerator='Ctrl+N')
        self.filemenu.add_separator()#添加分割线
        #添加多级菜单
        self.menubar.add_cascade(label="文件", menu=self.filemenu)
        # self.menubar.add_cascade(label="文件", menu=self.filemenu2)

        #将菜单添加到整个窗口
        self.master.config(menu=self.menubar)

        # self.master.bind_all("<Control-n>", lambda event: print '加速键Ctrl+N')

    def createWidgets(self):
        self.running_strvar = StringVar()
        self.running_strvar.set('诊断任务已停止')
        self.show_running = Label(self.master, textvariable=self.running_strvar, font=1)
        self.show_running.grid(row=10, column=2,columnspan=1)

    def select_origin_path(self):
        path_var = StringVar()
        def selected_path():
            origin_path = askdirectory()
            path_var.set(origin_path)
        def modify_task():
            self.data_origin_path = path_var.get()
            self.origin_path_strvar.set(self.data_origin_path)
            pass
        Label(self.master, text='原始路径').grid(row=3, column=0)
        Entry(self.master, textvariable=path_var, ).grid(row=3, column=1)
        Button(self.master, text='路径选择', command=selected_path).grid(row=3, column=2)
        search_Button = Button(self.master, text='确认', command=lambda: modify_task())
        search_Button.grid(row=3, column=4)

        self.origin_path_strvar = StringVar()
        self.origin_path_strvar.set('')
        self.show_origin_path = Label(self.master, textvariable=self.origin_path_strvar, font=1)
        self.show_origin_path.grid(row=3, column=6,columnspan=1)
        pass
    def select_dest_path(self):

        path_var = StringVar()
        def selected_path():
            origin_path = askdirectory()
            path_var.set(origin_path)
        def modify_task():
            self.data_dest_path = path_var.get()
            self.dest_path_strvar.set(self.data_dest_path)

            pass
        Label(self.master, text='目标路径').grid(row=4, column=0)
        Entry(self.master, textvariable=path_var, ).grid(row=4, column=1)
        Button(self.master, text='路径选择', command=selected_path).grid(row=4, column=2)
        search_Button = Button(self.master, text='确认', command=lambda: modify_task())
        search_Button.grid(row=4, column=4)
        self.dest_path_strvar = StringVar()
        self.dest_path_strvar.set('')
        self.show_dest_path = Label(self.master, textvariable=self.dest_path_strvar, font=1)
        self.show_dest_path.grid(row=4, column=6,columnspan=1)
        pass


    def begin_copy(self):
        print('begin feed!')
        self.feeding = 1
        t = threading.Thread(target=self.copy_threading)
        t.setDaemon(True)  # 将线程声明为守护线程，必须在start() 方法调用之前设置，如果不设置为守护线程程序会被无限挂起
        t.start()  # 启动子线程

    def copy_threading(self):
        while self.feeding:
            self.copy_func()
            time.sleep(1)
            print('feeding....')

    def copy_func(self):
        origin_file_list = sorted(os.listdir(self.data_origin_path))
        dest_file_list = sorted(os.listdir(self.data_dest_path))

        file_name = origin_file_list[len(dest_file_list)]
        file_name_full_path = os.path.join(self.data_origin_path,file_name)
        file_name_full_dest_path = os.path.join(self.data_dest_path,file_name)
        print('copying:',file_name_full_path,file_name_full_dest_path)
        shutil.copyfile(file_name_full_path, file_name_full_dest_path)


    def stop_copy(self):
        print('stop feed!')
        self.feeding = 0

if __name__ =='__main__':
    root = Tk() # 初始化Tk()

    app1 = Application(root)

    while True:
        try:
            root.mainloop()
            break
        except UnicodeDecodeError:
            pass


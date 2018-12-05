#!/usr/bin/env python
# coding:utf-8
import sys
# print(sys.path)
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
from sklearn import linear_model
from tkinter import ttk

# import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os
import time
# try:

    # from app.app_backend.net_search import get_file_list_from_net,url_download,url_get_input_dim
    # from app.app_backend.fileRead import get_file_list_local,get_now_user,\
    #     creat_dir, read_selected_task,get_signal_csv,get_csv,get_str_csv,read_json,get_addition_signal_csv
    # from app.app_backend.fileWriter import modifi_user,modifi_select_set,\
    #     modifi_view_set, write_json,write_add_csv
    # from app.app_backend.net_search import url_get_string,url_send_file
    # from app.app_backend.file_copy import copyfile_func

# except:
from app_backend.net_search import get_file_list_from_net, url_download,url_get_input_dim
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

URL='http://192.168.0.177:5000'


SIGNAL_PATH = './data/signal/'
# SIGNAL_PATH = '.\\data\signal\\'



class Canvases:
    def __init__(self, master=None,window_canvas=None):
        color_val = 240.0/255.0
        self.f = Figure(facecolor=[color_val,color_val,color_val],figsize=(7, 4), dpi=100)

        self.canvas = FigureCanvasTkAgg(self.f, master=root)
        self.canvas.show()
        # self.canvas.get_tk_widget().pack()
        window_canvas.create_window(500, 429,window=self.canvas.get_tk_widget())


        # 清空图像，以使得前后两次绘制的图像不会重叠
        self.f.clf()
        self.a = self.f.add_subplot(111)

        # 在[0,100]范围内随机生成sampleCount个数据点
        # x = []
        # y = []
        # color = ['b', 'r', 'y', 'g']
        # 绘制这些随机点的散点图，颜色随机选取
        # self.a.scatter(x, y,  color=color[np.random.randint(len(color))])
        # self.a.set_title('Prediction Line')
        self.canvas.show()


        pass



# 从Frame派生一个Application类，这是所有Widget的父容器
class Application():
    def __init__(self, master=None):

        self.master = master
        self.master.title('边缘诊断系统')
        self.window_canvas = tk.Canvas(self.master, width=1180, height=650)
        self.window_canvas.pack()
        self.creat_canvas()
        self.canvases = Canvases(master,self.window_canvas)

        self.user_now = ''
        self.user_login()
        self.multi_task_show()
        self.running = False
        self.predict_file_queue_list = []
        self.auto_updating = 0

        self.createWidgets()

        self.creatTitle('         任务列表      ',position=[90, 240])
        self.creatTitle('预测曲线', position=[250, 240])

        self.bar_chart_onece()

        self.creatMenu()

        # #设置窗口尺寸：‘宽*长+ 横坐标+纵坐标’
        # self.master.geometry('1000x600+500+5')
        self.threading_draw()


    def listbox(self,root):
        mylist = tk.Listbox(root,width=50)  # 列表框
        mylist.pack()

        for item in ["1", '1','1','1','1','1','1','1','1',"asdsa", "asdsadsa", "asdsadsad"]:  # 插入内容
            mylist.insert(tk.END, item)  # 从尾部插入

    ###########################################

    def matplotlib_thread(self,root):
        if self.running == False:
            threads = []
            name = os.path.join(SIGNAL_PATH, self.user_now)
            task_name = os.path.join(name, 'select_setting.json')
            selected_task_list = sorted(read_selected_task(task_name))
            # print('selected_task_list',selected_task_list)
            if self.model_path.split('DM')[-1] in '.model':
                self.classifier = ClassifierMachine(model_name=self.model_path)
                pass
            else:
                self.classifier = ClassifierGraph(model_name=self.model_path)

            for i in range(len(selected_task_list)):
                self.predict_file_queue_list.append(Queue())  # 队列实例化
                # infer_threading = threading.Thread(target=lambda:self.inference_run(i))


                data_produce_threading = threading.Thread(target=self.data_produce_run,args=(selected_task_list[i],self.predict_file_queue_list[i]))
                infer_threading = threading.Thread(target=self.inference_run,args=(selected_task_list[i],self.predict_file_queue_list[i]))
                threads.append(infer_threading)
                threads.append(data_produce_threading)

            # t1 = threading.Thread(target=lambda:self.drawPic(self.canvases))
            # t2 = threading.Thread(target=lambda:self.bar_chart())
            # threads.append(t1)  # 将这个子线程添加到线程列表中
            # threads.append(t2)  # 将这个子线程添加到线程列表中

            for t in threads:  # 遍历线程列表
                t.setDaemon(True)  # 将线程声
                t.start()  # 启动子线程
                self.running = True
                self.running_strvar.set('诊断任务运行中')

    def threading_draw(self):
        t = threading.Thread(target=lambda: self.drawPic(self.canvases))
        t.setDaemon(True)  # 将线程声
        # print('draw threading start')
        t.start()  # 启动子线程
    def drawPic(self, canvases, x=t, y=s, add_model=False):
        canvases=self.canvases
        # 在[0,100]范围内随机生成sampleCount个数据点
        color = ['b', 'r', 'y', 'g']
        y = []
        selected_view_name_old = self.selected_view_name
        # print('selected_view_name_old:',selected_view_name_old)
        while True:
            # print('loop!')
            if self.running:
                # start = time.time()
                path = './data/signal/'+self.user_now+'/'+self.selected_view_name +'/result.csv'
                path_smooth = './data/signal/'+self.user_now+'/'+self.selected_view_name +'/result_smooth.csv'
                # # print('path',path)
                # print(self.selected_view_name)

                if self.selected_view_name == selected_view_name_old:
                    # y = get_addition_signal_csv(path,y)
                    y = get_signal_csv(path)
                    y_smooth = get_signal_csv(path_smooth)
                else:
                    selected_view_name_old = self.selected_view_name
                    # y = get_signal_csv(path)
                    y_smooth = get_signal_csv(path_smooth)
                ##########################
                y = y_smooth
                ##########################
                lenthy = len(y)
                x = [i for i in range(lenthy)]
                # # print(y)
                # time_1 = time.time()
                # print('plot time, read:',  time_1- start)

                x_sub = x
                y_sub = y
                # 清空图像，以使得前后两次绘制的图像不会重叠
                canvases.a.cla()
                # if not add_model:
                #     canvases.f.clf()
                #     canvases.a = canvases.f.add_subplot(111)
                # time_2 = time.time()
                # print('plot time, clean:', time_2 - time_1)

                # 绘制这些随机点的散点图，颜色随机选取
                # drawPic.a.scatter(x, y, s=1, color=color[np.random.randint(len(color))])
                canvases.a.plot(x_sub, y_sub, color=color[2])
                x_len = lenthy+200
                # y_len = max(y_sub+[0])+200
                y_len = 3000
                canvases.a.axis((0,x_len,0,y_len))
                # canvases.a.set_title('Demo: Draw N Random Dot')
                canvases.canvas.show()

                if len(y)>1:
                    show_str = 'Time:'+ str(x[-1])+' RUL:'+str(round(y[-1],2))
                    self.rul_strvar.set(show_str)
                # canvases.canvas.draw()
                # print('plot time, plot:', time_2-start)

                # end = time.time()
                # print('plot time:', end-start)
                # time.sleep(0.05)

    def drawPic_onece(self, canvases, x=t, y=s, add_model=False):
        # print('selected_view_name:',self.selected_view_name)
        start = time.time()
        # 在[0,100]范围内随机生成sampleCount个数据点
        color = ['b', 'r', 'y', 'g']
        path = './data/signal/'+self.user_now+'/'+self.selected_view_name +'/result.csv'
        path_smooth = './data/signal/'+self.user_now+'/'+self.selected_view_name +'/result_smooth.csv'
        y = get_signal_csv(path)
        y_smooth = get_signal_csv(path_smooth)
        ##########################
        y = y_smooth
        ##########################

        x = [i for i in range(len(y))]
        x_sub = x
        y_sub = y
        # 清空图像，以使得前后两次绘制的图像不会重叠
        if not add_model:
            canvases.a.cla()
            # canvases.f.clf()
            # canvases.a = canvases.f.add_subplot(111)
        # 绘制这些随机点的散点图，颜色随机选取
        # drawPic.a.scatter(x, y, s=1, color=color[np.random.randint(len(color))])
        canvases.a.plot(x_sub, y_sub, color=color[2])
        x_len = len(x_sub)+200
        # y_len = max(y_sub+[0])+200
        y_len = 3000
        canvases.a.axis((0,x_len,0,y_len))
        # canvases.a.set_title('Demo: Draw N Random Dot')
        canvases.canvas.show()
        if len(y) > 1:
            show_str = 'Time:' + str(x[-1]) + ' RUL:' + str(round(y[-1], 2))
            self.rul_strvar.set(show_str)
        end = time.time()
        # print('plot once time',end-start)


    def inference_run(self, data_name,predict_file_queue):

        # 在[0,100]范围内随机生成sampleCount个数据点
        # predict_file_queue.start()
        while self.running:
            print('in inference,', len(predict_file_queue.queue))

            if len(predict_file_queue.queue)>0:
                # print('run inference')
                file_path = predict_file_queue.get()
                input_dim = self.model_path.split('DM')[-2]
                res = predict_rul_pb(file_path,self.classifier,int(input_dim))
                # # print('inference',data_name,' :',i)
                write_add_csv(filename='./data/signal/'+self.user_now+'/'+data_name+'/result.csv',
                              contex=[res]) #res is a 298.1
                (filepath, tempfilename) = os.path.split(file_path)
                # print(tempfilename)
                predicted_vib_sig_log = './data/signal/' + self.user_now + '/' + data_name + '/predicted_vib_sig.csv'
                write_add_csv(filename=predicted_vib_sig_log,
                              contex=[tempfilename])
                smooth_res = self.liner_regre('./data/signal/'+self.user_now+'/'+data_name+'/result.csv')
                if smooth_res[0]<200:
                    # print('alarm!')
                    self.running_strvar.set('低可靠警告！')
                if smooth_res[0]<100:
                    self.stop_running_flag()
                write_add_csv(filename='./data/signal/'+self.user_now+'/'+data_name+'/result_smooth.csv',
                              contex=smooth_res) #res is a list[[298.1]]
                time.sleep(0.2)
    def liner_regre(self,filename):
        predict_list = get_signal_csv(filename)
        if len(predict_list) > 500:
            predict_list = predict_list[-500:]

        if len(predict_list)>1:
            x = [[i] for i in range(len(predict_list))]
            linreg = linear_model.LinearRegression()
            linreg.fit(x,predict_list)
            result = linreg.predict([x[-1]])
        else:
            result = [predict_list[-1]]
        return result
        pass


    def data_produce_run(self, data_name,predict_file_queue):

        # 在[0,100]范围内随机生成sampleCount个数据点
        # predict_file_queue.start()

        # origin_path = r'G:\Bearing\bearing\data_set\FEMTOBearingDataSet\Training_set\Learning_set\Bearing1_1'
        origin_path =read_json('./data/signal/'+self.user_now+'/' + data_name + '/data_origin_path.json')[0]

        predicted_vib_sig_log = './data/signal/'+self.user_now+'/' + data_name + '/predicted_vib_sig.csv'
        while self.running:
            origin_files = sorted(os.listdir(origin_path))
            files = get_str_csv(predicted_vib_sig_log)
            for i in range(len(files),len(origin_files)):
                if self.running:
                    real_copy_files = os.path.join(origin_path,origin_files[i])
                    # copyfile_func(real_copy_files, dstfile)

                    # write_add_csv(filename=predicted_vib_sig_log,
                    #           contex=[origin_files[i]])

                    predict_file_queue.put(real_copy_files)
                    if self.auto_updating:
                        splited_list = data_name.split('_ident_')
                        url_send_file(real_copy_files, url=URL+'/edge_uploading/'+splited_list[-1]+'/'+self.user_now)
                    time.sleep(0.2)
                    # print('copy:',origin_files[i])
                else:
                    break
            time.sleep(0)
    def stop_running_flag(self):
        self.running = False
        self.running_strvar.set('诊断任务已停止')


    def change_auto_updating(self):
        if self.auto_updating:
            self.stop_auto_updating()
        else:
            self.begin_auto_updating()
        pass

    def begin_auto_updating(self):
        self.auto_updating = 1
        self.auto_updating_strvar.set('关闭自动模式')
        # print('自动模式：',self.auto_updating)
    def stop_auto_updating(self):
        self.auto_updating = 0
        self.auto_updating_strvar.set('开启自动模式')
        # print('自动模式：',self.auto_updating)

    #############################################
    def update_model_threading(self):

        download_thread = threading.Thread(target=self.update_model)
        download_thread.setDaemon(True)  # 将线程声明为守护线程，必须在start() 方法调用之前设置，如果不设置为守护线程程序会被无限挂起
        download_thread.start()  # 启动子线程
    def update_model(self):
        # print(self.model_path)
        self.update_strvar.set('开始更新模型')
        name_part = self.model_path.split('_')[0]
        name_num_list = self.get_models_cloud()
        name_full = '1311'
        for name in name_num_list:
            if name_part in name:
                name_full = name
                break
        if name_full!='1311':
            self.download_pb(name_full)
            self.update_strvar.set('模型已更新')
        else:
            self.update_strvar.set('模型未找到')
            # print("not found!")
        pass
    def get_models_cloud(self):
        # try:
        #     url_str = url_get_string(url = URL+'/return_list')
        # except:
        #     pass
        # print(url_str)
        # url_str = ['1111','222222']
        try:
            name_num_list = get_file_list_from_net(URL+'/train_task/'+self.user_now)
        except:
            print('net connection error!')
            name_num_list=[]
        return name_num_list


    #############################################


    def creatTitle(self, name,position=[100, 220]):
        # self.plottitle = Label(self.master,text='A1 Label',background='red', foreground='red',bg='brown')
        self.plottitle = Label(self.master,text=name,font=2)
        self.window_canvas.create_window(position[0], position[1],window=self.plottitle)
        pass
    def creatMenu(self):
        self.menubar = Menu(self.master)

        self.filemenu = Menu(self.menubar, tearoff=False)
        # self.filemenu.add_command(label="打开", command=self.openpicture, accelerator='Ctrl+N')
        self.filemenu.add_command(label="设置", command=self.openpicture)
        self.filemenu.add_command(label="选择模型文件", command=self.openfile)

        self.filemenu.add_separator()#添加分割线

        self.filemenu.add_command(label="退出", command=self.exitSystem)

        #添加多级菜单
        self.menubar.add_cascade(label="文件", menu=self.filemenu)
        # self.menubar.add_cascade(label="文件", menu=self.filemenu2)

        #将菜单添加到整个窗口
        self.master.config(menu=self.menubar)

        # self.master.bind_all("<Control-n>", lambda event: print '加速键Ctrl+N')
    def creat_canvas(self):
        imagepath1 = r'./873.JPG'  # include the path for the image (use 'r' before the path string to address any special character such as '\'. Also, make sure to include the image type - here it is jpg)
        self.image_down = ImageTk.PhotoImage(file=imagepath1)  # PIL module
        self.window_canvas.create_image(700, 240, image=self.image_down)  # create a canvas image to place the background image
        pass

    def createWidgets(self):

        self.rul_strvar = StringVar()
        self.rul_strvar.set('Time:0    RUL:0.0  ')
        self.show_rul = Label(self.master, textvariable=self.rul_strvar, font=1)
        self.window_canvas.create_window(650, 240, window=self.show_rul)


        self.update_strvar = StringVar()
        self.update_strvar.set('操作状态显示')
        self.show_update_strvar = Label(self.master, textvariable=self.update_strvar, font=1)
        self.window_canvas.create_window(985, 100, window=self.show_update_strvar)


        self.running_strvar = StringVar()
        self.running_strvar.set('诊断任务已停止')
        self.show_running = Label(self.master, textvariable=self.running_strvar, font=1)
        self.window_canvas.create_window(450, 240, window=self.show_running)


        self.begin_Plot_Button = Button(self.master, text='开始诊断任务',activeforeground='blue',activebackground='yellow', command=lambda:self.matplotlib_thread(self.master))
        self.window_canvas.create_window(100, 100,window=self.begin_Plot_Button)


        self.stop_Plot_Button = Button(self.master, text='停止诊断任务', activeforeground='blue',activebackground='yellow',command=lambda:self.stop_running_flag())
        self.window_canvas.create_window(100, 150, window=self.stop_Plot_Button)


        self.tasksetButton = Button(self.master, text='诊断任务选择', activeforeground='blue',activebackground='yellow',command=self.task_select)
        self.window_canvas.create_window(200, 100, window=self.tasksetButton)


        self.tasksetButton = Button(self.master, text='显示任务选择', activeforeground='blue',activebackground='yellow',command=self.show_select)
        self.window_canvas.create_window(200, 150, window=self.tasksetButton)

        self.tasknewButton = Button(self.master, text='  创建新任务 ', activeforeground='blue',activebackground='yellow',command=self.new_task)
        self.window_canvas.create_window(300, 100, window=self.tasknewButton)

        self.begin_Plot_Button = Button(self.master, text='清空诊断记录', activeforeground='blue',activebackground='yellow',command=lambda:self.clear_result())
        self.window_canvas.create_window(300, 150, window=self.begin_Plot_Button)


        self.piccloseButton = Button(self.master, text='云端查询模型', activeforeground='blue',activebackground='yellow',command=self.open_search)
        self.window_canvas.create_window(600, 100, window=self.piccloseButton)

        self.localmodelButton = Button(self.master, text='本地模型选择', activeforeground='blue',activebackground='yellow',command=self.open_pbfile)
        self.window_canvas.create_window(600, 150, window=self.localmodelButton)


        self.auto_updating_strvar = StringVar()
        self.auto_updating_strvar.set('开启自动模式')
        self.auto_updating_Button = Button(self.master, textvariable=self.auto_updating_strvar, activeforeground='blue',activebackground='yellow',command=self.change_auto_updating)
        self.window_canvas.create_window(700, 100, window=self.auto_updating_Button)

        self.update_model_Button = Button(self.master, text='   更新模型   ', activeforeground='blue',activebackground='yellow',command=self.update_model_threading)
        self.window_canvas.create_window(700, 150, window=self.update_model_Button)

        self.imglabel = Label(self.master, bg='brown')
    def bar_chart_onece(self):
        color_val=240.0/255.0
        self.figurebar = Figure(facecolor=[color_val,color_val,color_val],figsize=(4, 4), dpi=100)  # create a Figure (matplotlib module)
        self.barsubplot = self.figurebar.add_subplot(111)  # add a subplot
        self.barsubplot.set_ylim(0,3000)
        name = os.path.join(SIGNAL_PATH, self.user_now)
        task_name = os.path.join(name, 'select_setting.json')
        selected_task_list = sorted(read_selected_task(task_name))
        yAxis_good = []
        xAxis_good = []
        yAxis_danger = []
        xAxis_danger = []
        for i in range(len(selected_task_list)):
            path_smooth = './data/signal/' + self.user_now + '/' + selected_task_list[i] + '/result_smooth.csv'
            try:
                y_smooth = get_signal_csv(path_smooth)[-1]
            except:
                y_smooth = 0
            if y_smooth > 200:
                yAxis_good.append(y_smooth)
                xAxis_good.append(i)
            else:
                yAxis_danger.append(y_smooth)
                xAxis_danger.append(i)
        # xAxis = [float(1), float(2),
        #          float(3)]  # intakes the values inserted under x1, x2 and x3 to represent the x Axis
        # yAxis = [float(0.1), float(0.1),
        #          float(0.1)]  # intakes the values inserted under x1, x2 and x3 to represent the y Axis
        self.barsubplot.bar(xAxis_danger, yAxis_danger, color='r')  # create the bar chart based on the input variables x1, x2, and x3
        self.barsubplot.bar(xAxis_good, yAxis_good, color='b')  # create the bar chart based on the input variables x1, x2, and x3
        self.bar_char = FigureCanvasTkAgg(self.figurebar, self.master)  # create a canvas figure (matplotlib module)
        # bar1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=0)
        self.window_canvas.create_window(996, 429, window=self.bar_char.get_tk_widget())

        t = threading.Thread(target=lambda: self.bar_chart())
        t.setDaemon(True)  # 将线程声
        t.start()  # 启动子线程

    def bar_chart(self):
        while True:
            if self.running:
                name = os.path.join(SIGNAL_PATH, self.user_now)
                task_name = os.path.join(name, 'select_setting.json')
                selected_task_list = sorted(read_selected_task(task_name))
                yAxis_good = []
                xAxis_good = []
                yAxis_danger = []
                xAxis_danger = []
                for i in range(len(selected_task_list)):
                    path_smooth = './data/signal/'+self.user_now+'/'+selected_task_list[i] +'/result_smooth.csv'
                    try:
                        y_smooth = get_signal_csv(path_smooth)[-1]
                    except:
                        y_smooth  = 0
                    if y_smooth>200:
                        yAxis_good.append(y_smooth)
                        xAxis_good.append(i)
                    else:
                        yAxis_danger.append(y_smooth)
                        xAxis_danger.append(i)
                    # xAxis.append(one_task_name)
                # xAxis = [i for i in range(len(yAxis))]
                self.barsubplot.cla()
                self.barsubplot.set_ylim(0, 3000)
                self.barsubplot.bar(xAxis_good, yAxis_good, color='b')
                self.barsubplot.bar(xAxis_danger, yAxis_danger, color='r')
                self.bar_char.show()

                time.sleep(0.01)

    def hello(self):
        name = self.nameInput.get() or 'world'
        messagebox.showinfo('Message', 'Hello, %s' % name)

    def pictureRead(self,path=r"F:\\bee.jpg"):
        # 将图片添加到窗口
        img = cv2.imread(path)

        b, g, r = cv2.split(img)
        img = cv2.merge((r, g, b))

        # cv2.namedWindow("Image")
        # cv2.imshow("Image", img)

        im = Image.fromarray(img)

        # im.show()

        self.tk_image = ImageTk.PhotoImage(image=im)
        # print self.tk_image.height()

        #self.imglabel = Label(self, image=self.tk_image, bg='brown')
        # 在构造时候声明有个Label,在使用时候给Label填内容，否则容易出bug
        self.imglabel.configure(image = self.tk_image)



        # except:
        #     print ('Read Picture ERROR')

    def openpicture(self,path=r"F:\\bee.jpg"):
        self.pictureRead(path=path)
        try:
            self.imglabel.pack(padx=5, pady=5)
        except:
            # print ('Show Picture ERROR')
            pass

    #弹出新窗口打开
    #多线程包装一下的版本
    def openpictureInNew(self,path=r"F:\\bee.jpg"):
        import time, threading

        threads = []
        t1 = threading.Thread(target=self.openpictureInNew_func)
        threads.append(t1)  # 将这个子线程添加到线程列表中

        for t in threads:  # 遍历线程列表
            t.setDaemon(True)  # 将线程声明为守护线程，必须在start() 方法调用之前设置，如果不设置为守护线程程序会被无限挂起
            t.start()  # 启动子线程
    def openpictureInNew_func(self,path=r"./bee.jpg"):
        # for i in range(1000000):
            # print(i)
        img = cv2.imread(path)

        b, g, r = cv2.split(img)
        img = cv2.merge((r, g, b))

        im = Image.fromarray(img)

        self.tk_image = ImageTk.PhotoImage(image=im)

        top = Toplevel()
        topimglabel = Label(top, bg='brown')
        topimglabel.configure(image=self.tk_image)

        topimglabel.pack(padx=5, pady=5)
    def closepicture(self):
        self.imglabel.pack_forget()#将该组件隐藏

        #self.tk_image.__del__()

    def openfile(self):
        #获得了全路径名
        path = filedialog.askopenfilename(title='选择模型文件', initialdir='./pb_file',filetypes=[('图片', '*.jpg'),('Python', '*.py *.pyw'), ('All Files', '*')])

        self.openpicture(path=path)


    def exitSystem(self):

        exit()
####################################################################################
    def open_search(self):
        id = "New window"
        # print(id)
        self.window_search = Toplevel(self.master,)



        self.window_search.title('模型获取')
        # 设置窗口尺寸：‘宽*长+ 横坐标+纵坐标’
        self.window_search.geometry('400x300+500+5')

        plottitle2 = Label(self.window_search,text='点击查询云端模型',font=1)
        plottitle2.grid(row=0, column=1)

        search_Button = Button(self.window_search, text='查询', command=lambda:self.serch_internet_threading())
        # self.search_Button.pack()
        search_Button.grid(row=1, column=4)
        # label = Label(window, text=id)
        # label.pack(side="top", fill="both", padx=50, pady=40)
        pass

    def serch_internet_threading(self):
        search_pb_threading = threading.Thread(target=self.serch_internet)
        search_pb_threading.setDaemon(True)
        search_pb_threading.start()
        pass
    def serch_internet(self):
        # print('net connection!')
        # URL = 'http://192.168.0.177:5000'
        # try:
        #     url_str = url_get_string(url = URL+'/return_list')
        # except:
        #     pass
        # url_str = ['1111','222222']
        try:
            name_num_list = get_file_list_from_net(URL+'/train_task/'+self.user_now)
        except:
            self.update_strvar.set('无法访问云')
            # print('net connection error!')
            name_num_list=[]
        # if len(name_num_list)==0:
        #     name_num_list=['未查询到模型']

        def go(*args):  # 处理事件，*args表示可变参数
            i = comboxlist.get()  # 打印选中的值
            # # print(i)
            return i

        if len(name_num_list) >0:
            comvalue = StringVar(self.window_search)  # 窗体自带的文本，新建一个值
            comboxlist = ttk.Combobox(self.window_search, textvariable=comvalue)  # 初始化
            comboxlist["values"] = name_num_list
            comboxlist.current(0)  # 选择第一个
            comboxlist.bind("<<ComboboxSelected>>", go)  # 绑定事件,(下拉列表框被选中时，绑定go()函数)
            comboxlist.grid(row=2, column=4)

            download_Button = Button(self.window_search, text='下载', command=lambda:self.download_pb(selected=go()))
            # self.search_Button.pack()
            download_Button.grid(row=2, column=7)
        else:
            pass
        pass
    def download_pb(self,selected):
        # print('下载功能')

        # print(selected)
        name_num = selected.split('--')
        name = name_num[0]
        name_id = name_num[1]
        todo_list = name_num[-1]
        input_dim = url_get_input_dim(url=URL + '/getinputdim/' + self.user_now + '/' + todo_list)
        # url_download(url=URL+'/download/'+self.user_now+'/'+todo_list+'/mist_binary_clas_quaintized.pb',dist_path='./data/model/'+name+'.pb')
        try:
            url_download(url=URL+'/download/'+self.user_now+'/'+todo_list+'/_quaintized.pb',dist_path='./data/model/'+name+'_'+name_id+'_DM'+str(input_dim)+'DM'+'.pb')
        except:
            url_download(url=URL+'/download/'+self.user_now+'/'+todo_list+'/_quaintized.model',dist_path='./data/model/'+name+'_'+name_id+'_DM'+str(input_dim)+'DM'+'.model')
        # print('下载完成')
        try:
            self.window_search.destroy()
        except:
            pass
####################################################################


    def open_pbfile(self):
        id = "New window pb"
        # print(id)
        self.window_pbfile = Toplevel(self.master,)
        self.window_pbfile.title('模型选择')
        # 设置窗口尺寸：‘宽*长+ 横坐标+纵坐标’
        self.window_pbfile.geometry('400x300+500+5')
        plottitle2 = Label(self.window_pbfile, text='点击查询本地模型', font=1)
        plottitle2.grid(row=0, column=1)

        search_Button = Button(self.window_pbfile, text='查询', command=lambda:self.serch_pbfile())
        # self.search_Button.pack()
        search_Button.grid(row=0, column=4)
        # label = Label(window, text=id)
        # label.pack(side="top", fill="both", padx=50, pady=40)
        pass

    def serch_pbfile(self):
        # print('查询本地模型!')

        def go(*args):  # 处理事件，*args表示可变参数
            i = comboxlist.get()  # 打印选中的值
            # # print(i)
            return i

        filenames = get_file_list_local(path='./data/model/')

        comvalue = StringVar(self.window_pbfile)  # 窗体自带的文本，新建一个值
        comboxlist = ttk.Combobox(self.window_pbfile, textvariable=comvalue)  # 初始化
        comboxlist["values"] = filenames
        comboxlist.current(0)  # 选择第一个
        comboxlist.bind("<<ComboboxSelected>>", go)  # 绑定事件,(下拉列表框被选中时，绑定go()函数)
        comboxlist.grid(row=1, column=4)

        download_Button = Button(self.window_pbfile, text='确认使用', command=lambda:self.used_pb(selected=go()))
        # self.search_Button.pack()
        download_Button.grid(row=1, column=7)


        pass
    def used_pb(self,selected):
        # selected = selected
        # print(selected)

        name = os.path.join(SIGNAL_PATH,self.user_now)
        model_name = os.path.join(name,'model_setting.json')
        modifi_view_set([selected], model_name)
        self.model_path = read_json('./data/signal/'+self.user_now+'/model_setting.json')[0]
        self.window_pbfile.destroy()

        self.update_strvar.set('模型已选择')
############################################################################################

    def user_login(self):
        get_user_now = get_now_user()

        if get_user_now is not '':
            self.user_now = get_user_now
        else:
            self.user_now = 'admin'

        self.user_var = StringVar()
        self.user_var.set(self.user_now)

        self.user_show = Label(self.master, text='用户：', font=1)
        self.window_canvas.create_window(100, 50, window=self.user_show)

        self.user_name_show = Label(self.master, textvariable=self.user_var , font=1)
        self.window_canvas.create_window(150, 50, window=self.user_name_show)

        self.logoutButton = Button(self.master, text='更换用户', command=self.change_name)
        self.window_canvas.create_window(900, 50, window=self.logoutButton)


        self.new_user_Button = Button(self.master, text='新建本地用户', command=self.creat_user)
        self.window_canvas.create_window(980, 50, window=self.new_user_Button)
        self.model_path = read_json('./data/signal/'+self.user_now+'/model_setting.json')[0]
        pass

    def creat_user(self):

        def creat():

            name = self.newnameInput.get() or 'admin'
            user_dir = 'data/signal/' + name
            result = creat_dir(user_dir)

            if result:

                select_file_name = os.path.join(user_dir, 'select_setting.json')
                view_file_name = os.path.join(user_dir, 'view_setting.json')
                model_file_name = os.path.join(user_dir, 'model_setting.json')

                write_json([''], select_file_name)
                write_json([''], view_file_name)


                self.window_adduser.destroy()
            else:

                show_no_user = Label(self.window_adduser, text='用户已经存在', font=1)
                show_no_user.grid(row=4, column=0, rowspan=2, columnspan=2)
                pass
            pass


        id = "New window add name"
        # # print(id)
        self.window_adduser = Toplevel(self.master,)
        self.window_adduser.title('新建用户')
        # 设置窗口尺寸：‘宽*长+ 横坐标+纵坐标’
        self.window_adduser.geometry('400x300+500+5')

        show_function = Label(self.window_adduser, text='请输入新建用户名：', font=1)
        show_function.grid(row=0, column=0, rowspan=2, columnspan=4)

        self.newnameInput = Entry(self.window_adduser)
        self.newnameInput.grid(row=2, column=0)
        search_Button = Button(self.window_adduser, text='确认', command=lambda:creat())
        # self.search_Button.pack()
        search_Button.grid(row=2, column=4)
        pass

    def change_name(self):

        def modify_name():
            name = self.nameInput.get() or 'admin'
            exit_user = get_file_list_local('./data/signal')
            if name in exit_user:

                modifi_user(name)
                self.user_now = name
                self.user_var.set(self.user_now)
                self.renew_task_show()
                self.model_path = read_json('./data/signal/' + self.user_now + '/model_setting.json')[0]
                self.window_changename.destroy()
            else:
                show_no_user = Label(self.window_changename, text='用户未存在', font=1)
                show_no_user.grid(row=4, column=0, rowspan=2, columnspan=2)
                pass
            pass


        id = "New window change name"
        # # print(id)
        self.window_changename = Toplevel(self.master,)
        self.window_changename.title('模型选择')
        # 设置窗口尺寸：‘宽*长+ 横坐标+纵坐标’
        self.window_changename.geometry('400x300+500+5')

        show_function = Label(self.window_changename, text='请输入用户名：', font=1)
        show_function.grid(row=0, column=0, rowspan=2, columnspan=4)

        self.nameInput = Entry(self.window_changename)
        self.nameInput.grid(row=2, column=0)
        search_Button = Button(self.window_changename, text='确认', command=lambda:modify_name())
        # self.search_Button.pack()
        search_Button.grid(row=2, column=4)
        pass
    def new_task(self):

        path_var = StringVar()
        def selected_path():
            origin_path = askdirectory()

            path_var.set(origin_path)

        def modify_task():
            new_task_name = self.tasknameInput.get()

            name = os.path.join(SIGNAL_PATH, self.user_now)
            # task_names = os.path.join(name, 'select_setting.json')
            name_list = sorted(get_file_list_local(name))
            # if new_task_name not in name_list:
            creat_time = int(time.time())
            new_task_full_name = new_task_name + '_ident_'+str(creat_time)
            dir_name = os.path.join(name, new_task_full_name)
            os.makedirs(dir_name)
            origin_data_file = os.path.join(dir_name,'data_origin_path.json')
            data_origin_set_valu = path_var.get()
            # # print(origin_data_file)
            write_json([data_origin_set_valu], origin_data_file)
            self.renew_task_show()
            self.window_new_task.destroy()
            # else:
            #     show_no_user = Label(self.window_new_task, text='任务名已存在', font=1)
            #     show_no_user.grid(row=4, column=0, rowspan=2, columnspan=2)
            #     pass
            self.update_strvar.set('创建任务已完成')
            pass


        id = "New window change name"
        # # # print(id)
        self.window_new_task = Toplevel(self.master,)
        self.window_new_task.resizable(0,0)
        # self.window_new_task.attributes("-toolwindow",1)
        # self.window_new_task.wm_attributes("-topmost",1)

        self.window_new_task.title('新建任务')
        # 设置窗口尺寸：‘宽*长+ 横坐标+纵坐标’
        self.window_new_task.geometry('400x300+500+5')

        show_function = Label(self.window_new_task, text='任务设置', font=1)
        show_function.grid(row=0, column=0, rowspan=2, columnspan=4)

        show_function = Label(self.window_new_task, text='请输入新建任务名称：')
        show_function.grid(row=2, column=0, rowspan=1, columnspan=1)

        self.tasknameInput = Entry(self.window_new_task)
        self.tasknameInput.grid(row=2, column=1)

        Label(self.window_new_task,text='目标路径').grid(row=3, column=0)
        Entry(self.window_new_task,textvariable=path_var,).grid(row=3,column=1)

        Button(self.window_new_task,text='路径选择',command=selected_path).grid(row=3,column=2)
        search_Button = Button(self.window_new_task, text='确认', command=lambda:modify_task())
        # self.search_Button.pack()
        search_Button.grid(row=4, column=4)
        pass

############################################################################################

    def task_select(self):
        id = "New window task_select"
        # # print(id)
        window_task_select = Toplevel(self.master,)
        window_task_select.title('诊断任务选择')
        # 设置窗口尺寸：‘宽*长+ 横坐标+纵坐标’
        window_task_select.geometry('800x300+500+20')


        name = os.path.join(SIGNAL_PATH,self.user_now)

        task_name = os.path.join(name,'select_setting.json')
        name_list = sorted(get_file_list_local(name))
        task_flag = []

        def add_task_name(ii):
            task_flag[ii] = not task_flag[ii]
            # # print(ii,task_flag[ii])
            # self.task_name.append(name_list[i])

        taskbutton_list = []

        from functools import partial

        for ibb in range(len(name_list)):

            action_with_arg = partial(add_task_name, ibb)

            taskbutton_list.append(Checkbutton(window_task_select, text=name_list[ibb], command=action_with_arg))
            taskbutton_list[-1].deselect()
            task_flag.append(False)
            taskbutton_list[-1].grid(row=ibb//4, column=ibb%4)
        # ck2 = Checkbutton(frm, text='唱歌', command=click_2)

        # ck3 = Checkbutton(frm, text='旅游', command=click_3)


        def write_select_set():
            name_write_list = []
            # print(task_flag)
            for itt in range(len(name_list)):
                if task_flag[itt]:
                    name_write_list.append(name_list[itt])
            modifi_select_set(name_write_list,task_name)

            window_task_select.destroy()
            pass

        search_Button = Button(window_task_select, text='确定', command=lambda:write_select_set())
        search_Button.grid(row=len(name_list)//4+1, column=1)
        pass


    def show_select(self):
        id = "New window task_select"
        # print(id)
        self.window_show_select = Toplevel(self.master,)
        self.window_show_select.title('显示任务选择')
        # 设置窗口尺寸：‘宽*长+ 横坐标+纵坐标’
        self.window_show_select.geometry('800x300+500+20')


        name = os.path.join(SIGNAL_PATH,self.user_now)
        task_name = os.path.join(name,'select_setting.json')
        selected_task_list = sorted(read_selected_task(task_name))
        task_flag = []

        def add_task_name(ii):
            task_flag[ii] = not task_flag[ii]
            # print(ii,task_flag[ii])
            # self.task_name.append(name_list[i])

        showbutton_list = []

        from functools import partial

        for ibb in range(len(selected_task_list)):

            action_with_arg = partial(add_task_name, ibb)

            showbutton_list.append(Checkbutton(self.window_show_select, text=selected_task_list[ibb], command=action_with_arg))
            showbutton_list[-1].deselect()
            task_flag.append(False)
            showbutton_list[-1].grid(row=ibb//4, column=ibb%4)
        # ck2 = Checkbutton(frm, text='唱歌', command=click_2)

        # ck3 = Checkbutton(frm, text='旅游', command=click_3)


        def write_select_set():
            name_write_list = []
            # print(task_flag)
            for itt in range(len(selected_task_list)):
                if task_flag[itt]:
                    name_write_list.append(selected_task_list[itt])
            name = os.path.join(SIGNAL_PATH, self.user_now)
            show_name = os.path.join(name, 'view_setting.json')
            modifi_view_set(name_write_list,show_name)
            self.renew_task_show()

            self.window_show_select.destroy()
            pass

        search_Button = Button(self.window_show_select, text='确定', command=lambda:write_select_set())
        search_Button.grid(row=len(selected_task_list)//4+1, column=1)
        pass

    def multi_task_show(self):
        name = os.path.join(SIGNAL_PATH,self.user_now)
        show_name = os.path.join(name, 'view_setting.json')
        # task_list = sorted(get_file_list_local(name))
        self.task_list = sorted(read_selected_task(show_name))
        self.task_var = IntVar()
        self.multi_task = []
        self.selected_view_name = self.task_list[self.task_var.get()]
        for j in range(18):
            self.creatTitle('                            ',position=[50, 262+20*j])
        self.creatTitle('                            ', position=[50, 258 + 20 * 18])
        for i in range(len(self.task_list)):
            self.multi_task.append(Radiobutton(self.master, text=self.task_list[i], variable=self.task_var, value=i, command=self.show_one_task))
            # self.multi_task[i].grid(row=i+2, column=0,rowspan=2,columnspan=4)
            self.window_canvas.create_window(100, 290+25*i, window=self.multi_task[i])
            pass

        pass
    def renew_task_show(self):
        for i in range(len(self.multi_task)):
            self.multi_task[i].grid_forget()
        self.multi_task_show()

    def show_one_task(self):
        # print('show_one_task:',self.task_var.get())
        self.selected_view_name = self.task_list[self.task_var.get()]
        if not self.running:
            self.drawPic_onece(self.canvases)
            # pass
        pass

    def clear_result(self):
        name = os.path.join(SIGNAL_PATH, self.user_now)
        task_name = os.path.join(name, 'select_setting.json')
        selected_task_list = sorted(read_selected_task(task_name))

        if self.running == False:
            for i in range(len(selected_task_list)):
                data_name = selected_task_list[i]
                predicted_vib_sig_log = './data/signal/' + self.user_now + '/' + data_name + '/predicted_vib_sig.csv'
                result = './data/signal/' + self.user_now + '/' + data_name + '/result.csv'
                result_smooth = './data/signal/'+self.user_now+'/'+data_name+'/result_smooth.csv'

                try:
                    self.predict_file_queue_list[i].queue.clear()
                except:
                    # print('clean error')
                    self.update_strvar.set('内存已释放')
                try:
                    os.remove(predicted_vib_sig_log)
                except:
                    # print('clean error')
                    self.update_strvar.set('记录已清除')
                try:
                    os.remove(result)
                except:
                    self.update_strvar.set('结果已清除')
                try:
                    os.remove(result_smooth)
                except:
                    self.update_strvar.set('平滑已清除')
            self.drawPic_onece(self.canvases)
            self.running_strvar.set('诊断记录已清除')
        else:
            self.update_strvar.set('先停止任务')
############################################################################################
    # def selected_path(self):
    #     origin_path = askdirectory()
    #     path_var.set(origin_path)
    #     # path = '.data/signal/' + self.user_now + '/' + task_name + '/data_origin_path.json'
    #     # write_json([origin_path], path)
    #     pass
############################################################################################

    # 外部参数设置
    # root = Tk() # 初始化Tk()
    #
    # root.title("frame-test")    # 设置窗口标题
    #
    # root.geometry("300x200")    # 设置窗口大小 注意：是x 不是*
    #
    # root.resizable(width=True, height=False) # 设置窗口是否可以变化长/宽，False不可变，True可变，默认为True
    #
    # Label(root, text="frame", bg="red", font=("Arial",15)).pack()
    #
    # app = Application(root)
    # 设置窗口标题:
    # app.master.title('测试窗口')

if __name__ =='__main__':
    # root = Tk() # 初始化Tk()
    root = tk.Tk() # 初始化Tk()

    app1 = Application(root)

    while True:
        try:
            root.mainloop()
            break
        except UnicodeDecodeError:
            pass


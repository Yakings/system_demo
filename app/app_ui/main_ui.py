#!/usr/bin/env python
# coding:utf-8
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
from PIL import Image, ImageTk

import threading,cv2

from tkinter import ttk

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import time

from app.app_backend.net_search import get_file_list,url_download
from app.app_backend.fileRead import get_file_list,get_now_user,creat_dir
from app.app_backend.fileWriter import modifi_user

from numpy import arange, sin, pi
t = arange(0.0, 3000, 0.01)
s = sin(2 * pi * t)




class Canvases:
    def __init__(self, master=None):
        self.f = Figure(figsize=(8, 4), dpi=100)

        self.canvas = FigureCanvasTkAgg(self.f, master=root)
        self.canvas.show()
        self.canvas.get_tk_widget().grid(row=1, column=0,rowspan=10,columnspan=20)
        # self.canvas.get_tk_widget().pack()


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
        # 设置窗口标题:
        self.master.title('边缘计算系统')
        #设置窗口尺寸：‘宽*长+ 横坐标+纵坐标’
        self.master.geometry('800x600+500+5')


        self.canvases = Canvases(master)
        self.creatTitle('haha',position=[0,1])



        self.creatMenu()
        self.createWidgets()
        # self.listbox(root=master)

        self.user_login()



        self.running = False

    def listbox(self,root):
        mylist = tk.Listbox(root,width=50)  # 列表框
        mylist.pack()

        for item in ["1", '1','1','1','1','1','1','1','1',"asdsa", "asdsadsa", "asdsadsad"]:  # 插入内容
            mylist.insert(tk.END, item)  # 从尾部插入

    ###########################################

    def matplotlib_thread(self,root):
        threads = []
        infer_threading = threading.Thread(target=lambda:self.inference(self.canvases))

        t1 = threading.Thread(target=lambda:self.drawPic(self.canvases))
        threads.append(t1)  # 将这个子线程添加到线程列表中

        for t in threads:  # 遍历线程列表
            t.setDaemon(True)  # 将线程声
            self.running = True
            t.start()  # 启动子线程

    def drawPic(self, canvases, x=t, y=s, add_model=False):

        # 在[0,100]范围内随机生成sampleCount个数据点
        color = ['b', 'r', 'y', 'g']

        for i in range(0, 300000):

            if self.running:
                x_sub = x[i:i + 100]
                y_sub = y[i:i + 100]

                # 清空图像，以使得前后两次绘制的图像不会重叠
                if not add_model:
                    canvases.f.clf()
                    canvases.a = canvases.f.add_subplot(111)

                # 绘制这些随机点的散点图，颜色随机选取
                # drawPic.a.scatter(x, y, s=1, color=color[np.random.randint(len(color))])
                canvases.a.plot(x_sub, y_sub, color=color[2])
                # canvases.a.set_title('Demo: Draw N Random Dot')
                canvases.canvas.show()
                time.sleep(0.01)
            else:
                break


    def inference(self, canvases, x=t, y=s, add_model=False):

        # 在[0,100]范围内随机生成sampleCount个数据点
        color = ['b', 'r', 'y', 'g']

        for i in range(0, 300000):

            if self.running:
                # print('inference:',i)
                time.sleep(0.01)
            else:
                break

    def stop_running_flag(self):
        self.running = False


    #############################################

    def creatTitle(self, name,position=[1,1]):
        # self.plottitle = Label(self.master,text='A1 Label',background='red', foreground='red',bg='brown')
        self.plottitle = Label(self.master,text='预测曲线',font=1)
        self.plottitle.grid(row=position[0], column=position[1],rowspan=2,columnspan=2)
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

    def createWidgets(self):


        self.begin_Plot_Button = Button(self.master, text='开始动态绘图', command=lambda:self.matplotlib_thread(self.master))
        self.begin_Plot_Button.grid(row=11, column=1,columnspan=1)

        self.stop_Plot_Button = Button(self.master, text='停止动态绘图', command=lambda:self.stop_running_flag())
        self.stop_Plot_Button.grid(row=11, column=2,columnspan=1)

        self.piccloseButton = Button(self.master, text='云端查询模型', command=self.open_search)
        self.piccloseButton.grid(row=12, column=1,columnspan=1)

        self.piccloseButton = Button(self.master, text='本地模型选择', command=self.open_pbfile)
        self.piccloseButton.grid(row=12, column=2,columnspan=1)

        # self.netconnection_Button = Button(self.master, text='查询网络资源', command=lambda:self.serch_internet())
        # self.netconnection_Button.grid(row=12, column=1,columnspan=1)



        # self.nameInput = Entry(self.master)
        # self.nameInput.grid(row=12, column=1,columnspan=1)
        # pack()方法把Widget加入到父容器中，并实现布局。
        # pack()是最简单的布局，grid()可以实现更复杂的布局


        # self.alertButton = Button(self.master, text='确定', command=self.hello)
        # self.alertButton.grid(row=0, column=1,columnspan=1)
        # self.picButton = Button(self.master, text='打开图片', command=self.openpicture)
        # self.picButton.grid(row=0, column=1,columnspan=1)
        # self.piccloseButton = Button(self.master, text='关闭图片', command=self.closepicture)
        # self.piccloseButton.grid(row=0, column=1,columnspan=1)
        # self.piccloseButton = Button(self.master, text='打开新窗口', command=self.openpictureInNew)
        # self.piccloseButton.grid(row=0, column=1,columnspan=1)


        self.imglabel = Label(self.master, bg='brown')

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
            pass
            # print ('Show Picture ERROR')

        # start = time.clock()

        # while True:
        #     if time.clock() == start+10:
        #         self.imglabel.pack_forget()
        #         break
        # time.sleep(10)
        #self.imglabel.pack_forget()
        #self.imglabel.grid()
        #grid_forget将这个组件从grid中移除（并未删除，可以使用grid再将它显示出来)
        # self.imglabel.pack(padx=5, pady=5)


        # self.imglabel.place_forget()


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
        #     print(i)
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

        search_Button = Button(self.window_search, text='查询', command=lambda:self.serch_internet())
        # self.search_Button.pack()
        search_Button.grid(row=1, column=4)
        # label = Label(window, text=id)
        # label.pack(side="top", fill="both", padx=50, pady=40)
        pass
    def serch_internet(self):
        # print('net connection!')
        from app.app_backend.net_search import url_get_string




        url_str = url_get_string()
        # print(url_str)
        url_str = ['1111','222222']

        name_num_list = get_file_list()

        def go(*args):  # 处理事件，*args表示可变参数
            i = comboxlist.get()  # 打印选中的值
            # print(i)
            return i
        from tkinter import ttk
        comvalue = StringVar(self.window_search)  # 窗体自带的文本，新建一个值
        comboxlist = ttk.Combobox(self.window_search, textvariable=comvalue)  # 初始化
        comboxlist["values"] = name_num_list
        comboxlist.current(0)  # 选择第一个
        comboxlist.bind("<<ComboboxSelected>>", go)  # 绑定事件,(下拉列表框被选中时，绑定go()函数)
        comboxlist.grid(row=2, column=4)

        download_Button = Button(self.window_search, text='下载', command=lambda:self.download_pb(selected=go()))
        # self.search_Button.pack()
        download_Button.grid(row=2, column=7)


        pass
    def download_pb(self,selected):
        # print('下载功能')

        # print(selected)
        name_num = selected.split('--')
        name = name_num[0]
        todo_list = name_num[-1]
        url_download(url='http://0.0.0.0:5000/download/admin/'+todo_list+'/loss.csv',dist_path='./data/model/'+name+'.csv')

        self.window_search.destroy()
####################################################################


    def open_pbfile(self):
        id = "New window pb"
        # print(id)
        self.window_pbfile = Toplevel(self.master,)
        self.window_pbfile.title('模型选择')
        # 设置窗口尺寸：‘宽*长+ 横坐标+纵坐标’
        self.window_pbfile.geometry('400x300+500+5')

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
            # print(i)
            return i

        filenames = get_file_list(path='./data/model/')

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
        selected = selected
        # print(selected)

        self.window_pbfile.destroy()
############################################################################################

    def user_login(self):
        get_user_now = get_now_user()

        if get_user_now is not '':
            self.user_now = get_user_now
        else:
            self.user_now = 'admin'

        self.user_var = StringVar()
        self.user_var.set(self.user_now)

        user_show = Label(self.master, text='用户：', font=1)
        user_show.grid(row=0, column=6, rowspan=2, columnspan=1)

        self.user_name_show = Label(self.master, textvariable=self.user_var , font=1)
        self.user_name_show.grid(row=0, column=7, rowspan=2, columnspan=4)

        logoutButton = Button(self.master, text='更换用户', command=self.change_name)
        logoutButton.grid(row=1, column=13, columnspan=1)

        logoutButton = Button(self.master, text='新建本地用户', command=self.creat_user)
        logoutButton.grid(row=1, column=14, columnspan=1)
        pass

    def creat_user(self):

        def creat():

            name = self.newnameInput.get() or 'admin'
            user_dir = 'data/signal/' + name
            result = creat_dir(user_dir)

            if result:
                self.window_adduser.destroy()
            else:

                show_no_user = Label(self.window_adduser, text='用户已经存在', font=1)
                show_no_user.grid(row=4, column=0, rowspan=2, columnspan=2)
                pass
            pass


        id = "New window add name"
        # print(id)
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
            exit_user = get_file_list('./data/signal')
            if name in exit_user:
                modifi_user(name)
                self.user_now = name
                self.user_var.set(self.user_now)
                self.window_changename.destroy()
            else:
                show_no_user = Label(self.window_changename, text='用户未存在', font=1)
                show_no_user.grid(row=4, column=0, rowspan=2, columnspan=2)
                pass
            pass


        id = "New window change name"
        # print(id)
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
    root = Tk() # 初始化Tk()

    app1 = Application(root)

    while True:
        try:
            root.mainloop()
            break
        except UnicodeDecodeError:
            pass


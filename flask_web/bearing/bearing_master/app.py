#!/usr/bin/python
#-*- coding: UTF-8 -*-
from __future__ import unicode_literals


from flask import (Flask, render_template, redirect, url_for, request, flash)
from flask_bootstrap import Bootstrap
from flask_login import login_required, login_user, logout_user, current_user
from flask import send_file, send_from_directory
import os

# try:
#     from utils import unzip_func
#     from forms import TodoListForm, LoginForm, CNNForm, MLForm, CNNSetting, RegistorForm, DataListForm, DataSetting, \
#         Data_Select_Form, All_Set_Form, UploadForm,\
#         ML_Set_Form
#     from ext import db, login_manager
#     from models import TodoList, User, DataList, TrainSettingList
#
#     from dl_function.utils import get_loss
#     from dl_function.read_utils import get_signal_csv, get_all_csv, get_csv_path
#     from dl_function.get_device_info import get_Memo_rate_mome, cpu_core_rate_num, get_available_gpus, get_gpu_used
#     from dl_function.data_analysis import fft
#     from dl_function.feature_extractor import get_mean, get_var, get_abs_mean, get_max, get_min, get_qiaodu
#
#     from model_setting import net_model, Activation_Function, optimizer, ml_model, batch_size, Whether_data_augment, \
#         Net_losses
#
#     from bearing.deep_learning.models_mse_loss.train import trian_function, pb_generation_full, get_pred_result
#
# except:
#     from bearing_master.forms import TodoListForm, LoginForm, CNNForm, MLForm, CNNSetting, RegistorForm, DataListForm, DataSetting, \
#         Data_Select_Form, All_Set_Form, UploadForm, \
#         ML_Set_Form
#     from bearing_master.ext import db, login_manager
#     from bearing_master.models import TodoList, User, DataList, TrainSettingList
#
#     from bearing_master.dl_function.utils import get_loss
#     from bearing_master.dl_function.read_utils import get_signal_csv, get_all_csv, get_csv_path
#     from bearing_master.dl_function.get_device_info import get_Memo_rate_mome, cpu_core_rate_num, get_available_gpus, get_gpu_used
#     from bearing_master.dl_function.data_analysis import fft
#     from bearing_master.dl_function.feature_extractor import get_mean, get_var, get_abs_mean, get_max, get_min, get_qiaodu
#
#     from bearing_master.model_setting import net_model, Activation_Function, optimizer, ml_model, batch_size, Whether_data_augment, \
#         Net_losses
#
#     from deep_learning.models_mse_loss.train import trian_function, pb_generation_full, get_pred_result
#     from bearing_master.utils import unzip_func


from bearing_master.forms import TodoListForm, LoginForm, CNNForm, MLForm, CNNSetting, RegistorForm, DataListForm, DataSetting, \
    Data_Select_Form, All_Set_Form, UploadForm, \
    ML_Set_Form
from bearing_master.ext import db, login_manager
from bearing_master.models import TodoList, User, DataList, TrainSettingList
from bearing_master.dl_function.utils import get_loss
from bearing_master.dl_function.read_utils import get_signal_csv, get_all_csv, get_csv_path
from bearing_master.dl_function.get_device_info import get_Memo_rate_mome, cpu_core_rate_num, get_available_gpus, get_gpu_used
from bearing_master.dl_function.data_analysis import fft
from bearing_master.dl_function.feature_extractor import get_mean, get_var, get_abs_mean, get_max, get_min, get_qiaodu
from bearing_master.model_setting import net_model, Activation_Function, optimizer, ml_model, batch_size, Whether_data_augment, \
    Net_losses
from deep_learning.models_mse_loss.train import trian_function, pb_generation_full, get_pred_result
from bearing_master.utils import unzip_func

import re
import os
import time
import pymysql
import threading
pymysql.install_as_MySQLdb()

SECRET_KEY = 'This is my key'

app = Flask(__name__)
bootstrap = Bootstrap(app)

app.secret_key = SECRET_KEY
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql://test:test12345678@localhost:3306/test"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True


import os
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','csv','xml','zip','py'])
app.config['UPLOAD_FOLDER'] = os.path.realpath('.') + '/bearing_master/data/user_datas/'
app.config['TRAIN_INFO'] = os.path.realpath('.') + '/bearing_master/data/user_info/'

from multiprocessing import Process

# with app.app_context():
#     db.init_app(app)
db.init_app(app)
login_manager.init_app(app)
login_manager.login_view = "login"

DEBUG = True


@app.route('/jump', methods=['GET', 'POST'])
def select():
    print('jump')
    if request.method == 'POST':
        if DEBUG:
            print('jump POST')

        name = request.form['submit']
        if name is 'cnn':
            return 'cnn'
        else:
            return '机器学习'
    form = []
    form.append(CNNForm())
    form.append(MLForm())
    return render_template('select.html', form=form)




@app.route('/view/<int:id>')
@login_required
def view_todo_example(id):
     print('view_todo_example')
     # select_list = [(1, "1星"), (2, "2星"), (3, "3星"), (4, "4星"), (5, "5星")]
     select_list = [net_model, Activation_Function, optimizer, batch_size, ml_model,Whether_data_augment,Net_losses]

     if request.method == 'POST':
         if DEBUG:
             print('jump POST')
         label = request.form['tag_id']
         return select_list[int(label) - 1][1]
     form = []
     form.append(CNNSetting(select_list))
     # todolist = TodoList.query.filter_by(id=id).first_or_404()
     # db.session.delete(todolist)
     # db.session.commit()
     # flash('You are viewing a todo list')
     # return redirect(url_for('show_todo_list'))
     # return redirect(url_for('model_setting'))


     user_name_now = current_user.username
     user_id_now = User.query.filter_by(username=user_name_now).first().id


     todolist = TodoList.query.filter_by(id=id).first_or_404()
     create_time_ = todolist.create_time
     train_list = TrainSettingList.query.filter_by(todolist_id=create_time_).first_or_404()
     ml_class = train_list.ml_model_class

     train_data_list = train_list.data_paths

     train_net_class = train_list.deep_model_class
     # db.session.delete(todolist)
     # db.session.commit()

     train_data_list_split = train_data_list.split('/')

     train_data_str = list(filter(None, train_data_list_split))
     train_data_int = []
     for s in train_data_str:
         train_data_int.append(int(s))
     print(train_data_int)

     datalists = DataList.query.filter(DataList.user_id==user_id_now,DataList.create_time.in_(train_data_int))
     dataform = DataListForm()


     current_user_id = str(current_user.username)
     filepath = os.path.join(app.config['TRAIN_INFO'], current_user_id)

     model_creat_time = str(create_time_) #create_time_id
     filepath = os.path.join(filepath, model_creat_time)
     filename = os.path.join(filepath, 'loss.csv')

     loss = get_loss(filename)
     x_axis = [i for i in range(len(loss[0]))]

     # 网络类型
     # net_model = ['CNN', 'MobileNet', 'ResNet', 'FCNet', 'VAE', 'Auto_encoder']
     # 激活函数
     # Activation_Function = ['sigmod', 'tanh', 'ReLU', 'ELU', 'PReLU', 'Leaky ReLU']
     # 优化器
     # optimizer = ['Adam', 'SGD', 'BGD', 'Adagrad', 'Adadelta', 'RMSprop']
     # 损失函数
     # net_losses = ['mse', 'cross_entropy', 'combin_loss', 'exponential_Loss', 'hinge_loss']
     # 机器学习模型
     # ml_model = ['SVM', 'DT', 'Gauss']
     # 模型类别 分为两种
     # model_class = ['Deep Learning', 'Machine Learning']
     # select_list = [GPU_setting,whether_data_augment,deep_model_class,
     #                ml_model_class,input_dim,output_dim,weight_decay,
     #             learning_rate,activation_class,layers_num]
     train_activation_class = train_list.activation_class
     train_input_dim_class = train_list.input_dim
     train_output_dim_class = train_list.output_dim

     train_weight_decay_class = train_list.weight_decay
     train_learning_rate_class = train_list.learning_rate
     train_layers_num_class = train_list.layers_num
     train_GPU_setting_class = train_list.GPU_setting
     train_output_dim_class = train_list.output_dim

     trian_dataAug = train_list.whether_data_augment

     trian_loss = train_list.net_losses
     trian_optimizer = train_list.optimizer

     train_ml_model_class = train_list.ml_model_class
     if train_ml_model_class>=0:
         model_classes='机器学习模型'
     else:
         model_classes = '深度模型'
     model_setting=[
         ['模型类别:', model_classes],
         ['模型名称:', select_list[0][train_net_class]],
         ['激活函数:',select_list[1][train_activation_class]],
         ['优化器:',select_list[2][trian_optimizer]],
         ['损失函数:',select_list[6][trian_loss]],
         ['模型名称:', select_list[4][train_ml_model_class]],#
         ['输入维度:', train_input_dim_class],
         ['输出维度:', train_output_dim_class],
         ['权重衰减:', train_weight_decay_class],
         ['学习率:', train_learning_rate_class],
         ['GPU设置:',train_GPU_setting_class],
         ['网络层数:',train_layers_num_class ],
         ['是否数据增强:', select_list[5][trian_dataAug]],
     ] #模型设置结果

     memory_rate, memory_total = get_Memo_rate_mome()
     cpu_rate, num_core = cpu_core_rate_num()
     # gpu_nums = get_available_gpus()
     gpu_nums = 0


     # gpu_memo_useds = []
     # for i in range(gpu_nums):
     #    gpu_memo_used = get_gpu_used(i)
     #    gpu_memo_useds.append(gpu_memo_used)
     # print('gpu_memo_useds',gpu_memo_useds)
     gpu_memo_useds = 0
     gpu_memo_totals=0
     # for i in range(gpu_nums):
     #     get_gpu_one_used,get_gpu_one_total= get_gpu_used(i)
     #     gpu_memo_useds += get_gpu_one_used
     #     gpu_memo_totals += get_gpu_one_total
     print('gpu_memo_useds',gpu_memo_useds)


     if gpu_nums == 0:
         gpu_nums = 1

     gpu_num_used  = train_list.GPU_setting

     device_info = [memory_rate,memory_total/1024/1024/1024,
                    int(cpu_rate), num_core,
                    # gpu_nums,gpu_num_used,gpu_memo_useds
                    # gpu_memo_totals, gpu_num_used, gpu_memo_totals,gpu_memo_useds
                    ]
     # print('device_info',device_info)

     user_name_now = current_user.username
     user_id_now = User.query.filter_by(username=user_name_now).first().id
     # todolists = TodoList.query.filter_by(user_id=user_id_now)
     all_datalists = DataList.query.filter_by(user_id=user_id_now)


     if ml_class!=-1:
         return render_template('train_mlinfo.html',device_info=device_info,
                                model_setting=model_setting,
                                x_axis=x_axis, dataform=dataform, datalists=datalists)
     else:
         return render_template('train_info.html',device_info=device_info,
                                model_setting=model_setting, u_data=loss,
                                x_axis=x_axis, dataform=dataform, datalists=datalists,
                                all_datalists=all_datalists,model_id=id)


@app.route('/view_data/<int:id>')
@login_required
def view_data_example(id):
     print('view_data_example')
     select_list = [(2, "2星")]
     if request.method == 'POST':
         # if DEBUG:
         #     print('jump POST')
         label = request.form['tag_id']
         return select_list[int(label) - 1][1]
     form = []
     form.append(DataSetting(select_list))
     # todolist = TodoList.query.filter_by(id=id).first_or_404()
     # db.session.delete(todolist)
     # db.session.commit()
     # flash('You are viewing a todo list')
     # return redirect(url_for('show_todo_list'))
     # return redirect(url_for('model_setting'))
     # loss = get_loss()

     datalist = DataList.query.filter_by(create_time=id).first_or_404()
     file_name = str(datalist.create_time)

     filepath = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
     # print(filepath)
     filepath = get_csv_path(filepath)
     print(filepath)
     name_lists = sorted(os.listdir(filepath))
     print(name_lists[-1])
     # filepath_name = os.path.join(filepath, name_lists[0])
     filepath_name = os.path.join(filepath, name_lists[-1])
     print(filepath,filepath_name)

     # signal = get_signal_csv(file_name='data/aaaa/Bearing1_1/acc_00001.csv')
     signal = get_signal_csv(file_name=filepath_name)
     # signal = get_all_csv(path='data/aaaa/Bearing1_1/')
     x_axis = [i for i in range(len(signal))]

     frequency = fft(signal)[0:len(x_axis)//2].tolist()
     x_axis_f = x_axis[0:len(signal)//2]

     # get_mean, get_var, get_abs_mean, get_max, get_min, get_qiaodu
     s_mean = get_mean(signal)
     s_var = get_var(signal)
     s_abs_mean = get_abs_mean(signal)
     s_max = get_max(signal)
     s_min = get_min(signal)
     s_qiaodu = get_qiaodu(signal)
     data_analysis=[
         ['均值:', round(s_mean,5)],
         ['方差:',round(s_var,5)],
         ['绝对值均值:',round(s_abs_mean,5)],
         ['最大值:',round(s_max,5)],
         ['最小值:',round(s_min,5)],
         ['峭度:', round(s_qiaodu,5)]
         # [':', ],
         # [':', ],
         # [':', ],
         # [':', ],

     ] #模型设置结果

     return render_template('data_info.html',data_analysis=data_analysis,x_axis=x_axis, u_data=signal,x_axis_f=x_axis_f,frequency=frequency)
import numpy as np
def liner_regre(predict_list):
    if len(predict_list) > 500:
        predict_list = predict_list[-500:]
    if len(predict_list)>1:
        x = [[i] for i in range(len(predict_list))]
        linreg = linear_model.LinearRegression()
        linreg.fit(x,predict_list)
        result = linreg.predict([x[-1]])
    else:
        result = np.array([predict_list[-1]])
    return result
    pass
@app.route('/view_data_prediction/<int:id>/<int:model_id>')
@login_required
def view_data_prediction(id,model_id):
     print('model_id:',model_id)
     print('view_data_prediction')
     select_list = [(2, "2星")]
     if request.method == 'POST':
         if DEBUG:
             print('jump POST')
         label = request.form['tag_id']
         return select_list[int(label) - 1][1]
     form = []
     form.append(DataSetting(select_list))
     # todolist = TodoList.query.filter_by(id=id).first_or_404()
     # db.session.delete(todolist)
     # db.session.commit()
     # flash('You are viewing a todo list')
     # return redirect(url_for('show_todo_list'))
     # return redirect(url_for('model_setting'))
     # loss = get_loss()

     datalist = DataList.query.filter_by(create_time=id).first_or_404()
     file_name = str(datalist.create_time)

     filepath = os.path.join(app.config['UPLOAD_FOLDER'], file_name)

     filename_list = get_csv_path(filepath)
     filename_list = sorted(os.listdir(filename_list))
     # filepath = os.path.join(filepath, name_lists[0])

     # signal = get_signal_csv(file_name='data/aaaa/Bearing1_1/acc_00001.csv')
     # signal = get_signal_csv(file_name=filepath)


     #################################
     user_name_now = current_user.username
     user_id_now = User.query.filter_by(username=user_name_now).first().id
     todolist = TodoList.query.filter_by(id=model_id).first_or_404()
     create_time_ = todolist.create_time
     train_list = TrainSettingList.query.filter_by(todolist_id=create_time_).first_or_404()

     current_user_name = str(current_user.username)
     # datalist = DataList.query.filter_by(create_time=id).first_or_404()
     file_name = str(datalist.create_time)
     id_V = model_id
     todolist_v = TodoList.query.filter_by(id=id_V).first_or_404()
     trained_model_path = os.path.join(filepath, 'trained_model')
     # is_finished = trian_function(TodoList, id_V,trained_model_path)
     train_setting_list = TrainSettingList.query.filter_by(todolist_id=create_time_).first_or_404()
     ############################################

     modelpath = os.path.join(app.config['TRAIN_INFO'],current_user_name)
     modelpath = os.path.join(modelpath,  str(create_time_))

     trained_model_path = os.path.join(modelpath, 'trained_model')
     print(trained_model_path)

     signal = get_pred_result(filepath=filepath,filename_list=filename_list,
                              todolist=todolist,train_setting_list=train_setting_list,trained_model_path = trained_model_path)
     real_result = [i for i in range(len(signal)-1,-1,-1)]
     err = np.sqrt(np.average(np.square(np.subtract(signal,real_result))))

     # print(err)
     # signal = get_all_csv(path='data/aaaa/Bearing1_1/')
     x_axis = [i for i in range(len(signal))]

     frequency = []
     for ii in range(len(signal)):
         fre = liner_regre(signal[0:ii+1]).tolist()
         # print(fre)
         frequency.append(fre[0])
     # print(frequency)
     x_axis_f = [i for i in range(len(frequency))]

     # get_mean, get_var, get_abs_mean, get_max, get_min, get_qiaodu
     s_mean = get_mean(signal)
     s_var = get_var(signal)
     s_abs_mean = get_abs_mean(signal)
     s_max = get_max(signal)
     s_min = get_min(signal)
     s_qiaodu = get_qiaodu(signal)
     data_analysis=[
         ['均方根误差:', round(err,5)],
         ['方差:',round(s_var,5)],
         # ['绝对值均值:',round(s_abs_mean,5)],
         # ['最大值:',round(s_max,5)],
         # ['最小值:',round(s_min,5)],
         # ['峭度:', round(s_qiaodu,5)]

     ] #模型设置结果
     return render_template('data_pred_info.html',data_analysis=data_analysis,x_axis=x_axis, u_data=signal,x_axis_f=x_axis_f,frequency=frequency)
from sklearn import linear_model



@app.route('/register', methods=['GET', 'POST'])
def register():
    print('register')
    if request.method == 'POST':
        if DEBUG:
            print('POST')
        # user = User.query.filter_by(username=request.form['username'], password=request.form['password']).first()
        form = RegistorForm()
        if form.validate_on_submit():
            user = User(username=request.form['username'], password=request.form['password'])
            db.session.add(user)
            db.session.commit()
            return redirect(url_for('login'))
        else:
            flash('Invalid username or password')
    form = RegistorForm()
    return render_template('register.html', form=form)



@app.route('/model_setting/<int:id>', methods=['GET', 'POST'])
def model_setting(id):
    print('modelsetting')
    # select_list = [(1, "DNN"), (2, "CNN"), (3, "MobileNet"), (4, "ResNet"), (5, "RNN")]
    # net_model, Activation_Function, optimizer, ml_model,batch_size
    select_list = [net_model, Activation_Function, optimizer, batch_size, Whether_data_augment,Net_losses]

    user_name_now = current_user.username
    user_id_now = User.query.filter_by(username=user_name_now).first().id
    datalists = DataList.query.filter_by(user_id = user_id_now)

    todolist = TodoList.query.filter_by(id = id).first()

    counter = 0
    for data in datalists:
        counter += 1
    allform= All_Set_Form([counter,select_list])()
    # form = []
    # form.append(CNNSetting(select_list))
    if request.method == 'POST':
        if DEBUG:
            print('jump POST')

        # label = request.form['tag_id']
        # hobby = flask.request.form.getlist('dataset')
        # selected_dataset = request.form['tag_id18']
        forms = request.form

        file_names = ''
        for out in forms:
            if 'tag_id_dym' in out:
                print(out)
                count = int(out[10:])

                datalist = datalists[count]
                file_name = '/' + str(datalist.create_time)

                file_names += file_name

        # 如果以 todolist的create_time能找到TrainSettingList的数据，则删除重建
        exit_trainsettings = TrainSettingList.query.filter_by(todolist_id=todolist.create_time).all()
        [db.session.delete(exit_trainsetting) for exit_trainsetting in exit_trainsettings]
        db.session.commit()

        whether_data_augment = int(forms['tag_id04'])
        deep_model_class = int(forms['tag_id00'])
        ml_model_class = -1
        input_dim = int(forms['tag_id013'])

        # output_dim = int(forms['tag_id014'])
        output_dim = 1

        weight_decay = float(forms['tag_id017'])
        learning_rate = float(forms['tag_id016'])
        activation_class = int(forms['tag_id01'])
        layers_num = int(forms['tag_id015'])


        optimizer_class = int(forms['tag_id02'])
        batch_size_num = int(forms['tag_id03'])

        net_losses = int(forms['tag_id05'])


        trainsettinglist = TrainSettingList(
            user_id=current_user.id, create_time=time.time(), model_class=0,
            GPU_setting=0, data_paths=file_names, todolist_id=todolist.create_time,
            whether_data_augment=whether_data_augment, deep_model_class=deep_model_class,
            ml_model_class=ml_model_class, input_dim=input_dim, output_dim=output_dim, weight_decay=weight_decay,
            learning_rate=learning_rate, activation_class=activation_class, layers_num=layers_num,batch_size=batch_size_num,
            optimizer=optimizer_class, net_losses=net_losses
        )
        db.session.add(trainsettinglist)
        db.session.commit()

        return redirect(url_for('view_todo_example', id=id))
    user_name_now = current_user.username
    user_id_now = User.query.filter_by(username=user_name_now).first().id
    todolists = TodoList.query.filter_by(user_id=user_id_now)
    datalists = DataList.query.filter_by(user_id=user_id_now)

    return render_template('modelsetting.html', allform=allform,datalists=datalists)




@app.route('/ml_setting/<int:id>', methods=['GET', 'POST'])
def ml_setting(id):
    print('ml_setting')
    # select_list = [(1, "DNN"), (2, "CNN"), (3, "MobileNet"), (4, "ResNet"), (5, "RNN")]
    # net_model, Activation_Function, optimizer, ml_model,batch_size
    select_list = [ml_model, Activation_Function, optimizer, batch_size,ml_model]


    user_name_now = current_user.username
    user_id_now = User.query.filter_by(username=user_name_now).first().id
    datalists = DataList.query.filter_by(user_id = user_id_now)

    todolist = TodoList.query.filter_by(id = id).first()


    counter = 0
    for data in datalists:
        counter += 1
    allform= ML_Set_Form([counter,select_list])()
    # form = []
    # form.append(CNNSetting(select_list))
    if request.method == 'POST':
        if DEBUG:
            print('jump POST')
        # label = request.form['tag_id']
        # hobby = flask.request.form.getlist('dataset')
        # selected_dataset = request.form['tag_id18']
        forms = request.form
        file_names = ''
        for out in forms:
            if 'tag_id_dym' in out:
                print(out)
                count = int(out[10:])

                datalist = datalists[count]
                file_name = '/' + str(datalist.create_time)

                file_names += file_name
        # 如果以 todolist的create_time能找到TrainSettingList的数据，则删除重建
        exit_trainsettings = TrainSettingList.query.filter_by(todolist_id=todolist.create_time).all()
        [db.session.delete(exit_trainsetting) for exit_trainsetting in exit_trainsettings]
        db.session.commit()
        deep_model_class = -1
        ml_model_class = int(forms['tag_id00'])
        input_dim = int(forms['tag_id013'])

        # output_dim = int(forms['tag_id014'])
        output_dim = 1
        weight_decay = float(forms['tag_id017'])
        learning_rate = float(forms['tag_id016'])
        activation_class = -1
        whether_data_augment = -1
        layers_num = -1
        optimizer_class = -1
        batch_size_num = -1

        # whether_data_augment = int(forms['tag_id04'])
        # layers_num = int(forms['tag_id015'])
        # optimizer_class = int(forms['tag_id02'])
        # batch_size_num = int(forms['tag_id03'])


        trainsettinglist = TrainSettingList(
            user_id=current_user.id, create_time=time.time(), model_class=0,
            GPU_setting=0, data_paths=file_names, todolist_id=todolist.create_time,
            whether_data_augment=whether_data_augment, deep_model_class=deep_model_class,
            ml_model_class=ml_model_class, input_dim=input_dim, output_dim=output_dim, weight_decay=weight_decay,
            learning_rate=learning_rate, activation_class=activation_class, layers_num=layers_num,batch_size=batch_size_num,
            optimizer=-1, net_losses=-1)
        # trainsettinglist = TrainSettingList(
        #     user_id=current_user.id, create_time=time.time(), model_class=0,
        #     GPU_setting=0, data_paths=file_names, todolist_id=todolist.create_time)
        db.session.add(trainsettinglist)
        db.session.commit()

        return redirect(url_for('view_todo_example', id=id))
    user_name_now = current_user.username
    user_id_now = User.query.filter_by(username=user_name_now).first().id
    todolists = TodoList.query.filter_by(user_id=user_id_now)
    datalists = DataList.query.filter_by(user_id=user_id_now)

    return render_template('mlsetting.html', allform=allform,datalists=datalists)








@app.route('/data_selecting', methods=['GET', 'POST'])
@login_required
def data_select():
    print('data_select')
    # todolists = TodoList.query.all()
    user_name_now = current_user.username
    user_id_now = User.query.filter_by(username=user_name_now).first().id
    datalists = DataList.query.filter_by(user_id = user_id_now)

    counter = 0
    for data in datalists:
        counter += 1
    dataform = Data_Select_Form([counter])()
    if request.method == 'GET':

        return render_template('dataselecting.html', datalists=datalists, dataform=dataform)
    else:

        # if dataform.validate_on_submit():
        if True:
            # datalist = DataList(current_user.id, dataform.title.data, 0)
            # db.session.add(datalist)
            # db.session.commit()

            # for i in range(counter):
            #     name = 'tag_id'+str(i)
            #     print(name)
            get_outs = request.form

            # selected_hold = []
            file_names = ''
            for out in get_outs:
                if 'tag' in out:
                    print(out)
                    count = int(out[6:])

                    datalist = datalists[count]
                    file_name = '/' + str(datalist.create_time)

                    file_names += file_name
                    # selected_hold.append(count)
            print(file_names)

            return 'hellow'

            # flash('You have selected the datalists')
            # return render_template('data_uploading.html')
        else:
            flash(dataform.errors)
            return redirect(url_for('model_setting'))






def allowed_file(filename):
    return '.' in filename and filename.lower().rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/data_uploading/<int:id>', methods=['GET', 'POST'])
def data_uploading(id):
    uploadform = UploadForm()

    if uploadform.validate_on_submit():
        # name = uploadform.name.data
        # price = uploadform.price.data
        image = request.files['image']
        file_name = image.filename

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], str(id))

        if image and allowed_file(image.filename):
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            image.save(os.path.join(app.config['UPLOAD_FOLDER'], str(id), file_name))
        full_file_name = os.path.join(app.config['UPLOAD_FOLDER'], str(id), file_name)
        unzip_path = os.path.join(app.config['UPLOAD_FOLDER'], str(id))
        unzip_func(full_file_name,unzip_path)

        datalist = DataList.query.filter_by(create_time=id).first_or_404()
        datalist.status = 1
        db.session.add(datalist)
        db.session.commit()

        # product = Product(name, price, category, filename)
        # db.session.add(product)
        # db.session.commit()
        flash('The dataset has been uploaded!')
        return redirect(url_for('view_data_example',id=id))

    if uploadform.errors:
        flash(uploadform.errors, 'danger')

    print('data_uploading.html')

    return render_template('data_uploading.html', uploadform=uploadform)



@app.route('/edge_uploading/<int:id>/<string:user_name>', methods=['GET', 'POST'])
def edge_uploading(id,user_name):
    uploadform = UploadForm()

    if True:
        # name = uploadform.name.data
        # price = uploadform.price.data
        image = request.files['image']
        file_name = image.filename
        file_name_split = file_name.split('/')
        file_name_split = file_name_split[-1].split('\\')
        file_name_time_id = file_name_split[-1]
        file_uper_dir = file_name_split[-2]
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], str(id),file_uper_dir)

        if image and allowed_file(image.filename):
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            try:
                datalist = DataList.query.filter_by(create_time=id).first_or_404()
            except:
                user_id_now = User.query.filter_by(username=user_name).first().id
                datalist = DataList(user_id_now, file_uper_dir+'_from_edge', 0,creat_time=id)
                db.session.add(datalist)
                db.session.commit()
            final_save_path = os.path.join(file_path, file_name_time_id)
            image.save(final_save_path)

            file_name = str(datalist.create_time)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
            filepath = get_csv_path(filepath)

            name_lists = sorted(os.listdir(filepath))
            filepath = os.path.join(filepath, name_lists[-1])
            signal = get_signal_csv(file_name=filepath)
            max = get_max(signal)
            if max > 20:
                datalist.status = 1
                db.session.add(datalist)
                db.session.commit()

        # full_file_name = os.path.join(app.config['UPLOAD_FOLDER'], str(id), file_name)
        # unzip_path = os.path.join(app.config['UPLOAD_FOLDER'], str(id))
        # unzip_func(full_file_name,unzip_path)

        # product = Product(name, price, category, filename)
        # db.session.add(product)
        # db.session.commit()
        print('The dataset has been uploaded!')
        return 'saved!'
        # return redirect(url_for('view_data_example',id=id))

    # if uploadform.errors:
    #     flash(uploadform.errors, 'danger')

    # print('data_uploading.html')

    # return render_template('data_uploading.html', uploadform=uploadform)






##################################################################
import json

@app.route('/', methods=['GET', 'POST'])
@login_required
def show_todo_list():
    print('show_todo_list')
    form = TodoListForm()
    dataform= DataListForm()
    if request.method == 'GET':
        print('GET')
        # todolists = TodoList.query.all()
        user_name_now = current_user.username
        user_id_now = User.query.filter_by(username=user_name_now).first().id
        todolists = TodoList.query.filter_by(user_id = user_id_now)
        datalists = DataList.query.filter_by(user_id = user_id_now)

        return render_template('index.html', todolists=todolists, datalists=datalists, form=form, dataform=dataform)
    else:
        if form.validate_on_submit():
            # todolist = TodoList(current_user.id, form.title.data, form.status.data)
            todolist = TodoList(current_user.id, form.title.data, 0)

            model_class = form.status.data

            db.session.add(todolist)
            db.session.commit()
            id = todolist.id

            ## add new train file loss.csv

            current_user_name = str(current_user.username)
            filepath = os.path.join(app.config['TRAIN_INFO'], current_user_name)

            model_creat_time = str(todolist.create_time)  # create_time_id

            filepath = os.path.join(filepath, model_creat_time)

            if not os.path.exists(filepath):
                os.makedirs(filepath)
                print('创建路径：', filepath)

            filename = os.path.join(filepath, 'loss.csv')
            with open(filename, 'w') as f:
                pass
                f.close()

            filename2 = os.path.join(filepath, 'is_training.json')
            with open(filename2, "w") as f:
                json.dump([0], f)
            flash('You have add a new task to list')
            if model_class == '1':
                return redirect(url_for('model_setting', id=id))
            else:
                return redirect(url_for('ml_setting', id=id))

        elif dataform.validate_on_submit():
            # todolist = TodoList(current_user.id, form.title.data, form.status.data)
            datalist = DataList(current_user.id, dataform.title.data, 0)
            db.session.add(datalist)
            db.session.commit()
            create_time = datalist.create_time
            flash('You have add a new data to datalist')

            return redirect(url_for('data_uploading', id=create_time))

            # uploadform = UploadForm()
            # return render_template('data_uploading.html',uploadform=uploadform)
        else:
            flash(form.errors)
            return redirect(url_for('model_setting'))


# def show_todo_list():
#     print('show_todo_list')
#     form = TodoListForm()
#     if request.method == 'GET':
#         todolists = TodoList.query.all()
#         return render_template('index.html', todolists=todolists, form=form)
#     else:
#         if form.validate_on_submit():
#             todolist = TodoList(current_user.id, form.title.data, form.status.data)
#             db.session.add(todolist)
#             db.session.commit()
#             flash('You have add a new todo list')
#         else:
#             flash(form.errors)
#         return redirect(url_for('show_todo_list'))


@app.route('/delete/<int:id>')
@login_required
def delete_todo_list(id):
     print('delete_todo_list')
     todolist = TodoList.query.filter_by(id=id).first_or_404()
     db.session.delete(todolist)
     db.session.commit()
     flash('You have delete a todo list')
     return redirect(url_for('show_todo_list'))

@app.route('/delete_data/<int:id>')
@login_required
def delete_data_list(id):
     print('delete_data_list')
     datalist = DataList.query.filter_by(id=id).first_or_404()
     db.session.delete(datalist)
     db.session.commit()
     flash('You have delete a data list')
     return redirect(url_for('show_todo_list'))

@app.route('/finished_data_example/<int:id>')
@login_required
def finished_data_example(id):
     print('finish_data_list')
     # datalist = DataList.query.filter_by(id=id).first_or_404()

     datalist = DataList.query.filter_by(create_time=id).first_or_404()

     file_name = str(datalist.create_time)
     filepath = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
     filepath = get_csv_path(filepath)
     name_lists = sorted(os.listdir(filepath))
     filepath_name = os.path.join(filepath, name_lists[-1])
     signal = get_signal_csv(file_name=filepath_name)
     max_v = max(signal)
     min_v = min(signal)
     if max_v>20 or min_v<-20:
         datalist.status = 1
         db.session.add(datalist)
         db.session.commit()
         flash('You have set a data list finished')
         return redirect(url_for('show_todo_list'))
     else:
         flash('The data list unfinished')
         return redirect(url_for('show_todo_list'))
@app.route('/finished_set/<int:id>')
@login_required
def finished_set(id):
     print('finish_data_list')
     # datalist = DataList.query.filter_by(id=id).first_or_404()

     datalist = DataList.query.filter_by(create_time=id).first_or_404()
     datalist.status = 1
     db.session.add(datalist)
     db.session.commit()
     flash('You have set a data list finished')
     return redirect(url_for('show_todo_list'))

@app.route('/unfinished_set/<int:id>')
@login_required
def unfinished_set(id):
     print('unfinished_set')
     # datalist = DataList.query.filter_by(id=id).first_or_404()

     datalist = DataList.query.filter_by(create_time=id).first_or_404()
     datalist.status = 0
     db.session.add(datalist)
     db.session.commit()
     flash('You have set a data list finished')
     return redirect(url_for('show_todo_list'))
@app.route('/change/<int:id>', methods=['GET', 'POST'])
@login_required
def change_todo_list(id):
    print('change_todo_list')
    if request.method == 'GET':
        todolist = TodoList.query.filter_by(id=id).first_or_404()
        form = TodoListForm()
        form.title.data = todolist.title
        form.status.data = str(todolist.status)
        return render_template('modify.html', form=form)
    else:
        form = TodoListForm()
        if form.validate_on_submit():
            todolist = TodoList.query.filter_by(id=id).first_or_404()
            todolist.title = form.title.data
            todolist.status = form.status.data
            db.session.commit()
            flash('You have modify a todolist')
        else:
            flash(form.errors)
        return redirect(url_for('show_todo_list'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    print('login')
    if request.method == 'POST':
        if DEBUG:
            print('POST')
        user = User.query.filter_by(username=request.form['username'], password=request.form['password']).first()
        if user:
            login_user(user)
            flash('you have logged in!')
            return redirect(url_for('show_todo_list'))
        else:
            flash('Invalid username or password')
    form = LoginForm()
    return render_template('login.html', form=form)



@app.route('/logout')
@login_required
def logout():
    print('logout')
    logout_user()
    flash('you have logout!')
    return redirect(url_for('login'))


@login_manager.user_loader
def load_user(user_id):
    print('load_user')
    return User.query.filter_by(id=int(user_id)).first()


#################################################################


@app.route("/getinputdim/<username>/<task_id>", methods=['GET'])
def get_input_dim(username,task_id):
    # 需要知道2个参数, 第1个参数是本地目录的path, 第2个参数是文件名(带扩展名)
    # directory = os.getcwd()  # 假设在当前目录
    directory = app.config['TRAIN_INFO']+'/'+username+'/'+task_id+'/trained_model/output/'
    filename_list = os.listdir(directory)
    train_list = TrainSettingList.query.filter_by(todolist_id=task_id).first_or_404()
    input_dim = str(train_list.input_dim)

    return input_dim
@app.route("/download/<username>/<task_id>/<filename>", methods=['GET'])
def download_file(username,task_id,filename):
    # 需要知道2个参数, 第1个参数是本地目录的path, 第2个参数是文件名(带扩展名)
    # directory = os.getcwd()  # 假设在当前目录
    directory = app.config['TRAIN_INFO']+'/'+username+'/'+task_id+'/trained_model/output/'
    print(directory)
    filename_list = os.listdir(directory)

    train_list = TrainSettingList.query.filter_by(todolist_id=task_id).first_or_404()
    input_dim = train_list.input_dim
    selected_filename = filename
    for f_name in filename_list:
        if filename in f_name:
            selected_filename = f_name
            break
    print(selected_filename)
    return send_from_directory(directory, selected_filename, as_attachment=True)

@app.route("/train_task/<username>", methods=['Post'])
def train_task(username):
    user = User.query.filter_by(username=username).first()

    user_id = user.id
    todo_list = TodoList.query.filter_by(user_id=user_id,status=1).all()

    train_names = [(todo.title, todo.create_time) for todo in todo_list]
    all_name = ''
    all_time = ''
    for name in train_names:
        all_name = all_name + '///' + name[0]
        all_time = all_time + '///' + str(name[1])
    print('url get:',all_name+ '@@' +all_time)
    return all_name+ '@@' +all_time


@app.route("/return_list", methods=['Post'])
def return_list():
    # 需要知道2个参数, 第1个参数是本地目录的path, 第2个参数是文件名(带扩展名)
    # directory = os.getcwd()  # 假设在当前目录
    directory = 'data/user_info/admin/1535337539/'
    a = [11111,2,3,2,1,'1212']
    b = str(a)
    return b




@app.route('/begin_train/<int:id>')
@login_required
def begin_train_todo_list(id):
     print('begin_train_todo_list')
     todolist = TodoList.query.filter_by(id=id).first_or_404()

     # id = todolist.id


     ## add new train file loss.csv

     current_user_name = str(current_user.username)
     filepath = os.path.join(app.config['TRAIN_INFO'], current_user_name)

     model_creat_time = str(todolist.create_time)  # create_time_id

     filepath = os.path.join(filepath, model_creat_time)

     if not os.path.exists(filepath):
         os.makedirs(filepath)
         print('创建路径：', filepath)
     filename = os.path.join(filepath, 'is_training.json')
     with open(filename, 'r') as load_f:
         load_dict = json.load(load_f)
     is_training = load_dict[0]
     is_training = int(todolist.is_training)
     if is_training == 0:
         with open(filename, "w") as f:
             json.dump([1], f)
         todolist.is_training = 1
         db.session.commit()

         # return redirect(url_for('model_setting', id=id))

         def train_thread(id,filename):
             # threads = []

             # t1 = Process(target=begin_train, args=(10101,id,filename))
             # threads.append(t1)  # 将这个子线程添加到线程列表中
             # for t in threads:  # 遍历线程列表
             #     t.setDaemon(True)  # 将线程声
             #     t.start()  # 启动子线程
             # t1 = threading.Thread(target=begin_train, args=(10101,id,filename))
             t1 = Process(target=begin_train, args=(10101, id, filename))
             t1.daemon = True
             # t1.setDaemon(True)  # 将线程声
             t1.start()  # 启动子线程

         def begin_train(input,id_V,filename):

             user_name_now = current_user.username
             user_id_now = User.query.filter_by(username=user_name_now).first().id
             todolist = TodoList.query.filter_by(id=id).first_or_404()
             create_time_ = todolist.create_time
             train_list = TrainSettingList.query.filter_by(todolist_id=create_time_).first_or_404()
             train_data_list = train_list.data_paths

             train_data_list_split = train_data_list.split('/')
             train_data_str = list(filter(None, train_data_list_split))
             train_data_int = []
             for s in train_data_str:
                 train_data_int.append(int(s))
             print(train_data_int)
             #######################################################################
             datalists = DataList.query.filter(DataList.user_id == user_id_now,
                                               DataList.create_time.in_(train_data_int))
             # datalists = DataList.query.filter(DataList.user_id == user_id_now)
             ############################################################################
             current_user_id = str(current_user.username)
             filepath = os.path.join(app.config['TRAIN_INFO'], current_user_id)

             model_creat_time = str(create_time_)  # create_time_id
             filepath = os.path.join(filepath, model_creat_time)
             loss_filename = os.path.join(filepath, 'loss.csv')


             todolist_v = TodoList.query.filter_by(id=id_V).first_or_404()
             is_training_v = 1
             is_training_filename = filename
             trained_model_path = os.path.join(filepath, 'trained_model')
             is_finished = trian_function(is_training_v,is_training_filename,
                            TodoList,id_V,datalists,loss_filename,
                            trained_model_path,train_setting_list=train_list
                            )
             db.session.commit()
            ################################################

             current_user_name = str(current_user.username)
             filepath = os.path.join(app.config['TRAIN_INFO'], current_user_name)
             model_creat_time = str(todolist.create_time)  # create_time_id
             filepath = os.path.join(filepath, model_creat_time)
             if not os.path.exists(filepath):
                 os.makedirs(filepath)
                 print('创建路径：', filepath)
             json_filename = os.path.join(filepath, 'is_training.json')
             with open(json_filename, 'r') as load_f:
                 load_dict = json.load(load_f)

             # if int(todolist.is_training) == 1:
             #     with open(json_filename, "w") as f:
             #         json.dump([0], f)
             #     todolist.is_training = 0
             #     db.session.commit()
             #     flash('You have finish the task')

             print('is_finished:',is_finished)
             if  is_finished == 1:
                 with open(json_filename, "w") as f:
                     json.dump([0], f)
                 todolist.is_training = 0
                 db.session.commit()
                 flash('You have finish the task')
             pass

         train_thread(id,filename)
         flash('You have begin the task')
     else:
         flash('You have begin the task before')
     return redirect(url_for('show_todo_list'))


@app.route('/stop_train/<int:id>')
@login_required
def stop_train_todo_list(id):
    print('stop_train_todo_list')
    todolist = TodoList.query.filter_by(id=id).first_or_404()

    id = todolist.id

    ## add new train file loss.csv

    current_user_name = str(current_user.username)
    filepath = os.path.join(app.config['TRAIN_INFO'], current_user_name)

    model_creat_time = str(todolist.create_time)  # create_time_id

    filepath = os.path.join(filepath, model_creat_time)

    if not os.path.exists(filepath):
        os.makedirs(filepath)
        print('创建路径：', filepath)

    filename = os.path.join(filepath, 'is_training.json')
    with open(filename, 'r') as load_f:
        load_dict = json.load(load_f)
        is_training = load_dict[0]

    if int(todolist.is_training)==1:
        with open(filename, "w") as f:
            json.dump([0], f)
        todolist.is_training = 0
        db.session.commit()
        flash('You have stop the task')
    else:
        flash('You have stop the task before!')
    # return redirect(url_for('model_setting', id=id))

    # flash('You have delete a todo list')
    return redirect(url_for('show_todo_list'))


@app.route('/generate_pb/<int:id>')
@login_required
def generate_pb(id):
     print('generate_pb')
     todolist = TodoList.query.filter_by(id=id).first_or_404()

     ## add new train file loss.csv

     current_user_name = str(current_user.username)
     filepath = os.path.join(app.config['TRAIN_INFO'], current_user_name)

     model_creat_time = str(todolist.create_time)  # create_time_id
     create_time_ = todolist.create_time
     train_list = TrainSettingList.query.filter_by(todolist_id=create_time_).first_or_404()
     ml_class = train_list.ml_model_class

     input_dim = train_list.input_dim
     filepath = os.path.join(filepath, model_creat_time)
     # filename = os.path.join(filepath, 'trained_model')
     #################################################################################
     todolist = TodoList.query.filter_by(id=id).first_or_404()
     create_time_ = todolist.create_time
     train_setting_list = TrainSettingList.query.filter_by(todolist_id=create_time_).first_or_404()
     #################################################################################

     # def pb_generation_thread(filepath,input_dim,train_setting_list):
     #     t1 = Process(target=pb_generation_full, args=(filepath,input_dim,train_setting_list))
     #     t1.daemon = True
     #     # t1.setDaemon(True)  # 将线程声
     #     t1.start()  # 启动子线程
     if ml_class==-1:
        t1 = Process(target=pb_generation_full, args=(filepath, input_dim, train_setting_list))
        t1.daemon = True
        # t1.setDaemon(True)  # 将线程声
        t1.start()  # 启动子线程
        t1.join()
        # pb_generation_thread(filepath,input_dim,train_setting_list)
        # pb_generation_full(filepath,input_dim,train_setting_list)

     # time.sleep(5)
     todolist.status =1
     db.session.commit()
     flash('You have generated the pb file')
     return redirect(url_for('show_todo_list'))

def run_app(argv):
    app.run(host='0.0.0.0', port=5000, debug=True)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

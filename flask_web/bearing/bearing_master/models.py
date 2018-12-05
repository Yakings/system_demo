#!/usr/bin/python
#-*- coding: UTF-8 -*-
import time

from flask_login import UserMixin
try:
    from ext import db
except:
    from .ext import db


class TodoList(db.Model):
    __tablename__ = 'todolist'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    title = db.Column(db.String(1024), nullable=False)
    status = db.Column(db.Integer, nullable=False)
    create_time = db.Column(db.Integer, nullable=False)
    is_training = db.Column(db.Integer, nullable=False)

    def __init__(self, user_id, title, status):
        self.user_id = user_id
        self.title = title
        self.status = status
        self.create_time = time.time()
        self.is_training = 0


class User(UserMixin, db.Model):
    __tablename__ = "user"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(24), nullable=False)
    password = db.Column(db.String(24), nullable=False)

    def __init__(self, username, password):
        self.username = username
        self.password = password


class DataList(db.Model):
    __tablename__ = 'datalist'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    title = db.Column(db.String(1024), nullable=False)
    status = db.Column(db.Integer, nullable=False)
    create_time = db.Column(db.Integer, nullable=False)

    def __init__(self, user_id, title, status,creat_time=-1):
        self.user_id = user_id
        self.title = title
        self.status = status
        if creat_time<0:
            self.create_time = time.time()
        else:
            self.create_time = creat_time
'''
    `whether_data_augment` int(11) NOT NULL,        是否进行数据增强
    `deep_model_class` int(11) NOT NULL,        深度模型类别
    `ml_model_class` int(11) NOT NULL,        机器模型类别
     `input_dim` int(11) NOT NULL,             输入维度
     `output_dim` int(11) NOT NULL,             输出维度
     `weight_decay` int(11) NOT NULL,             权重衰减

     `learning_rate` int(11) NOT NULL,             学习率
    `activation_class` int(11) NOT NULL,        激活函数类别
    `layers_num` int(11) NOT NULL,             神经网络层数
    '''
class TrainSettingList(db.Model):
    __tablename__ = 'train_setting_list'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    create_time = db.Column(db.Integer, nullable=False)
    model_class = db.Column(db.Integer, nullable=False)
    GPU_setting = db.Column(db.Integer, nullable=False)

    data_paths = db.Column(db.String(1024), nullable=False)
    todolist_id = db.Column(db.Integer, nullable=False)


    ##################################
    whether_data_augment = db.Column(db.Integer, nullable=True)
    deep_model_class = db.Column(db.Integer, nullable=True)
    ml_model_class = db.Column(db.Integer, nullable=True)
    input_dim = db.Column(db.Integer, nullable=False)
    output_dim = db.Column(db.Integer, nullable=False)
    weight_decay = db.Column(db.Float, nullable=False)

    learning_rate = db.Column(db.Float, nullable=True)
    activation_class = db.Column(db.Integer, nullable=True)
    layers_num = db.Column(db.Integer, nullable=True)
    batch_size = db.Column(db.Integer, nullable=True)

    ##############
    optimizer = db.Column(db.Integer, nullable=True)
    net_losses = db.Column(db.Integer, nullable=True)



    def __init__(self, user_id, create_time,model_class,GPU_setting,data_paths,todolist_id,
                 whether_data_augment,deep_model_class,ml_model_class,input_dim,output_dim,weight_decay,
                 learning_rate,activation_class,layers_num,batch_size,optimizer,net_losses):
        self.user_id = user_id
        self.create_time = create_time
        self.model_class = model_class
        self.GPU_setting = GPU_setting
        self.data_paths = data_paths
        self.todolist_id = todolist_id

        self.whether_data_augment = whether_data_augment
        self.deep_model_class = deep_model_class
        self.ml_model_class = ml_model_class
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.activation_class = activation_class
        self.layers_num = layers_num
        self.batch_size = batch_size

        #####################
        self.optimizer = optimizer
        self.net_losses = net_losses

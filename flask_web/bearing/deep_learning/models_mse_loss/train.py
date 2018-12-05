#!/usr/bin/env python

import os
from datetime import datetime
import time
import tensorflow as tf
import re

import math
# import select
# from IPython import embed

try:
    from ext import db, login_manager
    from bearing.deep_learning.models_mse_loss.read_util import data_producer as data_input
    from bearing.deep_learning.models_mse_loss import resnet_con1d as resnet
    from bearing.deep_learning.pb_generation.auto_pb import infer_graph
    from bearing.deep_learning.pb_generation.convert_to_pb import  generate_pb,quantize_pb
    from bearing.deep_learning.machine_learning.train_ml import train_machine_learning

except:
    from bearing_master.ext import db, login_manager
    from deep_learning.models_mse_loss.read_util import data_producer as data_input
    from deep_learning.models_mse_loss import resnet_con1d as resnet
    from deep_learning.pb_generation.auto_pb import infer_graph
    from deep_learning.pb_generation.convert_to_pb import  generate_pb,quantize_pb
    from deep_learning.machine_learning.train_ml import train_machine_learning
import multiprocessing
import time

import json

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# # Dataset Configuration
#
# tf.app.flags.DEFINE_integer('num_classes', 1, """Number of classes in the dataset.""")
# tf.app.flags.DEFINE_integer('num_train_instance', 1281167, """Number of training images.""")
# # tf.app.flags.DEFINE_integer('num_val_instance', 50000, """Number of val images.""")
#
# # Network Configuration
# tf.app.flags.DEFINE_integer('batch_size', 4, """Number of images to process in a batch.""")
# tf.app.flags.DEFINE_integer('num_gpus', 1, """Number of GPUs.""")
#
# # Optimization Configuration
# tf.app.flags.DEFINE_float('l2_weight', 0.0001, """L2 loss weight applied all the weights""")
# tf.app.flags.DEFINE_float('momentum', 0.9, """The momentum of MomentumOptimizer""")
# tf.app.flags.DEFINE_float('initial_lr', 0.1, """Initial learning rate""")
# tf.app.flags.DEFINE_string('lr_step_epoch', "30.0,60.0", """Epochs after which learing rate decays""")
# tf.app.flags.DEFINE_float('lr_decay', 0.1, """Learning rate decay factor""")
# tf.app.flags.DEFINE_boolean('finetune', False, """Whether to finetune.""")
#
# # Training Configuration
# tf.app.flags.DEFINE_string('train_dir', './train', """Directory where to write log and checkpoint.""")
# tf.app.flags.DEFINE_integer('val_iter', 10, """Number of iterations during a val""")
#
#
# tf.app.flags.DEFINE_integer('max_steps', 500000, """Number of batches to run.""")
# tf.app.flags.DEFINE_integer('display', 100, """Number of iterations to display training info.""")
# tf.app.flags.DEFINE_integer('val_interval', 500, """Number of iterations to run a val""")
# tf.app.flags.DEFINE_integer('checkpoint_interval', 100, """Number of iterations to save parameters as a checkpoint""")
# tf.app.flags.DEFINE_float('gpu_fraction', 0.90, """The fraction of GPU memory to be allocated""")
# tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")
# # tf.app.flags.DEFINE_string('basemodel', './pretrained_model/init/', """Base model to load paramters""")
# tf.app.flags.DEFINE_string('basemodel', None, """Base model to load paramters""")
# tf.app.flags.DEFINE_string('checkpoint', None, """Model checkpoint to load""")
# # tf.app.flags.DEFINE_string('checkpoint', './train/', """Model checkpoint to load""")
#
# FLAGS = tf.app.flags.FLAGS
import numpy as np
from collections import namedtuple
FLAGSParams = namedtuple('FLAGSParams',
                    'num_classes, num_train_instance, batch_size,num_gpus, l2_weight, momentum, '
                     'initial_lr, lr_step_epoch, lr_decay,finetune, train_dir, val_iter, '
                     'max_steps, display, val_interval,checkpoint_interval, gpu_fraction, log_device_placement, '
                     'basemodel, checkpoint,')
# FLAGS = FLAGSParams(num_classes=1,num_train_instance=1281167,batch_size=4,
#                     num_gpus=1,l2_weight=0.0001, momentum=0.9,
#                     initial_lr=0.1, lr_step_epoch = "30.0,60.0",lr_decay = 0.1,
#                     finetune = False,
#                     train_dir = './train',
#                     val_iter = 10,max_steps = 500000,display = 100,
#                     val_interval = 100,checkpoint_interval = 100,gpu_fraction = 0.90, log_device_placement =False ,
#                     basemodel = None,checkpoint=None,)

DATA_ROOT_PATH = './bearing_master/data/user_datas/'


import csv
def write_add_csv(filename, contex = [0.1, 0.09]):
    with open(filename, "a+",newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(contex)
        # print('log loss')
    pass

def get_lr(initial_lr, lr_decay, lr_decay_steps, global_step):
    lr = initial_lr
    for s in lr_decay_steps:
        if global_step >= s:
            lr *= lr_decay
    return lr

def get_selected_step(trained_model_path):
    file_name = os.path.join(trained_model_path,'loss.csv')
    with open(file_name) as f:
        reader = csv.reader(f)

        column = [float(row[1]) for row in reader]
        selected = column.index(min(column))
        return selected
    pass
def pb_generation_full(trained_model_path,input_dim,train_setting_list):
    selected_step = get_selected_step(trained_model_path)
    trained_model_path = os.path.join(trained_model_path, 'trained_model')
    infer_graph(trained_model_path,selected_step=-1,input_dim=input_dim,train_setting_list=train_setting_list)
    time.sleep(1)
    generate_pb(trained_model_path)
    time.sleep(1)
    quantize_pb(trained_model_path,input_dim)


def get_pred_result(filepath, filename_list, todolist, train_setting_list,trained_model_path):
    train_list = train_setting_list
    ml_class = train_list.ml_model_class
    train_net_class = train_list.deep_model_class
    train_activation_class = train_list.activation_class
    train_input_dim_class = train_list.input_dim
    train_output_dim_class = train_list.output_dim

    train_weight_decay_class = train_list.weight_decay
    train_learning_rate_class = train_list.learning_rate
    train_layers_num_class = train_list.layers_num
    train_GPU_setting_class = train_list.GPU_setting

    train_optimizer = train_list.optimizer

    net_losses = train_list.net_losses


    FLAGS = FLAGSParams(num_classes=train_output_dim_class, num_train_instance=1281167,
                        batch_size=int(math.pow(2, train_list.batch_size)),
                        num_gpus=int(train_GPU_setting_class), l2_weight=train_weight_decay_class, momentum=0.9,
                        initial_lr=train_learning_rate_class, lr_step_epoch="30.0,60.0", lr_decay=0.1,
                        finetune=False,
                        train_dir='./train',
                        val_iter=10, max_steps=500000, display=100,
                        val_interval=100, checkpoint_interval=100, gpu_fraction=0.90, log_device_placement=False,
                        basemodel=None, checkpoint=None, )

    with tf.Graph().as_default():
        cpu_nums = 1

        init_step = 0
        global_step = tf.Variable(0, trainable=False, name='global_step')
        # Get images and labels of ImageNet
        if FLAGS.num_gpus > 0:
            num_threads = cpu_nums // FLAGS.num_gpus
        else:
            num_threads = cpu_nums

        # Get images and labels of ImageNet
        import multiprocessing
        print('Load ImageNet dataset(%d threads)' % num_threads)
        # train_images, train_labels = data_input.distorted_inputs(FLAGS.batch_size)
        train_images = tf.placeholder(tf.float32, shape=[1, train_input_dim_class, 1], name='image')
        # Build model
        lr_decay_steps = map(float, FLAGS.lr_step_epoch.split(','))

        hp = resnet.HParams(batch_size=FLAGS.batch_size,
                            num_gpus=FLAGS.num_gpus,
                            num_cpus=cpu_nums,
                            num_classes=FLAGS.num_classes,
                            input_dim=train_input_dim_class,
                            weight_decay=FLAGS.l2_weight,
                            momentum=FLAGS.momentum,
                            finetune=FLAGS.finetune,
                            ############################
                            layer_num=train_layers_num_class,
                            net_class=train_net_class,
                            activation_class=train_activation_class,
                            optimizer_id=train_optimizer,
                            net_losses=net_losses
                            )
        # train net
        network_train = resnet.ResNet(hp, [train_images], [0], global_step, name="train", is_train=False)
        network_train.build_model()
        # network_train.build_train_op()
        # train_summary_op = tf.summary.merge_all()  # Summaries(training)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            # device_count={"CPU": 4},  # limit to num_cpu_core CPU usage
            # inter_op_parallelism_threads=1,
            # intra_op_parallelism_threads=4,
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_fraction),
            allow_soft_placement=True,
            # allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        print(trained_model_path)
        if os.path.exists(trained_model_path):
            try:
                print('Load checkpoint %s' % trained_model_path)
                ckpt = tf.train.get_checkpoint_state(trained_model_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
                init_step = global_step.eval(session=sess)
            except:
                print('no checkpoint file!')
        else:
            print('no ckpt!')
        # _, train_loss_value, acc_value = \
        #     sess.run([network_train.train_op, network_train.loss, network_train.acc],
        #              feed_dict={network_train.is_train: False, network_train.lr: 0,})
        # filepath, filename_list
        all_pred_result = []
        for i in range(len(filename_list)):
            # print('filename_list',filename_list)
            if 'acc' in filename_list[i]:
                files = os.listdir(filepath)
                # print(files)
                if len(files) < 10:
                    for i in range(len(files)):
                        new_path = os.path.join(filepath, files[i])
                        if os.path.isdir(new_path):
                            filepath = new_path
                file_name = os.path.join(filepath, filename_list[i])
                # print(file_name)
                try:
                    with open(file_name) as f:
                        reader = csv.reader(f)
                        vib = [float(row[4]) for row in reader]
                    #################################
                    vib = vib[:2000]
                    max_size = len(vib)
                    set_max_size = (max_size // train_input_dim_class) * train_input_dim_class
                    div_size = max_size - set_max_size
                    first = 0
                    vib = np.array(vib)
                    vib = vib[first:first + set_max_size]
                    # vib = vib.reshape([VIB_SIZE//step,step])
                    # vib = np.max(vib,axis=1)
                    x_freq = np.fft.fft(vib)
                    freq = abs(x_freq)
                    step = set_max_size // train_input_dim_class
                    freq = freq.reshape([set_max_size // step, step])
                    freq = np.max(freq, axis=1)
                    freq = np.expand_dims(freq, axis=1)
                    # print(vib.dtype.name)
                    freq = freq.astype(np.float32)
                    #######################################
                    # print(freq[0])

                    preds_result = \
                        sess.run([network_train.preds_result],
                                 feed_dict={network_train.is_train: False, network_train.lr: 0, train_images:[freq]})
                    # print(preds_result)
                    all_pred_result.append(preds_result[0])
                except:
                    print('Prediction error! Finished')
                    break
        return all_pred_result
    pass


def train(is_training_v,is_training_filename,todolist,id_V,DATA_NAME_LIST,
          loss_filename,trained_model_path,train_setting_list):
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
    train_list = train_setting_list
    ml_class = train_list.ml_model_class
    train_net_class = train_list.deep_model_class
    train_activation_class = train_list.activation_class
    train_input_dim_class = train_list.input_dim
    train_output_dim_class = train_list.output_dim

    train_weight_decay_class = train_list.weight_decay
    train_learning_rate_class = train_list.learning_rate
    train_layers_num_class = train_list.layers_num
    train_GPU_setting_class = train_list.GPU_setting
    train_optimizer = train_list.optimizer

    learning_rate = train_list.learning_rate

    net_losses = train_list.net_losses
    # FLAGS.batch_size= int(math.pow(2,train_list.batch_size))
    # # FLAGS.ml_class = ml_class
    # # FLAGS.train_net_class = train_net_class
    # # FLAGS.train_activation_class = train_activation_class
    # # FLAGS.input_dim = train_input_dim_class
    # FLAGS.num_classes = train_output_dim_class
    # FLAGS.l2_weight = train_weight_decay_class
    # FLAGS.initial_lr = train_learning_rate_class
    # # FLAGS.layers_num = train_layers_num_class
    # FLAGS.num_gpus = int(train_GPU_setting_class)
    FLAGS = FLAGSParams(num_classes=train_output_dim_class, num_train_instance=1281167,
                        batch_size=int(math.pow(2,train_list.batch_size)),
                        num_gpus=int(train_GPU_setting_class),l2_weight=train_weight_decay_class, momentum=0.9,
                        initial_lr=train_learning_rate_class, lr_step_epoch = "30.0,60.0",lr_decay = 0.1,
                        finetune = False,
                        train_dir = './train',
                        val_iter = 10,max_steps = 500000,display = 100,
                        val_interval = 100,checkpoint_interval = 100,gpu_fraction = 0.90, log_device_placement =False ,
                        basemodel = None,checkpoint=None,)

    # model_setting = [
    #     ['模型类别:', select_list[0][train_net_class]],
    #     ['网络类型:', select_list[0][train_net_class]],
    #     ['激活函数:', select_list[1][train_activation_class]],
    #     ['优化器:', '还没实现'],
    #     ['损失函数:', '还没实现'],
    #     ['机器学习模型:', '还没实现'],
    #     ['输入维度:', train_input_dim_class],
    #     ['输出维度:', train_output_dim_class],
    #     ['权重衰减:', train_weight_decay_class],
    #     ['学习率:', train_learning_rate_class],
    #     ['GPU设置:', train_GPU_setting_class],
    #     ['网络层数:', train_layers_num_class],
    # ]  # 模型设置结果


    print('[Dataset Configuration]')
    # print('\tNumber of classes: %d' % FLAGS.num_classes)
    # print('\tNumber of training images: %d' % FLAGS.num_train_instance)
    # print('\tBatch size: %d' % FLAGS.batch_size)
    # print('\tNumber of GPUs: %d' % FLAGS.num_gpus)
    # print('\tBasemodel file: %s' % FLAGS.basemodel)
    # print('[Optimization Configuration]')
    # print('\tL2 loss weight: %f' % FLAGS.l2_weight)
    # print('\tThe momentum optimizer: %f' % FLAGS.momentum)
    # print('\tInitial learning rate: %f' % FLAGS.initial_lr)
    # print('\tEpochs per lr step: %s' % FLAGS.lr_step_epoch)
    # print('\tLearning rate decay: %f' % FLAGS.lr_decay)

    print('[Training Configuration]')
    # print('\tTrain dir: %s' % trained_model_path)
    # print('\tTraining max steps: %d' % FLAGS.max_steps)
    # print('\tSteps per displaying info: %d' % FLAGS.display)
    # print('\tSteps per validation: %d' % FLAGS.val_interval)
    # print('\tSteps during validation: %d' % FLAGS.val_iter)
    # print('\tSteps per saving checkpoints: %d' % FLAGS.checkpoint_interval)
    # print('\tGPU memory fraction: %f' % FLAGS.gpu_fraction)
    # print('\tLog device placement: %d' % FLAGS.log_device_placement)

    # cpu_nums = multiprocessing.cpu_count()
    cpu_nums = 1

    # time.sleep(10)

    with tf.Graph().as_default():
        init_step = 0
        global_step = tf.Variable(0, trainable=False, name='global_step')
        # Get images and labels of ImageNet
        if FLAGS.num_gpus>0:
            num_threads = cpu_nums // FLAGS.num_gpus
        else:
            num_threads = cpu_nums
        print('Load ImageNet dataset(%d threads)' % num_threads)
        train_path_list = []
        test_path_list = []
        for kk in range(len(DATA_NAME_LIST)):
            if kk%2==0:
                test_path_list.append(DATA_NAME_LIST[kk])
                pass
            else:
                test_path_list.append(DATA_NAME_LIST[kk])

        # with tf.device('/cpu:0'):
        # print('\tLoading training data from %s' % FLAGS.train_dataset)

        train_path_list = DATA_NAME_LIST
        test_path_list = DATA_NAME_LIST
        with tf.variable_scope('train_image'):
            # train_images, train_labels = data_input.distorted_inputs(FLAGS.train_image_root, FLAGS.train_dataset
            #                                , FLAGS.batch_size, True, num_threads=num_threads, num_sets=FLAGS.num_gpus)
            if FLAGS.num_gpus>0:
                train_images, train_labels = data_input.distorted_inputs(FLAGS.batch_size,train_path_list=train_path_list,num_sets=FLAGS.num_gpus,input_dim=train_input_dim_class)
            else:
                train_images, train_labels = data_input.distorted_inputs(FLAGS.batch_size,train_path_list=train_path_list,num_sets=cpu_nums,input_dim=train_input_dim_class)


        # print('\tLoading validation data from %s' % FLAGS.val_dataset)
        with tf.variable_scope('test_image'):
            # val_images, val_labels = data_input.inputs(eval_data='test',batch_size=FLAGS.batch_size)
            if FLAGS.num_gpus>0:
                val_images, val_labels = data_input.test_inputs(FLAGS.batch_size,test_path_list=test_path_list,num_sets=FLAGS.num_gpus,input_dim=train_input_dim_class)
            else:
                val_images, val_labels = data_input.test_inputs(FLAGS.batch_size,test_path_list=test_path_list,num_sets=cpu_nums,input_dim=train_input_dim_class)
                # val_images, val_labels = data_input.test_inputs(FLAGS.batch_size,test_path_list = test_path_list)


        # tf.summary.image('images', train_images[0])
        # Build model
        lr_decay_steps = map(float,FLAGS.lr_step_epoch.split(','))
        # lr_decay_steps = map(int,[s*FLAGS.num_train_instance/FLAGS.batch_size/FLAGS.num_gpus for s in lr_decay_steps])
        if FLAGS.num_gpus>0:
            lr_decay_steps = map(int,[s*FLAGS.num_train_instance/FLAGS.batch_size//FLAGS.num_gpus for s in lr_decay_steps])
        else:
            lr_decay_steps = map(int,[s*FLAGS.num_train_instance/FLAGS.batch_size//1 for s in lr_decay_steps])

        hp = resnet.HParams(batch_size=FLAGS.batch_size,
                            num_gpus=FLAGS.num_gpus,
                            num_cpus=cpu_nums,
                            num_classes=FLAGS.num_classes,
                            input_dim=train_input_dim_class,
                            weight_decay=FLAGS.l2_weight,
                            momentum=FLAGS.momentum,
                            finetune=FLAGS.finetune,
                            ############################
                            layer_num = train_layers_num_class,
                            net_class = train_net_class,
                            activation_class = train_activation_class,
                            optimizer_id=train_optimizer,
                            learning_rate = learning_rate,
                            net_losses=net_losses
                            )
        # train net
        network_train = resnet.ResNet(hp, train_images, train_labels, global_step, name="train",is_train = True)
        network_train.build_model()
        network_train.build_train_op()
        # train_summary_op = tf.summary.merge_all()  # Summaries(training)

        # test net
        network_val = resnet.ResNet(hp, val_images, val_labels, global_step, name="val", reuse_weights=True,is_train = False)
        network_val.build_model()
        print('Number of Weights: %d' % network_train._weights)
        print('FLOPs: %d' % network_train._flops)


        # Build an initialization operation to run below.
        # init = tf.global_variables_initializer()
        # print('here')

        # Start running operations on the Graph.
        # sess = tf.Session(config=tf.ConfigProto(
        #     # device_count={"CPU": 4},  # limit to num_cpu_core CPU usage
        #     # inter_op_parallelism_threads=1,
        #     # intra_op_parallelism_threads=4,
        #     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_fraction),
        #     allow_soft_placement=False,
        #     # allow_soft_placement=True,
        #     log_device_placement=FLAGS.log_device_placement))


        # sess = tf.Session(config=tf.ConfigProto(
        #     gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_fraction),
        #     allow_soft_placement=True,
        #     log_device_placement=FLAGS.log_device_placement))

        # with tf.Session(config=tf.ConfigProto(
        #     gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_fraction),
        #     allow_soft_placement=True,
        #     log_device_placement=FLAGS.log_device_placement)) as sess:

        # config = tf.ConfigProto(
        #         device_count={"CPU": 4},  # limit to num_cpu_core CPU usage
        #         inter_op_parallelism_threads=1,
        #         intra_op_parallelism_threads=4,
        #         gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_fraction),
        #         allow_soft_placement=True,
        #         log_device_placement=FLAGS.log_device_placement)
        config = tf.ConfigProto(
            # device_count={"CPU": 4},  # limit to num_cpu_core GPU usage
            allow_soft_placement=True
        )
        # config = tf.ConfigProto(
        #     device_count={"GPU": 0},  # limit to num_cpu_core GPU usage
        #     allow_soft_placement=True
        # )
        with tf.Session(config=config) as sess:
            coord = tf.train.Coordinator()
            init = tf.global_variables_initializer()
            print('run init')
            sess.run(init)
            print('here3')
            # Create a saver.
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=10000)
            # if FLAGS.checkpoint is not None:
                # print('Load checkpoint %s' % FLAGS.checkpoint)
                # ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint)
                # saver.restore(sess, ckpt.model_checkpoint_path)
                # init_step = global_step.eval(session=sess)
            if os.path.exists(trained_model_path):
                try:
                    print('Load checkpoint %s' % trained_model_path)
                    ckpt = tf.train.get_checkpoint_state(trained_model_path)
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    init_step = global_step.eval(session=sess)
                except:
                    print('no checkpoint file!')
            elif FLAGS.basemodel:
                # Define a different saver to save model checkpoints
                print('Load parameters from basemodel %s' % FLAGS.basemodel)
                variables = tf.global_variables()

                vars_restore = [var for var in variables
                                if not "Momentum" in var.name and
                                   not "global_step" in var.name and
                                   not re.match('.+Adam.+',var.name) and  # use adam
                                   not re.match('beta[0-9]_power', var.name) and # paraments in adam
                                   not re.match('logits/fc/.+', var.name)] # exclude fully connection layer
                # print(vars_restore)
                saver_restore = tf.train.Saver(vars_restore, max_to_keep=10000)

                # added
                #################
                ckpt = tf.train.get_checkpoint_state(FLAGS.basemodel)
                saver_restore.restore(sess, ckpt.model_checkpoint_path)
                # saver_restore.restore(sess, FLAGS.basemodel)
            else:
                print('No checkpoint file of basemodel found. Start from the scratch.')

            # Start queue runners & summary_writer
            queue_runner = tf.train.start_queue_runners(sess=sess,coord=coord)

            if not os.path.exists(trained_model_path):
                os.mkdir(trained_model_path)
            summary_writer = tf.summary.FileWriter(os.path.join(trained_model_path, str(global_step.eval(session=sess))),
                                                    sess.graph)

            # Training!
            val_best_acc = 0.0

            print('running!')
            is_finished = 0
            for step in range(init_step, FLAGS.max_steps):


                if is_training_v == 1:
                    is_finished = 1
                    time.sleep(2)
                    # Train
                    lr_value = get_lr(FLAGS.initial_lr, FLAGS.lr_decay, lr_decay_steps, step)
                    # start_time = time.time()
                    # _, train_loss_value, acc_value, train_summary_str = \
                    #         sess.run([network_train.train_op, network_train.loss, network_train.acc, train_summary_op],
                    #                 feed_dict={network_train.is_train:True, network_train.lr:lr_value})
                    _, train_loss_value, acc_value = \
                        sess.run([network_train.train_op, network_train.loss, network_train.acc],
                                 feed_dict={network_train.is_train: True, network_train.lr: lr_value})

                    # duration = time.time() - start_time

                    # Display & Summary(training)
                    # if step % FLAGS.display == 0 or step < 10:
                    #     num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
                    #     examples_per_sec = num_examples_per_step / duration
                    #     sec_per_batch = float(duration)
                    #     format_str = ('%s: (Training) step %d, loss=%.4f, acc=%.4f, lr=%f (%.1f examples/sec; %.3f '
                    #                   'sec/batch)')
                    #     print (format_str % (datetime.now(), step, train_loss_value, acc_value, lr_value,
                    #                          examples_per_sec, sec_per_batch))
                    #     summary_writer.add_summary(train_summary_str, step)

                    # Save the model checkpoint periodically.
                    # if (step > init_step and step % FLAGS.checkpoint_interval == 0) or (step + 1) == FLAGS.max_steps:
                    if (step % FLAGS.checkpoint_interval == 0) or (step + 1) == FLAGS.max_steps:
                        checkpoint_path = os.path.join(trained_model_path, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=step)

                        # val
                        # if step % FLAGS.val_interval == 10:
                        if 1:
                            val_loss, val_acc = 0.0, 0.0
                            for i in range(FLAGS.val_iter):
                                loss_value, acc_value = sess.run([network_val.loss, network_val.acc],
                                                                 feed_dict={network_val.is_train: False})
                                val_loss += loss_value
                                val_acc += acc_value
                                pass
                            val_loss /= FLAGS.val_iter
                            val_acc /= FLAGS.val_iter
                            val_best_acc = max(val_best_acc, val_acc)
                            format_str = ('%s: (val)     step %d, loss=%.4f, acc=%.4f')
                            print(format_str % (datetime.now(), step, val_loss, val_acc))

                            val_summary = tf.Summary()
                            val_summary.value.add(tag='val/loss', simple_value=val_loss)
                            val_summary.value.add(tag='val/acc', simple_value=val_acc)
                            val_summary.value.add(tag='val/best_acc', simple_value=val_best_acc)
                            summary_writer.add_summary(val_summary, step)
                            summary_writer.flush()


                            print('train loss:',train_loss_value,'val loss:',val_loss)

                            write_add_csv(loss_filename, contex=[train_loss_value,val_loss])

                    with open(is_training_filename, 'r') as load_f:
                        load_dict = json.load(load_f)
                    # todolist_v = TodoList.query.filter_by(id=id_V).first_or_404()
                    is_training_v = load_dict[0]
                    print('train!', step,',',is_training_v)
                else:
                    is_finished = 0
                    print('break')

                    break

            coord.request_stop()
            coord.join(queue_runner)
            sess.close()
            return is_finished
            pass




# def main(argv=None):  # pylint: disable=unused-argument
#   train(argv[0],argv[1],argv[2],argv[3],argv[4],argv[5])



def trian_function(is_training_v,is_training_filename,todolist,id_V,datalists,loss_filename,trained_model_path,train_setting_list):
    DATA_NAME_LIST = []
    for data in datalists:
        DATA_NAME_LIST.append(os.path.join(DATA_ROOT_PATH,str(data.create_time)))
    print(DATA_NAME_LIST)
    # while is_training_v == 1:
    #     print(is_training_v)
    #     with open(filename, 'r') as load_f:
    #         load_dict = json.load(load_f)
    #     todolist_v = TodoList.query.filter_by(id=id_V).first_or_404()
    #     is_training_v = load_dict[0]
    #     print('try!', is_training_v)
    #     time.sleep(1)

    # try:
    #     if train_setting_list.ml_model_class<0:
    #         if train_setting_list.deep_model_class in [0,1,2,3]:
    #             tf.app.run(train(is_training_v,is_training_filename,
    #                              todolist,id_V,DATA_NAME_LIST,loss_filename,
    #                              trained_model_path,train_setting_list))
    #     else:
    #         train_machine_learning(todolist,DATA_NAME_LIST,trained_model_path,train_setting_list)
    # except:
    #     print('Training Finished!')
    import sys
    if train_setting_list.ml_model_class < 0:
        if train_setting_list.deep_model_class in [0, 1, 2, 3]:
            is_finished = 0
            # tf.app.run(
            #     main=lambda is_finished:train(is_training_v,
            #                                   is_training_filename,todolist, id_V, DATA_NAME_LIST,
            #                                   loss_filename,trained_model_path, train_setting_list))
            is_finished = train(is_training_v,is_training_filename, todolist, id_V, DATA_NAME_LIST,
                  loss_filename, trained_model_path, train_setting_list)
            # print(is_finished)
            return is_finished
    else:
        train_machine_learning(todolist, DATA_NAME_LIST, trained_model_path, train_setting_list)
        return 1

    # tf.app.run(argv=[is_training_v,filename,TodoList,id_V,DATA_NAME_LIST,loss_filename])
    # tf.app.run(train(is_training_v, is_training_filename,
    #                 todolist, id_V, DATA_NAME_LIST, loss_filename,
    #                 trained_model_path, train_setting_list))

# if __name__ == '__main__':
#   tf.app.run()

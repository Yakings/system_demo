#!/usr/bin/env python

import os
from datetime import datetime
import time
import tensorflow as tf
import re
from ext import db, login_manager
import math
# import select
# from IPython import embed

try:
    from bearing.deep_learning.models_mse_loss.read_util import data_producer as data_input
    from bearing.deep_learning.models_mse_loss import resnet_con1d as resnet
    from bearing.deep_learning.pb_generation.auto_pb import infer_graph
    from bearing.deep_learning.pb_generation.convert_to_pb import  generate_pb,quantize_pb
    from bearing.deep_learning.machine_learning.train_ml import train_machine_learning

except:
    from ..models_mse_loss.read_util import data_producer as data_input
    from . import resnet_con1d as resnet
    from ..pb_generation.auto_pb import infer_graph
    from ..pb_generation.convert_to_pb import  generate_pb,quantize_pb
    from ..machine_learning.train_ml import train_machine_learning
import multiprocessing
import time
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

import numpy as np
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
                            activation_class=train_activation_class
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


        if os.path.exists(trained_model_path):
            try:
                print('Load checkpoint %s' % trained_model_path)
                ckpt = tf.train.get_checkpoint_state(trained_model_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
                init_step = global_step.eval(session=sess)
            except:
                print('no checkpoint file!')

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
                        new_path = os.path.join(filepath, files[0])
                        if os.path.isdir(new_path):
                            filepath = new_path
                file_name = os.path.join(filepath, filename_list[i])
                # print(file_name)
                try:
                    with open(file_name) as f:
                        reader = csv.reader(f)
                        vib = [float(row[4]) for row in reader]
                    #################################
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

import json

# os.environ['CUDA_VISIBLE_DEVICES'] = ''



if __name__ == '__main__':
  tf.app.run()

#!/usr/bin/env python

import os
from datetime import datetime
import time
import tensorflow as tf
import re

# import select
# from IPython import embed

try:
    # from models.read_util import data_producer as data_input
    from models import resnet_con1d as resnet
    # from models.read_util.tensorflow_input import IMAGE_SIZE
except:
    # from .read_util import data_producer as data_input
    # from .read_util.tensorflow_input import IMAGE_SIZE
    from .import resnet_con1d as resnet
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Dataset Configuration
#
# tf.app.flags.DEFINE_integer('num_classes', 1, """Number of classes in the dataset.""")
# tf.app.flags.DEFINE_integer('num_train_instance', 1281167, """Number of training images.""")
# tf.app.flags.DEFINE_integer('num_val_instance', 50000, """Number of val images.""")
#
# # Network Configuration
# tf.app.flags.DEFINE_integer('batch_size', 128, """Number of images to process in a batch.""")
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
# tf.app.flags.DEFINE_integer('max_steps', 500000, """Number of batches to run.""")
# tf.app.flags.DEFINE_integer('display', 100, """Number of iterations to display training info.""")
# tf.app.flags.DEFINE_integer('val_interval', 500, """Number of iterations to run a val""")
# tf.app.flags.DEFINE_integer('val_iter', 100, """Number of iterations during a val""")
# tf.app.flags.DEFINE_integer('checkpoint_interval', 500, """Number of iterations to save parameters as a checkpoint""")
# tf.app.flags.DEFINE_float('gpu_fraction', 0.90, """The fraction of GPU memory to be allocated""")
# tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")
# # tf.app.flags.DEFINE_string('basemodel', './pretrained_model/init/', """Base model to load paramters""")
# tf.app.flags.DEFINE_string('basemodel', None, """Base model to load paramters""")
# FLAGS = tf.app.flags.FLAGS

from collections import namedtuple
FLAGSParams = namedtuple('FLAGSParams',
                    'num_classes, num_train_instance, batch_size,num_gpus, l2_weight, momentum, '
                     'initial_lr, lr_step_epoch, lr_decay,finetune, train_dir, val_iter, '
                     'max_steps, display, val_interval,checkpoint_interval, gpu_fraction, log_device_placement, '
                     'basemodel, checkpoint,')
def get_lr(initial_lr, lr_decay, lr_decay_steps, global_step):
    lr = initial_lr
    for s in lr_decay_steps:
        if global_step >= s:
            lr *= lr_decay
    return lr

import math
def infer_graph(checkpoint_path,selected_step=-1,input_dim=10,train_setting_list=[]):
    # os.environ['CUDA_VISIBLE_DEVICES'] = ''
    print('[Dataset Configuration]')
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
        init_step = 0
        global_step = tf.Variable(0, trainable=False, name='global_step')

        # Get images and labels of ImageNet
        cpu_nums = 1
        num_gpus = 1
        import multiprocessing
        num_threads = multiprocessing.cpu_count() / num_gpus
        print('Load ImageNet dataset(%d threads)' % num_threads)
        # train_images, train_labels = data_input.distorted_inputs(FLAGS.batch_size)
        train_images = tf.placeholder(tf.float32,shape=[1,input_dim,1],name='image')
        # Build model
        # lr_decay_steps = map(float,FLAGS.lr_step_epoch.split(','))
        # lr_decay_steps = map(int,[s*FLAGS.num_train_instance/FLAGS.batch_size/FLAGS.num_gpus for s in lr_decay_steps])
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
                            learning_rate=learning_rate,
                            net_losses=net_losses
                            )
        # train net
        network_train = resnet.ResNet(hp, [train_images], [0], global_step, name="train", is_train=False)
        # ResNet(hp, [train_images], [], global_step, name="train",is_train = False)
        network_train.build_model()
        # network_train.build_train_op()
        # train_summary_op = tf.summary.merge_all()  # Summaries(training)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph.

        # sess = tf.Session(config=tf.ConfigProto(
        #     # device_count={"CPU": 4},  # limit to num_cpu_core CPU usage
        #     # inter_op_parallelism_threads=1,
        #     # intra_op_parallelism_threads=4,
        #     gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9),
        #     allow_soft_placement=True,
        #     # allow_soft_placement=True,
        #     log_device_placement=True))

        config = tf.ConfigProto(
            # device_count={"CPU": 4},  # limit to num_cpu_core GPU usage
            allow_soft_placement=True
        )
        with tf.Session(config=config) as sess:

            sess.run(init)

            name = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
            print(name)
            # print([_.name for _ in node])

            # Create a saver.
            saver = tf.train.Saver(tf.global_variables())
            if checkpoint_path is not None:
                print('Load checkpoint %s' % checkpoint_path)
                ckpt = tf.train.get_checkpoint_state(checkpoint_path)

                if selected_step>0:
                    selected_ckpt = ckpt.all_model_checkpoint_paths[0]
                else:
                    selected_ckpt = ckpt.model_checkpoint_path

                saver.restore(sess, selected_ckpt)
                save_pb_path = os.path.join(checkpoint_path,'output')
                isExists = os.path.exists(save_pb_path)
                # 判断结果
                if not isExists:
                    os.makedirs(save_pb_path)
                tf.train.write_graph(sess.graph_def, checkpoint_path,
                                     'output/%s_%s.pb' % ('mist', 'binary_clas'),
                                     as_text=True)

                saver2 = tf.train.Saver(tf.global_variables())
                # saver = tf.train.Saver(variables_to_restore)
                saver2.save(sess, checkpoint_path+'/output/%s_%s.ckpt' % ('mist', 'binary_clas'))
                print('Restored ckpt file.')

                # # Start queue runners & summary_writer
                # graph = tf.get_default_graph()
                # tf.train.write_graph(sess.graph_def, FLAGS.train_dir,  './output/%s_%s.ckpt'%('foxcoon','binary_clas'),
                #                      as_text=True)
                # saver = tf.train.Saver(tf.global_variables())
                # checkpoint_path = os.path.join(FLAGS.train_dir,  './output/%s_%s.ckpt'%('foxcoon','binary_clas'))
                # saver.save(sess, checkpoint_path)

                print('Successfully restored checkpoint file!')
            else:
                print('No checkpoint file of basemodel found. Start from the scratch.')







def main(argv=None):  # pylint: disable=unused-argument
  infer_graph()

def trian_function():
    tf.app.run()
if __name__ == '__main__':
  tf.app.run()

import os

# new_lib = '/usr/local/cuda-8.0/lib64'
# print(type(os.environ))
# if os.getenv('LD_LIBRARY_PATH') is not None and new_lib not in os.getenv('LD_LIBRARY_PATH'):
#     os.environ['LD_LIBRARY_PATH'] += ':'+new_lib
# else:
#     os.environ['LD_LIBRARY_PATH'] = new_lib
#
# os.environ['LD_LIBRARY_PATH']='$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64'
# print(os.environ)

import tensorflow as tf
# import resnet_eval
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64
FLAGS = tf.app.flags.FLAGS


# tf.app.flags.DEFINE_string('checkpoint', './train/model.ckpt-20000', """Path to the model checkpoint file""")
# tf.app.flags.DEFINE_boolean('finetune', False, """Whether to finetune.""")
# tf.app.flags.DEFINE_integer('num_classes', 2, """Number of classes in the dataset.""")
# tf.app.flags.DEFINE_string('checkpoint_dir', './train', """Path to the model checkpoint file""")
# tf.app.flags.DEFINE_string('dataset_name', 'mist', """Path to the model checkpoint file""")
# tf.app.flags.DEFINE_string('model_name', 'binary_clas', """Path to the model checkpoint file""")


# Optimization Configuration
# tf.app.flags.DEFINE_float('l2_weight', 0.0001, """L2 loss weight applied all the weights""")
# tf.app.flags.DEFINE_float('momentum', 0.9, """The momentum of MomentumOptimizer""")
# tf.app.flags.DEFINE_float('initial_lr', 0.1, """Initial learning rate""")
# tf.app.flags.DEFINE_string('lr_step_epoch', "80.0,120.0,160.0", """Epochs after which learing rate decays""")
# tf.app.flags.DEFINE_float('lr_decay', 0.1, """Learning rate decay factor""")

try:
    from models.tools import freeze_graph
    from models.tools import optimize_for_inference, quantize_graph

except:
    from .tools import freeze_graph
    from .tools import optimize_for_inference, quantize_graph

from tensorflow.core.protobuf import saver_pb2

DATA_SET_NAME = 'mist'
MODEL_NAME = 'binary_clas'
def generate_pb(checkpoint_path):
    print('%s/output/%s_%s.pb' %(checkpoint_path,DATA_SET_NAME,MODEL_NAME))
    freeze_graph.freeze_graph(input_graph='%s/output/%s_%s.pb' %(checkpoint_path,DATA_SET_NAME,MODEL_NAME),
                 input_saver='',
                 input_binary=False,
                 input_checkpoint=checkpoint_path + '/output/%s_%s.ckpt'%(DATA_SET_NAME,MODEL_NAME),
                 # output_node_names='tower_0/logits_1/fc/BiasAdd',
                 output_node_names='tower_0/Squeeze_1',
                 restore_op_name='save/restore_all',
                 filename_tensor_name='save/Const:0',
                 output_graph='%s/output/%s_%s_output.pb' %(checkpoint_path,DATA_SET_NAME,MODEL_NAME),
                 clear_devices=True,
                 initializer_nodes="",
                 variable_names_blacklist="",
                 checkpoint_version=saver_pb2.SaverDef.V1)
    print('Generated output pb file.')

def quantize_pb(checkpoint_path,input_dim):
    optimize_for_inference.optimize_for_inference(input='%s/output/%s_%s_output.pb' %(checkpoint_path,DATA_SET_NAME,MODEL_NAME),
                                                  output='%s/output/%s_%s_optimized.pb' %(checkpoint_path,DATA_SET_NAME,MODEL_NAME),
                           frozen_graph=True, input_names='image',
                           # output_names='tower_0/logits_1/fc/BiasAdd'
                           output_names='tower_0/Squeeze_1'
                                                  )
    print('Optimize over.')

    quantize_graph.quantize_graph(input='%s/output/%s_%s_optimized.pb' %(checkpoint_path,DATA_SET_NAME,MODEL_NAME),
                   # output_node_names='tower_0/logits_1/fc/BiasAdd',
                   output_node_names='tower_0/Squeeze_1',
                   output='%s/output/%s_%s_DM%sDM_quaintized.pb' %(checkpoint_path,DATA_SET_NAME,MODEL_NAME,input_dim),
                   mode='weights_rounded'
                   ##logtostderr='',
                                  )
    print('All over.')
    pass



from time import sleep
def main(argv=None):  # pylint: disable=unused-argument
    # restore_ckpt()
    # sleep(3)
    generate_pb()
    sleep(3)
    quantize_pb()


    pass

if __name__ == '__main__':
  tf.app.run()
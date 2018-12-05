import numpy as np
import tensorflow as tf

## TensorFlow helper functions

WEIGHT_DECAY_KEY = 'WEIGHT_DECAY'

def _relu(x, leakness=0.0, name=None):
    if leakness > 0.0:
        name = 'lrelu' if name is None else name
        return tf.maximum(x, x*leakness, name='lrelu')
    else:
        name = 'relu' if name is None else name
        return tf.nn.relu(x, name='relu')


def _conv(x, filter_size, out_channel, strides, pad='SAME', input_q=None, output_q=None, name='conv'):
    if (input_q == None)^(output_q == None):
        raise ValueError('Input/Output splits are not correctly given.')

    in_shape = x.get_shape()
    with tf.variable_scope(name):
        # Main operation: conv2d
        with tf.device('/CPU:0'):
            kernel = tf.get_variable('kernel', [filter_size, in_shape[2], out_channel],
                            tf.float32, initializer=tf.random_normal_initializer(
                                stddev=np.sqrt(2.0/filter_size/out_channel)))
        if kernel not in tf.get_collection(WEIGHT_DECAY_KEY):
            tf.add_to_collection(WEIGHT_DECAY_KEY, kernel)
            # print('\tadded to WEIGHT_DECAY_KEY: %s(%s)' % (kernel.name, str(kernel.get_shape().as_list())))
        conv = tf.nn.conv1d(x, kernel, strides, pad)

        # Split and split loss
        if (input_q is not None) and (output_q is not None):
            # w = tf.reduce_mean(kernel, axis=[0, 1])
            # w = tf.sqrt(tf.reduce_mean(tf.square(kernel), [0, 1]))
            _add_split_loss(kernel, input_q, output_q)

    return conv


def _fc(x, out_dim, input_q=None, output_q=None, name='fc'):
    if (input_q == None)^(output_q == None):
        raise ValueError('Input/Output splits are not correctly given.')

    with tf.variable_scope(name):
        # Main operation: fc
        with tf.device('/CPU:0'):
            w = tf.get_variable('weights', [x.get_shape()[1], out_dim],
                            tf.float32, initializer=tf.random_normal_initializer(
                                stddev=np.sqrt(1.0/out_dim)))
            b = tf.get_variable('biases', [out_dim], tf.float32,
                                initializer=tf.constant_initializer(0.0))
        if w not in tf.get_collection(WEIGHT_DECAY_KEY):
            tf.add_to_collection(WEIGHT_DECAY_KEY, w)
            # print('\tadded to WEIGHT_DECAY_KEY: %s(%s)' % (w.name, str(w.get_shape().as_list())))
        fc = tf.nn.bias_add(tf.matmul(x, w), b)

        # Split loss
        if (input_q is not None) and (output_q is not None):
            _add_split_loss(w, input_q, output_q)

    return fc


def _get_split_q(ngroups, dim, name='split', l2_loss=False):
    with tf.variable_scope(name):
        # alpha = tf.get_variable('alpha', shape=[ngroups, dim], dtype=tf.float32,
                              # initializer=tf.random_normal_initializer(stddev=0.1))
        # q = tf.nn.softmax(alpha, dim=0, name='q')
        std_dev = 0.01
        init_val = np.random.normal(0, std_dev, (ngroups, dim))
        init_val = init_val - np.average(init_val, axis=0) + 1.0/ngroups
        with tf.device('/CPU:0'):
            q = tf.get_variable('q', shape=[ngroups, dim], dtype=tf.float32,
                                # initializer=tf.constant_initializer(1.0/ngroups))
                                initializer=tf.constant_initializer(init_val))
        if l2_loss:
            if q not in tf.get_collection(WEIGHT_DECAY_KEY):
                tf.add_to_collection(WEIGHT_DECAY_KEY, q*2.236)

    return q

def _merge_split_q(q, merge_idxs, name='merge'):
    assert len(q.get_shape()) == 2
    ngroups, dim = q.get_shape().as_list()
    assert ngroups == len(merge_idxs)

    with tf.variable_scope(name):
        max_idx = np.max(merge_idxs)
        temp_list = []
        for i in range(max_idx + 1):
            temp = []
            for j in range(ngroups):
                if merge_idxs[j] == i:
                    temp.append(tf.slice(q, [j, 0], [1, dim]))
            temp_list.append(tf.add_n(temp))
        ret = tf.concat(0, temp_list)

    return ret


def _get_even_merge_idxs(N, split):
    assert N >= split
    num_elems = [(N + split - i - 1)/split for i in range(split)]
    expand_split = [[i] * n for i, n in enumerate(num_elems)]
    return [t for l in expand_split for t in l]


def _add_split_loss(w, input_q, output_q):
    # Check input tensors' measurements
    assert len(w.get_shape()) == 2 or len(w.get_shape()) == 4
    in_dim, out_dim = w.get_shape().as_list()[-2:]
    assert len(input_q.get_shape()) == 2
    assert len(output_q.get_shape()) == 2
    assert in_dim == input_q.get_shape().as_list()[1]
    assert out_dim == output_q.get_shape().as_list()[1]
    assert input_q.get_shape().as_list()[0] == output_q.get_shape().as_list()[0]  # ngroups
    ngroups = input_q.get_shape().as_list()[0]
    assert ngroups > 1

    # Add split losses to collections
    T_list = []
    U_list = []
    if input_q not in tf.get_collection('OVERLAP_LOSS_WEIGHTS'):
        tf.add_to_collection('OVERLAP_LOSS_WEIGHTS', input_q)
        print('\t\tAdd overlap & split loss for %s' % input_q.name)
        for i in range(ngroups):
            for j in range(ngroups):
                if i == j:
                    continue
                T_list.append(tf.reduce_sum(input_q[i,:] * input_q[j,:]))
            U_list.append(tf.square(tf.reduce_sum(input_q[i,:])))
    if output_q not in tf.get_collection('OVERLAP_LOSS_WEIGHTS'):
        print('\t\tAdd overlap & split loss for %s' % output_q.name)
        tf.add_to_collection('OVERLAP_LOSS_WEIGHTS', output_q)
        for i in range(ngroups):
            for j in range(ngroups):
                if i == j:
                    continue
                T_list.append(tf.reduce_sum(output_q[i,:] * output_q[j,:]))
            U_list.append(tf.square(tf.reduce_sum(output_q[i,:])))
    if T_list:
        tf.add_to_collection('OVERLAP_LOSS', tf.add_n(T_list))
    if U_list:
        tf.add_to_collection('UNIFORM_LOSS', tf.add_n(U_list))

    S_list = []
    for i in range(ngroups):
        if len(w.get_shape()) == 4:
            w_reduce = tf.reduce_mean(tf.square(w), [0, 1])
            wg_row = tf.matmul(tf.matmul(tf.diag(tf.square(1 - input_q[i,:])), w_reduce), tf.diag(tf.square(output_q[i,:])))
            wg_row_l2 = tf.reduce_sum(tf.sqrt(tf.reduce_sum(wg_row, 1)))
            wg_col = tf.matmul(tf.matmul(tf.diag(tf.square(input_q[i,:])), w_reduce), tf.diag(tf.square(1 - output_q[i,:])))
            wg_col_l2 = tf.reduce_sum(tf.sqrt(tf.reduce_sum(wg_col, 0)))
        else:  # len(w.get_shape()) == 2
            wg_row = tf.matmul(tf.matmul(tf.diag(1 - input_q[i,:]), w), tf.diag(output_q[i,:]))
            wg_row_l2 = tf.reduce_sum(tf.sqrt(tf.reduce_sum(wg_row * wg_row, 1)))
            wg_col = tf.matmul(tf.matmul(tf.diag(input_q[i,:]), w), tf.diag(1 - output_q[i,:]))
            wg_col_l2 = tf.reduce_sum(tf.sqrt(tf.reduce_sum(wg_col * wg_col, 0)))
        S_list.append(wg_row_l2 + wg_col_l2)
    S = tf.add_n(S_list)
    tf.add_to_collection('WEIGHT_SPLIT', S)

    # Add histogram for w if split losses are added
    scope_name = tf.get_variable_scope().name
    tf.histogram_summary("%s/weights" % scope_name, w)
    print('\t\tAdd split loss for %s(%dx%d, %d groups)' \
          % (tf.get_variable_scope().name, in_dim, out_dim, ngroups))

    return




def _bn_back(x, is_train, global_step=None, moving_average_decay = 0.9, name='bn'):
    # moving_average_decay = 0.99
    # moving_average_decay_init = 0.99
    with tf.variable_scope(name):
        decay = moving_average_decay
        bn = tf.contrib.layers.batch_norm(inputs=x, decay=decay,
                                          updates_collections=[tf.GraphKeys.UPDATE_OPS], center=True,
                                          scale=True, epsilon=1e-5, is_training=is_train,
                                          trainable=True)
    return bn

# def _bn(x, is_train, global_step=None, moving_average_decay = 0.9, name='bn'):
#     # moving_average_decay = 0.99
#     # moving_average_decay_init = 0.99
#     with tf.variable_scope(name):
#         decay = moving_average_decay
#         batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
#         with tf.device('/CPU:0'):
#             moving_mean = tf.get_variable('mu', batch_mean.get_shape(), tf.float32,
#                             initializer=tf.zeros_initializer(), trainable=False)
#             moving_variance = tf.get_variable('sigma', batch_var.get_shape(), tf.float32,
#                             initializer=tf.ones_initializer(), trainable=False)
#             beta = tf.get_variable('beta', batch_mean.get_shape(), tf.float32,
#                             initializer=tf.zeros_initializer())
#             gamma = tf.get_variable('gamma', batch_var.get_shape(), tf.float32,
#                             initializer=tf.ones_initializer())
#         # BN when training
#         update = 1.0 - decay
#         if is_train:
#             update_moving_mean = moving_mean.assign_sub(update * (moving_mean - batch_mean))
#             update_moving_variance = moving_variance.assign_sub(update * (moving_variance - batch_var))
#             tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_mean)
#             tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_variance)
#
#             batch_mean, batch_var = tf.nn.moments(x,[0,1,2])
#             # print(batch_mean.get_shape())
#             train_mean = tf.assign(moving_mean, moving_mean * decay + batch_mean * (1-decay))
#             train_var = tf.assign(moving_variance, moving_variance * decay + batch_var * (1-decay))
#             with tf.control_dependencies([train_mean,train_var]):
#                 return tf.nn.batch_normalization(x, train_mean, train_var, beta, gamma, 1e-5)
#         else:
#             return tf.nn.batch_normalization(x, moving_mean, moving_variance, beta, gamma, 1e-5)

# def _bn(inputs, isTrain=True,global_step=None, moving_average_decay = 0.9, name='bn'):
#     with tf.variable_scope(name):
#
#         batch_size, feature_size,out_batch_size = inputs.get_shape()
#         bn_size = 1
#         pop_mean = tf.Variable(tf.zeros([bn_size]),trainable=False)
#         pop_var = tf.Variable(tf.ones([bn_size]),trainable=False)
#         scale = tf.Variable(tf.ones([bn_size]))
#         shift = tf.Variable(tf.zeros([bn_size]))
#         eps = 0.001
#         decay = 0.999
#         if isTrain:
#             batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
#             # print(batch_mean.get_shape())
#             train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1-decay))
#             train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1-decay))
#             with tf.control_dependencies([train_mean,train_var]):
#                 return tf.nn.batch_normalization(inputs,batch_mean,batch_var,shift,scale,eps)
#         else:
#             return tf.nn.batch_normalization(inputs,pop_mean,pop_var,shift,scale,eps)

def _bn(x, is_train, global_step=None, moving_average_decay = 0.9, name='bn'):
    # moving_average_decay = 0.99
    # moving_average_decay_init = 0.99
    with tf.variable_scope(name):
        decay = moving_average_decay
        # if global_step is None:
            # decay = moving_average_decay
        # else:
            # decay = tf.cond(tf.greater(global_step, 100)
                            # , lambda: tf.constant(moving_average_decay, tf.float32)
                            # , lambda: tf.constant(moving_average_decay_init, tf.float32))
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
        with tf.device('/CPU:0'):
            moving_mean = tf.get_variable('mu', batch_mean.get_shape(), tf.float32,
                            initializer=tf.zeros_initializer(), trainable=False)
            moving_variance = tf.get_variable('sigma', batch_var.get_shape(), tf.float32,
                            initializer=tf.ones_initializer(), trainable=False)
            beta = tf.get_variable('beta', batch_mean.get_shape(), tf.float32,
                            initializer=tf.zeros_initializer())
            gamma = tf.get_variable('gamma', batch_var.get_shape(), tf.float32,
                            initializer=tf.ones_initializer())
        # BN when training
        update = 1.0 - decay
        # with tf.control_dependencies([tf.Print(decay, [decay])]):
            # update_moving_mean = moving_mean.assign_sub(update*(moving_mean - batch_mean))
        update_moving_mean = moving_mean.assign_sub(update*(moving_mean - batch_mean))
        update_moving_variance = moving_variance.assign_sub(update*(moving_variance - batch_var))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_mean)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_variance)

        mean, var = tf.cond(is_train, lambda: (batch_mean, batch_var),
                            lambda: (moving_mean, moving_variance),name='bn_cond')
        bn = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)

        # bn = tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, 1e-5)

        # bn = tf.contrib.layers.batch_norm(inputs=x, decay=decay,
        #                                   updates_collections=[tf.GraphKeys.UPDATE_OPS], center=True,
        #                                   scale=True, epsilon=1e-5, is_training=is_train,
        #                                   trainable=True)
    return bn


## Other helper functions




import tensorflow as tf
import numpy as np
try:
    from models.read_util.read_train_data import get_all_bearings
except:
    from ..read_util.read_train_data import get_all_bearings
    # from read_train_data import get_all_bearings

VIB_SIZE = 2000
step = 20
IMAGE_SIZE = VIB_SIZE//step

# Global constants describing the CIFAR-10 data set.
NUM_JUDGE = 2
NUM_CLASSES = 4
NUM_DEGREES = 5
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1000
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1000
def read_cifar10(data_queue, fix_size=True):
    """Reads and parses examples from CIFAR10 data files.

    Recommendation: if you want N-way read parallelism, call this function
    N times.  This will give you N independent Readers reading different
    files & positions within those files, which will give better mixing of
    examples.

    Args:
      filename_queue: A queue of strings with the filenames to read from.

    Returns:
      An object representing a single example, with the following fields:
        height: number of rows in the result (32)
        width: number of columns in the result (32)
        depth: number of color channels in the result (3)
        key: a scalar string Tensor describing the filename & record number
          for this example.
        label: an int32 Tensor with the label in the range 0..9.
        uint8image: a [height, width, depth] uint8 Tensor with the image data
    """

    class CIFAR10Record(object):
        pass

    result = CIFAR10Record()
    # image_string = tf.read_file(data_queue[0])
    image_string = data_queue[0]
    # image_tensor = tf.image.decode_jpeg(image_string, channels=1, name='image_tensor')
    image_tensor = image_string


    # label_tensor = tf.constant(label,name='label_tensor')
    label_tensor = tf.cast(data_queue[1], dtype=tf.int32)
    # label_tensor = tf.cast(data_queue[1], dtype=tf.float32)

    # Dimensions of the images in the CIFAR-10 dataset.
    # input format.
    label_bytes = 1  # 2 for CIFAR-100
    result.depth = 1
    # Read a record, getting filenames from the filename_queue.  No
    # header or footer in the CIFAR-10 format, so we leave header_bytes
    # and footer_bytes at their default of 0.
    # The first bytes represent the label, which we convert from uint8->int32.
    # result.label = label_tensor[0]
    result.label = label_tensor


    # Convert from [depth, height, width] to [height, width, depth].
    # result.uint8image = tf.transpose(depth_major, [1, 2, 0])
    result.uint8image = image_tensor

    result.filename = tf.cast(data_queue[0], dtype=tf.string)

    return result



def _generate_image_and_label_batch(image, label1, min_queue_examples,
                                    batch_size, shuffle=True):
    """Construct a queued batch of images and labels.

    Args:
      image: 3-D Tensor of [height, width, 3] of type.float32.
      label: 1-D Tensor of type.int32
      min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
      batch_size: Number of images per batch.
      shuffle: boolean indicating whether to use a shuffling queue.

    Returns:
      images: Images. 4D tensor of [batch_size, height, width, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 256
    if shuffle:
        # Imagine inputs is a list or tuple of tensors representing single training example.
        # In my case, inputs is a tuple (features, label) obtained by reading TFRecords.
        # dtypes = list(map(lambda x: x.dtype, inputs))
        # shapes = list(map(lambda x: x.get_shape(), inputs))
        # queue = tf.RandomShuffleQueue(CAPACITY, MIN_AFTER_DEQUEUE, dtypes)
        # enqueue_op = queue.enqueue(inputs)
        # qr = tf.train.QueueRunner(queue, [enqueue_op] * NUM_THREADS)
        # tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS, qr)
        # inputs = queue.dequeue()
        # for tensor, shape in zip(inputs, shapes):
        #     tensor.set_shape(shape)

        # Now you can use tf.train.batch with dynamic_pad=True, and the order in which
        # it enqueues elements will be permuted because of RandomShuffleQueue.
        # inputs_batch = tf.train.batch(inputs, batch_size, capacity=min_queue_examples + 3 * batch_size,
        #                             dynamic_pad=True, name=name)

        # images, label_batch1, label_batch2, label_batch3 = tf.train.batch(
        #     [image, label1, label2, label3], batch_size,
        #     num_threads=num_preprocess_threads,
        #     capacity=min_queue_examples + 3 * batch_size,
        #     dynamic_pad=True, name='batch')

        images, label_batch1 = tf.train.shuffle_batch(
            [image, label1],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch1 = tf.train.batch(
            [image, label1],
            batch_size=batch_size,
            num_threads=num_preprocess_threads//2,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    # tf.summary.image('images', images)

    return images, tf.reshape(label_batch1, [batch_size, 1])

import csv
import random

#fft
def fft(x_vibration):
    x_freq = np.fft.fft(x_vibration)
    x_freqabs = abs(x_freq)
    return x_freqabs
# 高斯噪声
def wgn(length, mu=0, sigma=0.1):
    noise = [random.gauss(mu, sigma) for i in range(length)]
    return np.array(noise)
def read_from_string_func(name):
    # 旋转角度范围
    name = np.reshape(name, [1,])
    # print('name',name)
    csv_reader = csv.reader(open(name[0]))
    vib = []
    for row in csv_reader:
        vib.append(float(row[4]))
    # print('vib:',vib)
    max_size = 2560 - VIB_SIZE
    first = random.randint(0,max_size)
    vib = np.array(vib)
    vib = vib[first:first+VIB_SIZE]

    noise = wgn(VIB_SIZE)
    vib += noise
    # vib = vib.reshape([VIB_SIZE//step,step])
    # vib = np.max(vib,axis=1)

    freq = fft(vib)

    freq = freq.reshape([VIB_SIZE//step,step])
    freq = np.max(freq,axis=1)

    freq = np.expand_dims(freq, axis=1)
    # print(vib.dtype.name)
    freq = freq.astype(np.float32)
    # print(vib.dtype.name)
    return freq
def distorted_inputs(data_dir, batch_size=32, fix_size=True):
    """Construct distorted input for CIFAR training using the Reader ops.

    Args:
      data_dir: Path to the CIFAR-10 data directory.
      batch_size: Number of images per batch.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    filename_list, label_list = get_all_bearings(data_dir)
    print('the number of train data is:', len(filename_list))

    # (full_name, 1, class_dict[label], degree_dict[degree])
    filename_list = np.expand_dims(filename_list,axis=1)
    label_list = np.expand_dims(label_list,axis=1)

    data = np.concatenate([filename_list, label_list], axis=1)
    np.random.shuffle(data)
    # print(data)
    filenames = data[:, 0]
    labels = data[:, 1:]
    # labels = data[:, 1]

    labels = labels.astype(np.int32)
    # print(filenames)
    # print(labels)
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    # filename_queue = tf.train.string_input_producer(filenames)
    # input_queue = tf.train.slice_input_producer([filenames,labels])
    input_queue = tf.train.slice_input_producer([filenames, labels[:, 0:]])

    with tf.name_scope('data_augmentation'):
        # Read examples from files in the filename queue.
        read_input = read_cifar10(input_queue, fix_size=fix_size)


        vibration = tf.py_func(read_from_string_func, [read_input.uint8image], tf.float32)

        # image_rotate = tf.py_func(random_rotate_90_image_func, [read_input.uint8image], tf.uint8)
        # image_rotate = tf.py_func(random_rotate_image_func, [image_rotate], tf.uint8)

        # reshaped_image = tf.cast(image_rotate, tf.float32)

        # height = IMAGE_SIZE - 24
        # width = IMAGE_SIZE - 24

        # Randomly flip the image horizontally.

        # Because these operations are not commutative, consider randomizing
        # the order their operation.
        # NOTE: since per_image_standardization zeros the mean and makes
        # the stddev unit, this likely has no effect see tensorflow#1458.
        if fix_size:
            # Set the shapes of tensors.
            vibration.set_shape([IMAGE_SIZE,1])
        else:
            picture = read_input.uint8image
            shape = picture.shape
            # print(shape)
            vibration.set_shape(shape)

        read_input.label.set_shape([1,])

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                                 min_fraction_of_examples_in_queue)
        print('Filling queue with %d CIFAR images before starting to train. '
              'This will take a few minutes.' % min_queue_examples)
    if fix_size:
        # Generate a batch of images and labels by building up a queue of examples.
        return _generate_image_and_label_batch(vibration, read_input.label,
                                               min_queue_examples, batch_size,
                                               shuffle=True)
    else:
        return vibration, read_input.label


def test_read_from_string_func(name):
    # 旋转角度范围
    name = np.reshape(name, [1,])
    # print('name',name)
    csv_reader = csv.reader(open(name[0]))
    vib = []
    for row in csv_reader:
        vib.append(float(row[4]))
    # print('vib:',vib)
    max_size = 2560 - IMAGE_SIZE
    first = random.randint(0,max_size)
    vib = np.array(vib)
    vib = vib[first:first+IMAGE_SIZE]
    noise = wgn(IMAGE_SIZE)
    vib += noise
    freq = fft(vib)

    freq = np.expand_dims(freq, axis=1)
    # print(vib.dtype.name)
    freq = freq.astype(np.float32)
    # print(vib.dtype.name)
    return freq
def test_inputs(data_dir, batch_size=32, fix_size=True):
    """Construct distorted input for CIFAR training using the Reader ops.

    Args:
      data_dir: Path to the CIFAR-10 data directory.
      batch_size: Number of images per batch.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    filename_list, label_list = get_all_bearings(data_dir)
    print('the number of test data is:', len(filename_list))

    filename_list = np.expand_dims(filename_list,axis=1)
    label_list = np.expand_dims(label_list,axis=1)

    data = np.concatenate([filename_list, label_list], axis=1)

    np.random.shuffle(data)

    # print(data)
    filenames = data[:, 0]
    labels = data[:, 1:]
    # labels = data[:, 1]

    labels = labels.astype(np.int32)
    # print(filenames)
    # print(labels)
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    # filename_queue = tf.train.string_input_producer(filenames)
    # input_queue = tf.train.slice_input_producer([filenames,labels])
    input_queue = tf.train.slice_input_producer([filenames, labels[:, 0:]])

    with tf.name_scope('data_augmentation_test'):
        # Read examples from files in the filename queue.
        read_input = read_cifar10(input_queue, fix_size=fix_size)


        vibration = tf.py_func(test_read_from_string_func, [read_input.uint8image], tf.float32)

        # Randomly flip the image horizontally.

        # Because these operations are not commutative, consider randomizing
        # the order their operation.
        # NOTE: since per_image_standardization zeros the mean and makes
        # the stddev unit, this likely has no effect see tensorflow#1458.
        if fix_size:
            # Set the shapes of tensors.
            vibration.set_shape([IMAGE_SIZE,1])
        else:
            picture = read_input.uint8image
            shape = picture.shape
            # print(shape)
            vibration.set_shape(shape)

        read_input.label.set_shape([1,])

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                                 min_fraction_of_examples_in_queue)
        print('Filling queue with %d CIFAR images before starting to train. '
              'This will take a few minutes.' % min_queue_examples)
    if fix_size:
        # Generate a batch of images and labels by building up a queue of examples.
        return _generate_image_and_label_batch(vibration, read_input.label,
                                               min_queue_examples, batch_size,
                                               shuffle=True)
    else:
        return vibration, read_input.label



def test_func():
    # path = r'G:\Bearing\bearing\data_set\FEMTOBearingDataSet\Training_set\Learning_set'
    path = '/home/sunyaqiang/Myfile/bearing/data_set/FEMTOBearingDataSet/Training_set/Learning_set/Bearing1_1/acc_00001.csv'

    # distorted_inputs(path)
    read_from_string_func(path)
    # print(wgn(100))
    pass

if __name__=="__main__":
    test_func()

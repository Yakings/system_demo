import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
#os.environ["KMP_SETTINGS"] = str()

# os.environ["KMP_BLOCKTIME"] = str(10)
# os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
# os.environ["OMP_NUM_THREADS"]= str(4)

import tensorflow as tf
from tensorflow.python.platform import gfile

# try:
#     from . import DetectionLiveCPythonInterface as detectionlive
# except:
#     import DetectionLiveCPythonInterface as detectionlive

# import sys




import csv
import random
VIB_SIZE=2000
step = 20
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



class ClassifierGraph():
    def __init__(self):
        with tf.Session() as sess:
            # model_filename = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/model/mobilenet_optimizedquantized.pb'

            #################################
            ####  pb　file path   ####
            ###############################
            # model_filename ='/home/yaqiang/MyFile/auto_train/Train_dirs/images_latest/images_latest_mobilenet_quaintized.pb'
            model_filename ='./train/output/foxcoon_binary_clas_quaintized.pb'
            with gfile.FastGFile(model_filename, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                g_in = tf.import_graph_def(graph_def)
            self.sess = sess
            self.graph = sess.graph
            self.images = sess.graph.get_tensor_by_name("import/image:0")
            self.prediction = sess.graph.get_tensor_by_name("import/tower_0/logits_1/fc/BiasAdd:0")
            # writer = tf.summary.FileWriter(".\Summary_file", sess.graph)
            node = self.graph.as_graph_def().node
            # print([_.name for _ in node])
            # self.prediction = sess.graph.get_tensor_by_name("Predictions/Softmax:0")
        tf.reset_default_graph()
        # for i in range(len(label_list_origin)):
        #     label_list.append(None)
        #     label_list_pinyin.append(None)
        # for idx in range(len(shuffle_table)):
        #     train_idx = shuffle_table[idx][0]
        #     doc_idx = shuffle_table[idx][1]
        #     label_list[train_idx] = label_list_origin[doc_idx-1]
        #     label_list_pinyin[train_idx] = label_list_origin_pinyin[doc_idx - 1]
        #     hashtable[train_idx] = doc_idx

    def run(self, img):
        feed_dict = {self.images: img}
        res = self.sess.run(self.prediction, feed_dict=feed_dict)
        #print("time= ", t2-t1)
        # sort = res[0].argsort()
        # top5 = sort[-4:][::-1]
        # return top5, res[0][top5]
        return res
import numpy as np

import time
if __name__ == "__main__":

    classifier = ClassifierGraph()
    # img = [[[1] for i in range(100)]]
    img = read_from_string_func(['/home/sunyaqiang/Myfile/bearing/data_set/FEMTOBearingDataSet'
                                 '/Training_set/Learning_set/Bearing1_1/acc_02705.csv'])
    start = time.clock()
    res = classifier.run(img)
    end = time.clock()
    res = classifier.run(img)
    print('time:', end - start)
    print('result:', res)


    # for i in range(100):
    #     img = [[[i] for j in range(100)]]
    #
    #     start = time.clock()
    #     res = classifier.run(img)
    #     end = time.clock()
    #     res = classifier.run(img)
    #     print('time:', end - start)
    #     print('result:',res)

    # from PIL import Image
    # classifier = ClassifierGraph()
    # print(0)
    #
    # ###############################
    # ##pictures dictionary
    # ##############################
    # # rootDir1 = '/home/yaqiang/MyFile/auto_train/Datasets/images_latest/1/'
    # count = 3
    # # rootDir1 = '/home/yaqiang/MyFile/auto_train/Datasets/images_latest_test/'+str(count)+'/'
    # rootDir1 = 'G:\PythonWork\auto_train\Datasets\images_latest\ALKAOUA'
    #
    #
    #
    #
    # lists1 = os.listdir(rootDir1)
    # counter =0
    # for i in range(0, len(lists1)):
    #     l1 = lists1[i]
    #     name = os.path.join(rootDir1, l1)
    #     # img = cv2.imread(name)
    #     #
    #     # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #     # #
    #     # img = cv2.resize(img, (224, 224))
    #     # # # img = preprocess_for_eval(img, 224,224)
    #     # #
    #
    #     img = Image.open(name)
    #
    #
    #     # img = img.crop(239,0,)
    #
    #
    #     img = img.resize((224, 224))
    #     img = np.asanyarray(img, dtype=np.float32)/255.*2-1
    #     img = np.expand_dims(img, 0)
    #
    #     # img = np.reshape(img, [1,224,224,3])
    #     # img = img / (255 * 1.0)
    #
    #
    #     start = time.clock()
    #     top5, confidence = classifier.run(img)
    #     end = time.clock()
    #     print('time:',end-start)
    #     # print(l1)
    #     print(top5)
    #     # print(confidence)
    #     if top5[0] == count:
    #         counter += 1
    #
    # print('acc:', counter*1.0/len(lists1))

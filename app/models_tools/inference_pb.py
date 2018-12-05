import os
# import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''

#os.environ["KMP_SETTINGS"] = str()

# os.environ["KMP_BLOCKTIME"] = str(10)
# os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
# os.environ["OMP_NUM_THREADS"]= str(4)

import tensorflow as tf
from tensorflow.python.platform import gfile
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# try:
#     from . import DetectionLiveCPythonInterface as detectionlive
# except:
#     import DetectionLiveCPythonInterface as detectionlive

# import sys




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
def read_from_string_func(name,input_dim):
    # 旋转角度范围
    # name = np.reshape(name, [1,])
    # print('name',name)
    csv_reader = csv.reader(open(name[0]))
    vib = [float(row[4]) for row in csv_reader]
    # for row in csv_reader:
    #     vib.append(float(row[4]))
    # print('vib:',vib)
    vib =vib[:2000]
    max_size = len(vib)
    set_max_size = (max_size//input_dim)*input_dim
    # div_size = max_size-set_max_size
    # first = random.randint(0,div_size)
    first = 0
    vib = np.array(vib)
    vib = vib[first:first+set_max_size]

    # noise = wgn(set_max_size)
    # vib += noise
    # vib = vib.reshape([VIB_SIZE//step,step])
    # vib = np.max(vib,axis=1)

    freq = fft(vib)
    step = set_max_size//input_dim
    freq = freq.reshape([set_max_size//step,step])
    freq = np.max(freq,axis=1)

    # freq = np.expand_dims(freq, axis=1)
    # print(vib.dtype.name)
    freq = freq.astype(np.float32)
    return freq

class ClassifierGraph():
    def __init__(self,model_name='foxcoon_binary_clas_quaintized.pb'):
        self.model_name=model_name
        with tf.Session() as sess:
            # model_filename = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/model/mobilenet_optimizedquantized.pb'

            #################################
            ####  pb　file path   ####
            ###############################
            # model_filename ='/home/yaqiang/MyFile/auto_train/Train_dirs/images_latest/images_latest_mobilenet_quaintized.pb'
            model_filename =os.path.join('data/model',self.model_name)
            with gfile.FastGFile(model_filename, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                g_in = tf.import_graph_def(graph_def)
            self.sess = sess
            self.graph = sess.graph
            self.images = sess.graph.get_tensor_by_name("import/image:0")
            # self.prediction = sess.graph.get_tensor_by_name("import/tower_0/sqeeze_output:0")
            self.prediction = sess.graph.get_tensor_by_name("import/tower_0/Squeeze_1:0")
            # writer = tf.summary.FileWriter(".\Summary_file", sess.graph)
            # node = self.graph.as_graph_def().node
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
        img = np.expand_dims(img, axis=1)
        feed_dict = {self.images: [img]}
        res = self.sess.run(self.prediction, feed_dict=feed_dict)
        #print("time= ", t2-t1)
        # sort = res[0].argsort()
        # top5 = sort[-4:][::-1]
        # return top5, res[0][top5]
        return res
import numpy as np
# import time



def predict_rul_pb(csv_path,classifier,input_dim):
    # start = time.clock()
    # print(csv_path)
    img = read_from_string_func([csv_path],input_dim)
    # start2 = time.clock()
    # print(img[0])
    res = classifier.run(img)
    # end = time.clock()
    # res = classifier.run(img)
    # print('all time:', end - start)
    # print('dl time:', end - start2)
    # print('result:', res)
    return res

if __name__ == "__main__":
    import time
    classifier = ClassifierGraph(r'G:\Bearing\app\data\model\foxcoon444_DM200DM.pb')
    for i in range(100000000):
        # img = [[[1] for i in range(100)]]
        start_0 = time.clock()
        for i in range(64):
            img = read_from_string_func([r'G:\Bearing\bearing_data\watch\acc_00704.csv'],input_dim=200)
        start = time.clock()
        res = classifier.run(img)
        end = time.clock()
        res = classifier.run(img)
        # print('time fft:', start - start_0)
        # print('net time:', end - start)
        # print('result:', res)


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

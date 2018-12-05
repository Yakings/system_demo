from sklearn import svm
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import tree
from sklearn import neighbors

from sklearn.externals import joblib


import os
import time

import csv
import random

class ClassifierMachine():
    def __init__(self,model_name='foxcoon_binary_clas_quaintized.pb'):
        self.model_name=model_name
        self.clf = joblib.load(os.path.join('data/model',self.model_name))

    def run(self, img):
        res = self.clf.predict([img])
        return [res]


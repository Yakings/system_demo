# Copyright 2018 The Authors Sunyaqiang. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the triditional machine learning model.

Summary of available functions:

"""
# pylint: disable=missing-docstring
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

from sklearn import svm
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import tree
from sklearn import neighbors

from sklearn.externals import joblib
import os

# Nu SVM
def NuSVR_train(x,y):
    clf = svm.NuSVR()
    clf.fit(x, y)
    return clf
def SVM_train(x,y):
    svr_rbf = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
    # svr_lin = svm.SVR(kernel='linear', C=1e3)
    # svr_poly = svm.SVR(kernel='poly', C=1e3, degree=2)
    # clf = svm.SVR()
    clf = svr_rbf
    clf.fit(x, y)

    return clf
# 线性回归
def LinearRegression_train(x,y):
    clf = linear_model.LinearRegression()
    clf.fit(x, y)
    return clf
# 逻辑回归
def LogisticRegression_train(x,y):
    clf = linear_model.LogisticRegression()
    clf.fit(x, y)
    return clf
# 高斯伯努利
def GaussianNB_train(x,y):
    clf = naive_bayes.GaussianNB()
    clf.fit(x, y)
    return clf
def MultinomialNB_train(x,y):
    clf = naive_bayes.MultinomialNB()
    clf.fit(x, y)
    return clf
def BernoulliNB_train(x,y):
    clf = naive_bayes.BernoulliNB()
    clf.fit(x, y)
    return clf
# 决策树回归
def DecisionTreeClassifier_train(x,y):
    # clf = tree.DecisionTreeRegressor(criterion='gini', max_depth=None,
    #                                   min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
    #                                   max_features=None, random_state=None, max_leaf_nodes=None,
    #                                   min_impurity_split=None,
    #                                   presort=False)
    clf = tree.DecisionTreeRegressor()
    clf.fit(x, y)
    return clf
# k近邻回归
def KNeighborsRegressor_train(x,y):
    clf = neighbors.KNeighborsRegressor(n_neighbors=5, n_jobs=1) # 回归
    clf.fit(x, y)
    return clf


# inference
def inference(x,clf):
    print(clf.coef_)
    pre_y = clf.predict(x)
    return pre_y

# save model
def save_model(clf, path,name='mist_binary_clas_ml_quaintized'):
    pname = os.path.join(path,name+'.model')
    joblib.dump(clf, pname)
# restore model
def resotre_model(path, name):
    clf = joblib.load(path+name+'.model')
    return clf

def test():
    import numpy as np
    from sklearn.svm import SVR
    import matplotlib.pyplot as plt

    # #############################################################################
    # Generate sample data
    X = np.sort(5 * np.random.rand(40, 1), axis=0)
    y = np.sin(X).ravel()

    # #############################################################################
    # Add noise to targets
    y[::5] += 3 * (0.5 - np.random.rand(8))

    # #############################################################################
    # Fit regression model
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    y_rbf = svr_rbf.fit(X, y).predict(X)
    y_lin = svr_lin.fit(X, y).predict(X)
    y_poly = svr_poly.fit(X, y).predict(X)

    # #############################################################################
    # Look at the results
    lw = 2
    plt.scatter(X, y, color='darkorange', label='data')
    plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
    # plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
    # plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('Support Vector Regression')
    # plt.legend()
    plt.show()


import math
import csv
def get_file(file_name):
    with open(file_name) as f:
            reader = csv.reader(f)
            loss = [row[2] for row in reader]
            loss = loss[1:]
            loss = np.array(loss, np.float)
    loss = loss/4
    return loss

    pass
def select_step():
    import numpy as np
    from sklearn.svm import SVR
    import matplotlib.pyplot as plt

    # x = [[i*0.01] for i in range(1000)]
    # y = [math.sin(x[i][0]) for i in range(1000)]
    # filename = './run_0-tag-val_mse_loss.csv'
    filename = './run_0-tag-val_loss.csv'
    y = get_file(filename)[:60]
    x = [[i+1] for i in range(len(y))]
    x_v = [[1/i, math.log(i),1] for i in range(1,len(y)+1)]
    clf = LinearRegression_train(x_v,y)
    # clf = KNeighborsRegressor_train(x,y)
    # clf = DecisionTreeClassifier_train(x,y)
    # clf = SVM_train(x,y)


    # test_x = [[i*0.01] for i in range(1000)]
    test_y = inference(x_v,clf)
    print(test_y)
    lw = 2
    # plt.scatter(x, y, color='darkorange', label='data')
    plt.plot(x, y, color='darkorange', label='data')
    plt.plot(x, test_y, color='navy', lw=lw, label='RBF model')
    plt.show()
    # test()
if __name__ == '__main__':
    import numpy as np
    from sklearn.svm import SVR
    import matplotlib.pyplot as plt

    # x = [[i*0.01] for i in range(1000)]
    # y = [math.sin(x[i][0]) for i in range(1000)]
    # filename = './run_0-tag-val_mse_loss.csv'
    filename = './run_0-tag-val_loss.csv'
    y = get_file(filename)[:60]
    x = [[i+1] for i in range(len(y))]
    x_v = [[1/i, math.log(i),1.0] for i in range(1,len(y)+1)]
    # clf = LinearRegression_train(x_v,y)
    # clf = KNeighborsRegressor_train(x,y)
    # clf = DecisionTreeClassifier_train(x,y)
    clf = SVM_train(x,y)


    # test_x = [[i*0.01] for i in range(1000)]
    test_y = inference(x_v,clf)
    print(test_y)
    lw = 2
    y_p = [[87247.40751326/i+11090.26631271*math.log(i)+146394.8275374809] for i in range(1,len(y)+1)]
    print(y_p)

    # plt.scatter(x, y, color='darkorange', label='data')
    plt.plot(x, y, color='darkorange', label='data')
    plt.plot(x, test_y, color='navy', lw=lw, label='RBF model')
    plt.plot(x, y_p, color='b', lw=lw, label='RBF model')
    plt.show()
    # test()

    pass
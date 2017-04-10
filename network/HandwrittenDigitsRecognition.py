#!/usr/bin/python
#-*- encoding:utf-8 -*-
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import LabelBinarizer
from neuralNetwork import NeuralNetWork
from sklearn.cross_validation import train_test_split

digits = load_digits()
X = digits.data
y = digits.target
#  X转化到0-1之间,要求的
X -= X.min()
X /= X.max()
#print X
#print X.shape  # 1797*64

nn = NeuralNetWork([64,100,10],'logistic')   #8*8的图片，有64个像素值需要64个输入单元，64列表示每个实例64个特征，100个隐藏层神经元，10个输出单元
X_train,X_test,y_train,y_test= train_test_split(X,y)
labels_train = LabelBinarizer().fit_transform(y_train)
labels_test = LabelBinarizer.fit_transform(y_test)
print labels_train
print labels_test
print "start fitting"
nn.fit(X_train,labels_train,epochs=3000)
prediction = []   #预测的数字
for i in range(X_test.shape[0]):
    o = nn.predict(X_test[i])
    prediction.append(np.argmax(o))


print confusion_matrix(y_test,prediction)
print classification_report(y_test,prediction)

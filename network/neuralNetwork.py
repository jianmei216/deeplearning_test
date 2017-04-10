#!/usr/bin/python
#-*- encoding:utf-8 -*-

import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0 - np.tanh(x)*np.tanh(x)

def logistic(x):
    return 1/(1+np.exp(-x))

def logistic_deriv(x):
    return logistic(x)*(1-logistic(x))


class NeuralNetWork:
    def __init__(self,layers,activation='tanh'):
        """
        :param layers: A list containing the number of units in each layer,should bu two value at least
        :param activation: The activation function to be used.Can be 'logistic' or 'tanh'
        """

        if activation == 'logistic':
            self.activation =logistic
            self.activation_deriv = logistic_deriv
        elif activation=='tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv

        self.weights = []
        for i in range(1,len(layers) - 1):
            self.weights.append((2*np.random.random(layers[i-1]+1,layers[i]+1)-1)*0.25)
            self.weights.append((2*np.random.random(layers[i]+1,layers[i+1])-1)*0.25)

    def fit(self,X,y,learning_rate=0.2,epochs=1000):
        #X 输入：（10行100列）:每一行一个实例，有很多特征值，10个实例，每个实例100个特征值，epochs:最多循环次数s
        # y 输出：是一列
        X = np.atleast_2d(X)   #确认X至少是二位的数组。 如果是图片，RGB是三维的
        temp = np.ones([X.shape[0],X.shape[1]+1])    #初始化一个数组，值都是1，行数=X行数，列数=X列数+1，用来初始化bias的
        temp[:,0:-1] = X  # 行：取X所有的行，列：第一列到除了最后一列
        X = temp
        y = np.array(y)   #标准成np的数组

        for k in range(epochs):
            i = np.random.randint(X.shape[0])   #随机取一个样本（一行）进行训练，更新神经网络
            a = [X[i]]

            for l in range(len(self.weights)):  # 计算隐藏层和输出层所有神经元的值
                a.append(self.activation(np.dot(a[l],self.weights[l])))  #点积，a增加一行，将没一行的输入算出隐藏层的没个神经元的值

            error = y[i] - a[-1]    #a[-1]是最后一层即输出层数组，error=每个实例的误差
            deltas = [error * self.activation_deriv(a[-1])]   #self.activation_deriv(a[-1]) 是最后输出层的偏差

            # start backprobagation
            for l in range(len(a)-2,0,1):   #从最后一层到输出层,反向更新每个实例的神经元
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))

            deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i]+=learning_rate*layer.T.dot(delta)

    # 更正weight
    def predict(self,x):
        x = np.array(x)
        temp = np.ones(x.shape[0]+1)
        temp[0:-1] = x
        a = temp
        for l in range(0,len(self.weights)):
            a = self.activation(np.dot(a,self.weights[l]))
        return a
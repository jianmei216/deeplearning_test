#!/usr/bin/python
#-*- encoding:utf-8 -*-

import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

# public define
py_dir = os.path.dirname(os.path.realpath(__file__))
project_dir = os.path.dirname(py_dir)
img_from_dir = project_dir+'/files/'
img_from_file = img_from_dir+ 'a.png'
img_to_dir = project_dir+'/filesoutput/letters_lower/'


#green=np.uint8([[[255,255,255]]])
#hsv_green=cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
#print hsv_green

X = np.matrix([[2,3,4],
    [4,6,5]])
temp = np.ones([X.shape[0],X.shape[1]+1])    #初始化一个数组，值都是1，行数=X行数，列数=X列数+1，用来初始化bias的
temp[:,0:-1] = X
print  temp
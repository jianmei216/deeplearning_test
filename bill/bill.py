#!/usr/bin/python
#-*- encoding:utf-8 -*-

import numpy as np
import cv2
import os

# public define
py_dir = os.path.dirname(os.path.realpath(__file__))
project_dir = os.path.dirname(py_dir)


# start read image
img = cv2.imread(project_dir+'/files/bill.jpg')


#img[:,:,2]=0   #使所有像素的红色通道值都为 0



#Applying Grayscale filter to image 作用Grayscale（灰度）过滤器到图像上,保存过滤过的图像到新文件中
cv2.imwrite('graytest.jpg',cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
cv2.imwrite('hsvtest.jpg',cv2.cvtColor(img, cv2.COLOR_BGR2HSV))

print img.shape

#end show image
cv2.imshow('image',img)
k = cv2.waitKey(0)   # 窗口关闭的情况
if k==27:   # wait for esc or return key to exit
    cv2.destroyAllWindows()
elif k==ord('s'):   # wait for 's' to save and exit
    cv2.imwrite(project_dir+'/filesoutput/saveimg.png',img)
    cv2.destroyAllWindows()
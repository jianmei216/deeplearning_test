#!/usr/bin/python
#-*- encoding:utf-8 -*-

import numpy as np
import cv2

#读取图像
img = cv2.imread('files/invoice.jpg',cv2.IMREAD_UNCHANGED)

#img[:,:,2]=0   #使所有像素的红色通道值都为 0


#显示图像
cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# 窗口关闭的情况
k = cv2.waitKey(0)
if k==27:   # wait for esc or return key to exit
    cv2.destroyAllWindows()
elif k==ord('s'):   # wait for 's' to save and exit
    cv2.imwrite('saveimg.png',img)
    cv2.destroyAllWindows()

#Applying Grayscale filter to image 作用Grayscale（灰度）过滤器到图像上
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#保存过滤过的图像到新文件中
cv2.imwrite('graytest.jpg',gray)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imwrite('hsvtest.jpg',gray)
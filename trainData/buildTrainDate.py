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
img_to_dir = project_dir+'/filesoutput/'

letters_up = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
letters_low =['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
number = ['0','1','2','3','4','5','6','7','8','9']

# create letters,numbers
def create_img(letter,path):
    img = np.zeros((200, 200, 4), np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.putText(img, letter,(50, 100),font, 4, (255, 255, 255), 2)
    cv2.imwrite(path + letter+'.png', img)
    img = cv2.imread(path + letter+'.png')
    cv2.imwrite(path + letter+'.png', img)

def createImg():
    for l in letters_up:
        create_img(l,img_from_dir+'/letters_upper/')
    for i in letters_low:
        create_img(i,img_from_dir+'/letters_lower/')
    for i in number:
        create_img(i,img_from_dir + '/number/')

def show_img(imgwindowname,img):
    cv2.namedWindow(imgwindowname)
    cv2.imshow(imgwindowname, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def img_turn(name,sourcepath,targetpath):
    img = cv2.imread(sourcepath + name + '.png')
    #放大缩小
    #res = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    #cv2.imwrite(targetpath + name+'_resize.png', res)

    #倾斜
    rows, cols, coler = img.shape
    for v in range(1,30):
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), v*6, 0.6)
        dst = cv2.warpAffine(img, M, (cols, rows))
        cv2.imwrite(targetpath +name +'_'+str(v*6)+'.png', dst)
    #show_img('dst', dst)

    # 垂直翻转
    flipped = cv2.flip(img, 1)
    cv2.imwrite(targetpath +name + '_flip1.png', flipped)

    #水平翻转
    flipped = cv2.flip(img, 0)
    cv2.imwrite(targetpath +name + '_flip0.png', flipped)

    # 图像水平垂直翻转
    flipped = cv2.flip(img, -1)
    cv2.imwrite(targetpath +name + '_flip2.png', flipped)

# 仿射变换
def img_affineTransform(name,sourcepath,targetpath):
    img = cv2.imread(sourcepath + name + '.png')
    rows, cols, ch = img.shape
    for i in range(1,10):
        rand = i*5
        pts1 = np.float32([[rand, rand], [200, 50], [50, 200]])
        pts2 = np.float32([[10, 100], [200, 50], [100, 200]])
        M = cv2.getAffineTransform(pts1, pts2)
        dst = cv2.warpAffine(img, M, (cols, rows))
        cv2.imwrite(targetpath +name+ '_transform'+str(rand)+'.png', dst)
        #plt.subplot(100, 200)
        #plt.subplot(121, plt.imshow(img), plt.title('Output'))
        plt.show()


def img_shijueTransform(name,sourcepath,targetpath):
    rows, cols, ch = img.shape
    for i in range(1,10):
        rand = i*5
        pts1 = np.float32([[10, 50], [10, 100], [15, 150], [150, 150]])
        pts2 = np.float32([[0, 0], [200, 0], [0, 200], [200, 200]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, M, (200, 200))
        cv2.imwrite(targetpath + name + '_sjtransform' + str(rand) + '.png', dst)
        #cv2.imwrite(img_to_dir + 'A_7.png', dst)


def adaptiveThreshold(img):
    img = cv2.medianBlur(img, 5) # 中值滤波
    ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # 11 为 Blocksize,2 为 C 值
    th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 11, 2)
    th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)
    titles = ['Original Image', 'Global Thresholding (v = 127)',
              'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [img, th1, th2, th3]

    cv2.imwrite(img_to_dir + 'A_9.png', th1)
    cv2.imwrite(img_to_dir + 'A_10.png', th2)
    cv2.imwrite(img_to_dir + 'A_11.png', th3)
    for i in xrange(4): plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
    plt.show()


def simpleThreshold(img):
    ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
    titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
    for i in xrange(6):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
        cv2.imwrite(img_to_dir +'A_simple_'+str(i)+'.png', images[i])

    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
    plt.show()

def blur(img):
    blur = cv2.blur(img, (15, 15))
    plt.subplot(121), plt.imshow(img), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(blur), plt.title('Blurred')
    plt.xticks([]), plt.yticks([])
    plt.show()


def addGussiaNoise(name,sourcepath,targetpath):
    img = cv2.imread(sourcepath + name + '.png',0)
    #param = 300
    for i in range(1,25):
        param = 8*i

        # 灰阶范围
        grayscale = 256
        w = img.shape[1]
        h = img.shape[0]
        newimg = np.zeros((h, w), np.uint8)

        for x in xrange(0, h):
            for y in xrange(0, w, 2):
                r1 = np.random.random_sample()
                r2 = np.random.random_sample()
                z1 = param * np.cos(2 * np.pi * r2) * np.sqrt((-2) * np.log(r1))
                z2 = param * np.sin(2 * np.pi * r2) * np.sqrt((-2) * np.log(r1))

                fxy = int(img[x, y] + z1)
                fxy1 = int(img[x, y + 1] + z2)
                # f(x,y)
                if fxy < 0:
                    fxy_val = 0
                elif fxy > grayscale - 1:
                    fxy_val = grayscale - 1
                else:
                    fxy_val = fxy
                # f(x,y+1)
                if fxy1 < 0:
                    fxy1_val = 0
                elif fxy1 > grayscale - 1:
                    fxy1_val = grayscale - 1
                else:
                    fxy1_val = fxy1
                newimg[x, y] = fxy_val
                newimg[x, y + 1] = fxy1_val
        cv2.imwrite(targetpath + name + '_gussianoise_'+str(param)+'.png', newimg)

    '''
    彩色图像添加高斯噪音
    param=30
    #灰阶范围
    grayscale=256
    w=img.shape[1]
    h=img.shape[0]
    newimg=np.zeros((h,w,3),np.uint8)
    
    for x in xrange(0,h):
        for y in xrange(0,w,2):
            r1=np.random.random_sample()
            r2=np.random.random_sample()
            z1=param*np.cos(2*np.pi*r2)*np.sqrt((-2)*np.log(r1))
            z2=param*np.sin(2*np.pi*r2)*np.sqrt((-2)*np.log(r1))    
            .........
            .........
            newimg[x,y,0]=fxy_val_0
            newimg[x,y,1]=fxy_val_1
            newimg[x,y,2]=fxy_val_2
            newimg[x,y+1,0]=fxy1_val_0
            newimg[x,y+1,1]=fxy1_val_1
            newimg[x,y+1,2]=fxy1_val_2
    '''

def addSaltNoise(name,sourcepath,targetpath):
    img = cv2.imread(sourcepath + name + '.png')
    #coutn = 10000
    for number in range(1,20):
        coutn = 400*number
        # 循环添加n个椒盐
        for k in range(coutn):
            # 随机选择椒盐的坐标
            i = int(np.random.random() * img.shape[1])
            j = int(np.random.random() * img.shape[0])
            # 如果是灰度图
            if img.ndim == 2:
                img[j, i] = 255
            # 如果是RBG图片
            elif img.ndim == 3:
                '''
                img[j, i, 0] = 255
                img[j, i, 1] = 255
                img[j, i, 2] = 255
                '''
                img[j, i, 0] = 25
                img[j, i, 1] = 20
                img[j, i, 2] = 20
        cv2.imwrite(targetpath + name + '_saltnoise_' + str(coutn) + '.png', img)


# 1 Create a black image
#createImg()

# 原图
#img = cv2.imread(img_from_dir+'letters_upper/'+'A.png')
#img0 = cv2.imread(img_from_dir+'letters_upper/'+'A.png',0)
#show_img('A',img)

# 2 几何变换 缩放，平移，旋转

for l in letters_up:
    img_turn(l,img_from_dir+'letters_upper/',img_to_dir+'letters_upper/')
for l in letters_low:
    img_turn(l, img_from_dir + 'letters_lower/', img_to_dir + 'letters_lower/')
for l in number:
    img_turn(l, img_from_dir + 'number/', img_to_dir + 'number/')

# 3 AffineTransform 放射变换
#img_affineTransform(l,img_from_dir+'letters_upper/',img_to_dir+'letters_upper/')
for l in letters_up:
    img_affineTransform(l,img_from_dir+'letters_upper/',img_to_dir+'letters_upper/')
for l in letters_low:
    img_affineTransform(l, img_from_dir + 'letters_lower/', img_to_dir + 'letters_lower/')
for l in number:
    img_affineTransform(l, img_from_dir + 'number/', img_to_dir + 'number/')


# 3 透视变换
#img_shijueTransforimages[i]m(img)

#for l in letters_up:
#    img_shijueTransform(l,img_from_dir+'letters_upper/',img_to_dir+'letters_upper/')

# 简单阈值
#simpleThreshold(img0)

# 自适应阈值
#adaptiveThreshold(img0)

# 高斯模糊
#blur(img)

# 添加 高斯白噪声，椒盐噪声
#addSimpleNoise(img)

for l in letters_up:
    addGussiaNoise(l,img_from_dir+'letters_upper/',img_to_dir+'letters_upper/')
    addSaltNoise(l,img_from_dir+'letters_upper/',img_to_dir+'letters_upper/')
for l in letters_low:
    addGussiaNoise(l,img_from_dir+'letters_lower/',img_to_dir+'letters_lower/')
    addSaltNoise(l,img_from_dir+'letters_lower/',img_to_dir+'letters_lower/')
for l in number:
    addGussiaNoise(l,img_from_dir+'number/',img_to_dir+'number/')
    addSaltNoise(l,img_from_dir+'number/',img_to_dir+'number/')

# 图片叠加，随机的50张







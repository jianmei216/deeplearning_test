#-*- encoding:utf-8 -*-
import cv2
from numpy import *
import os

# public define
py_dir = os.path.dirname(os.path.realpath(__file__))
project_dir = os.path.dirname(py_dir)
img_from_dir = project_dir+'/files/'
img_to_dir = project_dir+'/filesoutput/letters_lower/'
img_from_file =img_from_dir+ 'a.png'



def SaltAndPepper(src, percetage):
    NoiseImg = src
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):
        randX = random.random_integers(0, src.shape[0] - 1)
        randY = random.random_integers(0, src.shape[1] - 1)
        if random.random_integers(0, 1) == 0:
            NoiseImg[randX, randY] = 0
        else:
            NoiseImg[randX, randY] = 20
    return NoiseImg


if __name__ == '__main__':
    img = cv2.imread(img_from_file, flags=0)
    gimg = cv2.GaussianBlur(img, (7, 7), sigmaX=0)  # 高斯平滑
    #NoiseImg = SaltAndPepper(gimg, 0.4)    # 椒盐噪声
    cv2.imshow('img',gimg)
    # figure()
    Pers = [0.2]
    for i in Pers:
        NoiseImg = SaltAndPepper(gimg, i)
        fileName = 'GaussianSaltPepper' + str(i) + '.jpg'
        cv2.imwrite(fileName, NoiseImg, [cv2.IMWRITE_JPEG_QUALITY, 100])
    #cv2.imshow('img2', NoiseImg)
    cv2.waitKey()
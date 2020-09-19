# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 09:53:59 2020

@author: yy
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


"""0. 加载图片"""
file_path = r'C:\Users\yy\Desktop\BI\L4\L4-2\sample_pic.jpg'
image = Image.open(file_path) 
img = np.array(image)


"""1. 打印图片"""
title_lists = ['original image', 'R channel', 'G channel', 'B channel']
plt.figure(figsize=(10.8, 12))

for i in range(4):
    plt.subplot(2,2,i+1)
    if i == 0:
        plt.imshow(img)      #绘制原图图片
    else:
        plt.imshow(img[:,:,i-1], plt.cm.gray)      #绘制第i个色道
    plt.title(title_lists[i])
    plt.axis('off')
plt.show()

"""2. 使用SVD提取图像前k维特征"""
from scipy.linalg import svd
def get_image_feature(img_array, k):
    p,s,q = svd(img_array, full_matrices=False)
    s_temp = np.zeros(s.shape[0])
    s_temp[0:k] = s[0:k]
    s = s_temp * np.identity(s.shape[0])
    temp = p.dot(s).dot(q)
    return temp

"""3. 对原图进行灰度化，进行特征提取并对比效果"""
from skimage import color
gray_img = color.rgb2gray(img)

title_lists = ['original gray', '1% features', '10% features', '15 features']
n = [1, 0.01, 0.1, 0.15]
plt.figure(figsize=(10.8, 12))

for i in range(4):
    plt.subplot(2,2,i+1)
    if i == 0:
        plt.imshow(gray_img, plt.cm.gray)      #绘制原图灰图
    else:
        k = round(min(gray_img.shape[0], gray_img.shape[1]) * n[i])
        temp_img = get_image_feature(gray_img, k)
        plt.imshow(temp_img, plt.cm.gray)
    plt.title(title_lists[i])
    plt.axis('off')
plt.show()

"""4. 对原图(彩图)进行特征提取并对比效果"""
full_img = img.reshape(img.shape[0], img.shape[1] * img.shape[2])

title_lists = ['original image', '1% features', '10% features', '15 features']
n = [1, 0.01, 0.1, 0.15]
plt.figure(figsize=(10.8, 12))

for i in range(4):
    plt.subplot(2,2,i+1)
    if i == 0:
        plt.imshow(img)      #绘制彩色原图
    else:
        k = round(min(full_img.shape[0], full_img.shape[1]) * n[i])
        temp_img = get_image_feature(full_img, k)
        temp_img = temp_img.reshape(img.shape[0], img.shape[1], img.shape[2])
        plt.imshow(temp_img.astype(np.uint8))
    plt.title(title_lists[i])
    plt.axis('off')
plt.show()

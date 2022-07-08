# -*- codeing = utf-8 -*-
# @Time : 2022/4/30 8:17
# @Author : Li Wenhai
# @Software : PyCharm
import cv2
import pandas as pd
import numpy as np

# 图像的读取保存npy文件
X = []  # 图片灰度值
Y = []  # 分类标签
data = pd.read_csv('label.txt') #读取label.txt内容
name = data['name'] #获取name列 文件名 img_x.jpg
a = data['label']   #获取label列 0-39 对应json的类别
# print(name)
# print(a)

#对40类垃圾进行划分为4类 通过索引可以进行归类
# 0:'其他垃圾',1:'厨余垃圾',2:'可回收物',3:'有害垃圾'
for i in range(len(data)):
    if a[i]<6:
        Y.append(0)
    elif 5<a[i]<14:
        Y.append(1)
    elif 13<a[i]<37:
        Y.append(2)
    elif 36<a[i]<40:
        Y.append(3)
    else:
        break
for i in range(len(data)):
      path = 'train_data/' + name[i] #每一张图片的路径
      Images = cv2.imread(path) #读取路径
      # print(path)
      image = cv2.resize(Images, (256,256), interpolation=cv2.INTER_LINEAR)
      #修改图像大小，采用双线性差值
      X.append((image))
np.save('./npy/x.npy', X) #x是图像
np.save('./npy/y.npy', Y) #y是标签
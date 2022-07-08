# -*- codeing = utf-8 -*-
# @Time : 2022/4/30 8:01
# @Author : Li Wenhai
# @Software : PyCharm
import os
import pandas as pd
# 首先读取数据分析样本和标签格式,制作样本标签文件
filenames = os.listdir(r'..\ResNet50\label') #label文件夹路径
filenames.sort(key=lambda x:int(x.split('img_')[1].split('.txt')[0])) #文件排序
# print(filenames)
label = []
text = []
label = pd.DataFrame(label) #此前说明过，类似电子表格的数据结构，包含一个经过排序的列表集，默认生成整数索引
for i in range(len(filenames)):#遍历文件夹
    path = 'label/' + filenames[i] #路径
    df1 = pd.read_csv(path,header=None) #读取
    label = label.append(df1) #添加
label.columns = ['name','label']

label.to_csv('label.txt') #保存label文件
print(label)
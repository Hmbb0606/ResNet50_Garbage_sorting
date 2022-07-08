from tensorflow.keras.models import *
import pandas as pd
import cv2
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行注释掉就是使用gpu，不注释就是使用cpu

def model():#加载模型
    model = load_model('../model.h5')
    return model
def read(path):#读取图片转化格式
    img = cv2.imread(path) #读取路径
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR) #修改图片大小，利用 双线性插值法
    img = img.reshape(1, 256, 256, 3) #修改图片维度，以适应模型去预测
    return img
def pre(model,img):#预测图片分类结果
    pred = model.predict(img) #预测结果
    y = np.argmax(pred, axis=-1) #返回最大值的索引
    labels= {0:'其他垃圾',1:'厨余垃圾',2:'可回收物',3:'有害垃圾'} #标签
    y = pd.DataFrame(y) #类似电子表格的数据结构，包含一个经过排序的列表集
    # print(y)
    y[0]=y[0].map(labels)
    # print(y[0])
    y = y.values.flatten() #将y展平变成一维
    print('该垃圾为:',y[0])
    return y

if __name__ == '__main__':
    path = r'C:\2.jpg'
    img = read(path)
    model = model()
    a=pre(model,img)

# -*- codeing = utf-8 -*-
# @Time : 2022/5/5 10:12
# @Author : Li Wenhai
# @Software : PyCharm
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import Xception,ResNet50,MobileNetV2,VGG16
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tensorflow.keras import layers, optimizers, models ,activations
from tensorflow.keras.layers import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# 这一行注释掉就是使用GPU，不注释就是使用CPU
# 由于我的GPU不能使用，该项目默认使用CPU
'''
申明：本小组使用了keras内置ResNet50模块，便于训练
    内置库的运行速度要比自行搭建的训练速度快
    在其他文件也尝试搭建了ResNet50模型
    并且两个文件均有相关注释
'''

def data_split():
    #标签独热码的转化和数据集的划分
    global x_train, x_test, y_train, y_test
    X = np.load('./npy/x.npy')
    Y = np.load('./npy/y.npy')
    #转化成npy文件后运行时，大幅加快读取速度
    Y = to_categorical(Y, 4)
    #4分类任务
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=2)
    #生成对应训练集合验证集，不赘述

def ResNet_50():
    # 模型搭建和配置训练
    conv_base = ResNet50(weights=None, include_top=False, input_shape=(256, 256, 3))
    #实例化ResNet50模型，weight参数为None：不使用预权重（迁移学习），参数为imagenet使用预权重。
    #input_shape设置输入图片大小参数，include_top设置
    model = models.Sequential()
    #Sequential()方法是一个容器，来搭建神经网络
    model.add(conv_base)
    #添加conv_base层
    model.add(layers.Flatten())
    #添加flatten展平层，为全连接准备
    model.add(Dense(1024, activation='relu'))
    ## 添加一个全连接层，神经元1024个；该层使用的激活函数relu
    model.add(Dropout(0.5))
    # Dropout通过随机（概率0.5）断开神经网络之间的连接，减少每次训练时实际参与计算的模型的参数量，
    # 从而减少了模型的实际容量，来防止过拟合
    model.add(layers.Dense(4, activation='softmax'))
    #添加一个全连接层，神经元4个（四分类输出），使用softmax函数
    conv_base.trainable = False
    model.summary()
    #输出模型各层的参数状况
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.001), metrics=['acc'])
    #使用Adam优化器，学习率取0.0001，损失函数使用多类的对数损失，metrics标注网络评价指标
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=6)
    #monitor监控的数据接口，monitor的意思是可以忍受在多少个epoch内没有改进，防止因为前期抖动导致过早停止训练。
    model_checkpoint = ModelCheckpoint('model5.h5', monitor='val_acc', verbose=1, save_best_only=True)
    #保存文件名，monitor监控的数据接口，verbose是信息展示模式(checkpoint的保存信息，类似Epoch 00001: saving model to ...)
    #设置save_best_only为True时，监测值有改进时才会保存当前的模型
    history = model.fit(x_train, y_train, epochs=10, batch_size=200,validation_data=(x_test,y_test),callbacks=[early_stop,model_checkpoint])
    #训练10个epoch，最初批次取200

if __name__ == '__main__':
    data_split()
    ResNet_50()
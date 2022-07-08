# -*- codeing = utf-8 -*-
# @Time : 2022/4/29 20:12
# @Author : Li Wenhai
# @Software : PyCharm
import numpy as np
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
from tensorflow.keras import layers, optimizers, models ,activations
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
'''
申明：由于ResNet模型过于复杂，
    该自建模型代码源于网络开源代码，且略有改动
    在理解结构后，本小组加以注释说明
'''
def data_split():
    #标签独热码的转化和数据集的划分
    global x_train, x_test, y_train, y_test
    X = np.load('./npy/x.npy')
    Y = np.load('./npy/y.npy')
    #转化成npy文件后运行时，大幅加快读取速度
    Y = to_categorical(Y, 4)
    #4分类任务
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=1)
    #生成对应训练集合验证集，不赘述

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    # conv_block在残差边加入了卷积操作，可以对输入矩阵的形状进行调整，使得残差边和卷积可以进行相加
    # 例如传入的参数为：(x, 3, [64, 64, 256], stage=2, block=‘a’, strides=(1, 1))
    filters1, filters2, filters3 = filters
    #重点：64,64,256！！！实际上是每轮残差结构的三个卷积核个数
    #stage和block是用来命名确定该层位置的参数
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    #conv_name_base = 'res2a_branch'
    #bn_name_base = 'bn2a_branch'

    # 降维 该模块内的第一次卷积
    # 1*1*64卷积核
    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    # 该模块内的第二次卷积
    # 3*3*64卷积核
    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    # 升维 该模块内的第三次卷积操作
    # 1*1*256卷积核
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)
    #注意没有对其直接激活函数，而是与残差边相加后再通过激活函数输出

    # 残差边
    # 通过1*1卷积核升维
    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)


    #最终残差边的输出和卷积层的输出相加，经激活函数后输出
    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

def identity_block(input_tensor, kernel_size, filters, stage, block):
    #Identity Block的输出和输入必须是有相同的形状 与conv_block相比减少了残差边上的1*1卷积核操作
    #以[64,64,256]为例
    filters1, filters2, filters3 = filters
    #stage和block是用来命名确定该层位置的参数
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'


    # 降维 该模块内的第一次卷积
    # 1*1*64卷积核
    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)


    # 该模块内的第二次卷积
    # 3*3*64卷积核
    x = Conv2D(filters2, kernel_size,padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)


    # 升维 该模块内的第三次卷积操作
    # 1*1*256卷积核
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    #Identity Block模块残差边不进行任何操作，将第三次卷积后的结果与输入相加，然后经过激活函数即得到输出结果
    x = Activation('relu')(x) #激活函数
    return x

def ResNet50(input_shape=[256,256,3],classes=4):

    # ResNet50模型就是将很多个Conv Block和Identity Block进行堆叠

    #stage0
    img_input = Input(shape=input_shape)
    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)# 进行BN操作
    x = Activation('relu')(x)# 使用relu激活函数
    # conv2_x先为一层3*3的最大池化下采样，步长为2
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # stage1
    # (input_tensor, kernel_size, filters, stage, block, strides)
    x = conv_block(x, 3, [64, 64, 256], stage=1, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=1, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=1, block='c')

    # stage2
    x = conv_block(x, 3, [128, 128, 512], stage=2, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=2, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=2, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=2, block='d')

    # stage3
    x = conv_block(x, 3, [256, 256, 1024], stage=3, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=3, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=3, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=3, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=3, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=3, block='f')

    # stage4
    x = conv_block(x, 3, [512, 512, 2048], stage=4, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=4, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=4, block='c')

    # 平均池化代替全连接层
    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    # 进行预测
    x = Flatten()(x)
    #展平输入到全连接层
    # x = tf.keras.layers.Dropout(0.3)(x)
    #加入dropout防止过拟合
    x = Dense(classes, activation='softmax', name='fc1000')(x)
    #通过softmax分类
    model = Model(img_input, x, name='resnet50')

    return model

if __name__ == '__main__':
    data_split()
    model_ResNet50 = ResNet50()
    # model_ResNet50.summary()
    model = models.Sequential()
    # Sequential()方法是一个容器，来搭建神经网络
    model.add(model_ResNet50)
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.0001), metrics=['acc'])
    # 使用Adam优化器，学习率取0.0001，损失函数使用多类的对数损失，metrics标注网络评价指标
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=6)
    # monitor监控的数据接口，monitor的意思是可以忍受在多少个epoch内没有改进，防止因为前期抖动导致过早停止训练。
    model_checkpoint = ModelCheckpoint('model6.h5', monitor='val_acc', verbose=1, save_best_only=True)
    # 保存文件名，monitor监控的数据接口，verbose是信息展示模式(checkpoint的保存信息，类似Epoch 00001: saving model to ...)
    # 设置save_best_only为True时，监测值有改进时才会保存当前的模型
    history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test),
                        callbacks=[early_stop, model_checkpoint])
    # 训练10个epoch，最初批次取32




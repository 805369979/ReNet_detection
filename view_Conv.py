import cv2
import keras
from keras.utils.np_utils import to_categorical
from sklearn.metrics import f1_score, recall_score
from tensorflow.keras.layers import Conv2D, MaxPooling2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, PReLU, Input, \
    BatchNormalization, GlobalMaxPooling2D, SeparableConv2D, LeakyReLU, Concatenate,Lambda,Flatten,SeparableConvolution2D
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers, layers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.layers import Activation, Dropout, Flatten, AveragePooling2D, add
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import regularizers
import os
import numpy as np
import warnings
from keras import backend as K
import time
from keras.applications import vgg16,mobilenet_v3,densenet,nasnet,efficientnet
import tensorflow.keras.layers
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.applications.efficientnet import EfficientNetB7
import sklearn
import seaborn as sns
from tensorflow.python.keras.layers import concatenate, Conv2DTranspose, ZeroPadding2D, Convolution2D

import loss
import cbam
import numpy as np
import uuid
# 用于避免卷积层同名报错

unique_random_number = uuid.uuid4()

class FerModel(object):
    def __init__(self):
        self.x_shape = (224, 224,3)
        self.epoch = 100
        self.batchsize = 16
        self.weight_decay = 0.0005
        self.classes = 10
        self.model = self.build_model()

    @staticmethod
    def get_call_backs():
        call_backs = [
            # ModelCheckpoint('./logs/' + 'best.h5',
            #                 save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2),
            TensorBoard('./logs/balanced_data_model'),
            # EarlyStopping(monitor='val_loss', patience=40)
        ]

        print("调用回调函数!!!")

        return call_backs
    def channel_attention(self,inputs):
        # 定义可训练变量，反向传播可更新
        gama = tf.Variable(tf.ones(1))  # 初始化1

        # 获取输入特征图的shape
        b, h, w, c = inputs.shape

        # 重新排序维度[b,h,w,c]==>[b,c,h,w]
        x = tf.transpose(inputs, perm=[0, 3, 1, 2])  # perm代表重新排序的轴
        # 重塑特征图尺寸[b,c,h,w]==>[b,c,h*w]
        x_reshape = tf.reshape(x, shape=[-1, c, h * w])

        # 重新排序维度[b,c,h*w]==>[b,h*w,c]
        x_reshape_trans = tf.transpose(x_reshape, perm=[0, 2, 1])  # 指定需要交换的轴
        # 矩阵相乘
        x_mutmul = x_reshape_trans @ x_reshape
        # 经过softmax归一化权重
        x_mutmul = tf.nn.softmax(x_mutmul)

        # reshape后的特征图与归一化权重矩阵相乘[b,x,h*w]
        x = x_reshape @ x_mutmul
        # 重塑形状[b,c,h*w]==>[b,c,h,w]
        x = tf.reshape(x, shape=[-1, c, h, w])
        # 重新排序维度[b,c,h,w]==>[b,h,w,c]
        x = tf.transpose(x, perm=[0, 2, 3, 1])
        # 结果乘以可训练变量
        x = x * gama

        # 输入和输出特征图叠加
        x = add([x, inputs])

        return x

    # （2）位置注意力
    def position_attention(self,inputs):
        # 定义可训练变量，反向传播可更新
        gama = tf.Variable(tf.ones(1))  # 初始化1

        # 获取输入特征图的shape
        b, h, w, c = inputs.shape

        # 深度可分离卷积[b,h,w,c]==>[b,h,w,c//8]
        x1 = SeparableConv2D(filters=c // 8, kernel_size=(1, 1), strides=1, padding='same')(inputs)
        # 调整维度排序[b,h,w,c//8]==>[b,c//8,h,w]
        x1_trans = tf.transpose(x1, perm=[0, 3, 1, 2])
        # 重塑特征图尺寸[b,c//8,h,w]==>[b,c//8,h*w]
        x1_trans_reshape = tf.reshape(x1_trans, shape=[-1, c // 8, h * w])
        # 调整维度排序[b,c//8,h*w]==>[b,h*w,c//8]
        x1_trans_reshape_trans = tf.transpose(x1_trans_reshape, perm=[0, 2, 1])
        # 矩阵相乘
        x1_mutmul = x1_trans_reshape_trans @ x1_trans_reshape
        # 经过softmax归一化权重
        x1_mutmul = tf.nn.softmax(x1_mutmul)

        # 深度可分离卷积[b,h,w,c]==>[b,h,w,c]
        x2 = SeparableConv2D(filters=c, kernel_size=(1, 1), strides=1, padding='same')(inputs)
        # 调整维度排序[b,h,w,c]==>[b,c,h,w]
        x2_trans = tf.transpose(x2, perm=[0, 3, 1, 2])
        # 重塑尺寸[b,c,h,w]==>[b,c,h*w]
        x2_trans_reshape = tf.reshape(x2_trans, shape=[-1, c, h * w])

        # 调整x1_mutmul的轴，和x2矩阵相乘
        x1_mutmul_trans = tf.transpose(x1_mutmul, perm=[0, 2, 1])
        x2_mutmul = x2_trans_reshape @ x1_mutmul_trans

        # 重塑尺寸[b,c,h*w]==>[b,c,h,w]
        x2_mutmul = tf.reshape(x2_mutmul, shape=[-1, c, h, w])
        # 轴变换[b,c,h,w]==>[b,h,w,c]
        x2_mutmul = tf.transpose(x2_mutmul, perm=[0, 2, 3, 1])
        # 结果乘以可训练变量
        x2_mutmul = x2_mutmul * gama

        # 输入和输出叠加
        x = add([x2_mutmul, inputs])
        return x

    # （3）DANet网络架构
    def danet(self,inputs):
        # 输入分为两个分支
        x1 = self.channel_attention(inputs)  # 通道注意力
        x2 = self.position_attention(inputs)  # 位置注意力

        # 叠加两个注意力的结果
        x = add([x1, x2])
        return x
    def conv_bn(self, x, nb_filters, kernel_size, padding="same", strides=(1, 1), name=None):
        if name is not None:
            bn_name = name + "_bn"
            conv_name = name + "_conv"
        else:
            bn_name = None
            conv_name = None
        x = Conv2D(nb_filters, kernel_size, padding=padding, strides=strides, name=conv_name)(x)

        x = BatchNormalization(axis=-1, name=bn_name)(x)
        x = Activation('relu')(x)

        return x

    def bottle_block(self, input, nb_filters, padding="same", strides=(1, 1), with_conv_shortcut=False):
        k1, k2, k3 = nb_filters
        x = self.conv_bn(input, k1, (1, 1), padding=padding, strides=strides)
        x = self.conv_bn(x, k2, (3,3), padding=padding)
        x = self.conv_bn(x, k3, (1, 1), padding=padding)
        # x = cbam.cbam_module(x)
        if with_conv_shortcut:
            shortcut = self.conv_bn(input, k3, (1, 1), padding=padding, strides=strides)
            x = add([x, shortcut])
        else:
            x = add([x, input])
        return x

    def build_model(self):
        input = Input(shape=(448,448,3))
        x = self.bottle_block(input, (32, 32, 64), strides=(1, 1), with_conv_shortcut=True)
        x = self.bottle_block(x, (64, 64, 128), strides=(1, 1), with_conv_shortcut=True)
        x = self.bottle_block(x, (128, 128, 256), strides=(1, 1), with_conv_shortcut=True)
        x = self.bottle_block(x, (256, 256, 512), strides=(1, 1), with_conv_shortcut=True)
        model = Model(inputs=input, outputs=x)
        model.summary()
        return model


def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row,col

def visualize_feature_map(img_batch):
    feature_map = np.squeeze(img_batch,axis=0)
    print(feature_map.shape)

    feature_map_combination=[]
    plt.figure()

    num_pic = feature_map.shape[2]
    row,col = get_row_col(num_pic)

    for i in range(0,num_pic):
        feature_map_split=feature_map[:,:,i]
        # cv2.imwrite("feature_map_split.jpg",feature_map_split)
        feature_map_combination.append(feature_map_split)
        # plt.subplot(row,col,i+1)
        # plt.imshow(feature_map_split)
        # axis('off')
        # title('feature_map_{}'.format(i))

    # plt.savefig('feature_map.jpg')
    # plt.show()

    # 各个特征图按1：1 叠加
    feature_map_sum = sum(ele for ele in feature_map_combination)
    plt.imshow(feature_map_sum)
    plt.savefig("feature_map_sum.eps")
    plt.savefig("feature_map_sum.jpg")
def conv_bn(self, x, nb_filters, kernel_size, padding="same", strides=(1, 1), name=None):
    if name is not None:
        bn_name = name + "_bn"
        conv_name = name + "_conv"
    else:
        bn_name = None
        conv_name = None
    x = Conv2D(nb_filters, kernel_size, padding=padding, strides=strides, name=conv_name)(x)

    x = BatchNormalization(axis=-1, name=bn_name)(x)
    x = Activation('relu')(x)

    return x

def bottle_block(self, input, nb_filters, padding="same", strides=(1, 1), with_conv_shortcut=False):
    k1, k2, k3 = nb_filters
    x = self.conv_bn(input, k1, (1, 1), padding=padding, strides=strides)
    x = self.conv_bn(x, k2, (3, 3), padding=padding)
    x = self.conv_bn(x, k3, (1, 1), padding=padding)
    x = cbam.cbam_module(x)
    if with_conv_shortcut:
        shortcut = self.conv_bn(input, k3, (1, 1), padding=padding, strides=strides)
        x = add([x, shortcut])
    else:
        x = add([x, input])
    return x
def create_model():
    model = Sequential()

    # 第一层CNN
    # 第一个参数是卷积核的数量，第二三个参数是卷积核的大小
    model.add(Convolution2D(64, 3, strides=(1, 1),padding="same", input_shape=img.shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 第二层CNN
    model.add(Convolution2D(128, 3, strides=(1, 1),padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # # # # # # # #
    # # # # # # 第三层CNN
    model.add(Convolution2D(256, 3, strides=(1, 1),padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # # # # # # # #
    # # # # # # # # 第四层CNN
    model.add(Convolution2D(512, 3, strides=(1,1),padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))


    # # model.add(Convolution2D(256, 3, 3, input_shape=img.shape))
    # # model.add(Activation('relu'))
    # # model.add(MaxPooling2D(pool_size=(2, 2)))

    return model


if __name__ == "__main__":

    img = cv2.imread('person_xiushi.png')
    model = FerModel().model
    img_batch = np.expand_dims(img, axis=0)
    print(img_batch)
    conv_img = model.predict(img_batch)  # conv_img 卷积结果
    visualize_feature_map(conv_img)
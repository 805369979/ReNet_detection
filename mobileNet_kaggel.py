
import keras
import keras_cv_attention_models.fastervit
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
from keras_cv_attention_models import mobilevit,repvit,efficientformer
import numpy as np
import warnings
from keras import backend as K
import time
from keras.applications import vgg16,mobilenet_v3,densenet,nasnet,efficientnet
import tensorflow.keras.layers
from tensorflow.python.keras.applications.efficientnet import EfficientNetB7
import sklearn
import seaborn as sns
from tensorflow.python.keras.applications.mobilenet import MobileNet
from tensorflow.python.keras.layers import concatenate, Conv2DTranspose, ZeroPadding2D, multiply, UpSampling2D, Add

import cbam
import numpy as np
import uuid
# 用于避免卷积层同名报错
unique_random_number = uuid.uuid4()

class FerModel(object):
    def __init__(self):
        self.x_shape = (224, 224,3)
        self.epoch = 200
        self.batchsize = 16
        self.weight_decay = 0.0005
        self.classes = 10
        self.model = self.build_model()
        self.call_backs = self.get_call_backs()
        start = time.time()
        self.history = self.train()
        end = time.time()
        print(f'训练共耗时{round(end - start, 2)}s')

        # self.show_history()

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
        x = SeparableConv2D(nb_filters, kernel_size, padding=padding, strides=strides, name=conv_name)(x)

        x = BatchNormalization(axis=-1, name=bn_name)(x)
        x = Activation('relu')(x)

        return x

    def bottle_block(self, input, nb_filters, padding="same", strides=(1, 1), with_conv_shortcut=False):
        k1, k2, k3 = nb_filters
        x1 = self.conv_bn(input, k1, (3, 3), padding=padding, strides=strides)
        x2 = self.conv_bn(x1, k2, (5, 5), padding=padding)
        x3 = self.conv_bn(x2, k3, (7, 7), padding=padding)
        x4 = cbam.cbam_module(x3)
        if with_conv_shortcut:
            shortcut = self.conv_bn(input, k3, (1, 1), padding=padding, strides=strides)
            x = add([x4, shortcut])
        else:
            x = add([x1, input])
        return x

    # def build_model(self):
    #     # 骨干网络（Backbone）
    #     base_model = tensorflow.keras.applications.mobilenet.MobileNet(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    #
    #
    #     # 提取特征层（根据MobileNet结构选择）
    #     c3 = base_model.get_layer('block_6_expand_relu').output  # 1/8尺度
    #     c4 = base_model.get_layer('block_13_expand_relu').output  # 1/16尺度
    #     c5 = base_model.output  # 1/32尺度
    #
    #     # FPN构建
    #     # 顶层处理路径（P5）
    #     p5 = Conv2D(256, 1, name='c5_reduced')(c5)
    #     p5_upsampled = UpSampling2D()(p5)
    #
    #     # 中间层处理路径（P4）
    #     c4_reduced = Conv2D(256, 1, name='c4_reduced')(c4)
    #     p4 = Add()([p5_upsampled, c4_reduced])
    #     p4 = Conv2D(256, 3, padding='same', activation='relu')(p4)
    #     p4_upsampled = UpSampling2D()(p4)
    #
    #     # 底层处理路径（P3）
    #     c3_reduced = Conv2D(256, 1, name='c3_reduced')(c3)
    #     p3 = Add()([p4_upsampled, c3_reduced])
    #     p3 = Conv2D(256, 3, padding='same', activation='relu')(p3)
    #
    #     # 附加金字塔层
    #     p6 = Conv2D(256, 3, strides=2, padding='same', name='p6')(c5)
    #     p7 = Conv2D(256, 3, strides=2, padding='same', activation='relu', name='p7')(p6)
    #
    #     # 分类头（示例）
    #     outputs = Dense(10, activation='softmax')(x)
    #
    #     return Model(inputs=base_model.input, outputs=outputs)

    def build_model(self):
        base_model = tensorflow.keras.applications.mobilenet.MobileNet(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        base = base_model.output

        # base_model = keras_cv_attention_models.fastervit.FasterViT(input_shape=(224, 224, 3), pretrained='imagenet')
        # outputs = [layer.output for layer in base_model1.layers if layer.name == 'features_swish']

        x = self.bottle_block(base, (64, 64, 64), strides=(1, 1), with_conv_shortcut=True)
        x = self.bottle_block(x, (128, 128, 128), strides=(1, 1), with_conv_shortcut=True)
        x = self.bottle_block(x, (256, 256, 256), strides=(1, 1), with_conv_shortcut=True)

        # x = self.bottle_block(base_model.layers[-3].output, (32, 32, 64), strides=(1, 1), with_conv_shortcut=True)
        # x = self.bottle_block(x, (64, 64, 128), strides=(1, 1), with_conv_shortcut=True)
        # x = self.bottle_block(x, (128, 128, 256), strides=(1, 1), with_conv_shortcut=True)

        # print(outputs)
        x = GlobalAveragePooling2D()(x)
        # outputs = GlobalAveragePooling2D()(base_model.layers[-3].output)
        base = GlobalAveragePooling2D()(base)

        x = tf.keras.layers.concatenate([base,x], name=str(uuid.uuid4()))
        x = BatchNormalization()(x)
        # x = Dropout(0.2)(x)
        predictions = Dense(10, activation="softmax", kernel_initializer='he_normal')(x)
        # model = Model(inputs=[base_model1.input], outputs=predictions)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.summary()
        return model
    def train(self):
        X_train = np.load('./driver_feature_kaggle_224/test/images.npy')
        X_train = X_train.astype(np.float16)
        X_train = X_train.reshape([-1, 224, 224, 3])
        X_train = X_train - np.mean(X_train, axis=0)

        np.random.seed(2025)
        np.random.shuffle(X_train)

        y_train = np.load('./driver_feature_kaggle_224/test/labels.npy')
        np.random.seed(2025)
        np.random.shuffle(y_train)

        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=2025)
        print(X_train.shape)
        print(y_train.shape)
        print(X_valid.shape)
        print(y_valid.shape)

        # 创建Momentum优化器
        momentum_optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.95)
        self.model.compile(optimizer=momentum_optimizer,
                           loss='categorical_crossentropy',  # 损失函数
                           metrics=['accuracy'])  # 指标
        # training the model
        history = self.model.fit(X_train, y_train,
                                 batch_size=self.batchsize,
                                 epochs=self.epoch,
                                 verbose=1,
                                 # shuffle=True,
                                 validation_data=(X_valid, y_valid),
                                 callbacks=[self.call_backs]
                                 )

        yPred = []
        y = []
        for i in range(len(X_valid)):
            image = np.expand_dims(X_valid[i], axis=0)
            predictions = self.model.predict(image)
            top_class_index = tf.argmax(predictions, axis=-1)
            # top_class_probability = predictions[0][top_class_index]
            yPred.append([int(top_class_index.numpy())])
            x_real = [k for k, v in enumerate(y_valid[i]) if v == 1]
            y.append(x_real)
        draw_confu(y, yPred, name='test')

        self.model.save('my_model.h5')

def draw_confu(y, y_pred, name=''):
    sns.set(font_scale=3)
    confusion_matrix = sklearn.metrics.confusion_matrix(y, y_pred)
    plt.xticks(fontsize=10)  # 设置x轴刻度字体大小为12
    plt.yticks(fontsize=10)
    plt.figure(figsize=(16, 14))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 20});
    plt.title("Confusion matrix", fontsize=32)
    plt.ylabel('Actual Label', fontsize=28)
    plt.xlabel('Predicted Label', fontsize=28)
    plt.savefig('./result_%s.eps' % (name))
    plt.savefig('./result_%s.svg' % (name))
    plt.savefig('./result_%s.jpg' % (name))

def evaluate(model, X, Y):
    accuracy = model.evaluate(X, Y)
    return accuracy[0]

def get_flops(model, model_inputs) -> float:
    """
    Calculate FLOPS [GFLOPs] for a tf.keras.Model or tf.keras.Sequential model
    in inference mode. It uses tf.compat.v1.profiler under the hood.
    """
    # if not hasattr(model, "model"):
    #     raise wandb.Error("self.model must be set before using this method.")

    if not isinstance(
            model, (tf.keras.models.Sequential, tf.keras.models.Model)
    ):
        raise ValueError(
            "Calculating FLOPS is only supported for "
            "`tf.keras.Model` and `tf.keras.Sequential` instances."
        )

    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2_as_graph,
    )

    # Compute FLOPs for one sample
    batch_size = 1
    inputs = [
        tf.TensorSpec([batch_size] + inp.shape[1:], inp.dtype)
        for inp in model_inputs
    ]

    # convert tf.keras model into frozen graph to count FLOPs about operations used at inference
    real_model = tf.function(model).get_concrete_function(inputs)
    frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)

    # Calculate FLOPs with tf.profiler
    run_meta = tf.compat.v1.RunMetadata()
    opts = (
        tf.compat.v1.profiler.ProfileOptionBuilder(
            tf.compat.v1.profiler.ProfileOptionBuilder().float_operation()
        )
            .with_empty_output()
            .build()
    )

    flops = tf.compat.v1.profiler.profile(
        graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
    )

    tf.compat.v1.reset_default_graph()

    # convert to GFLOPs
    return (flops.total_float_ops / 1e9) / 2

if __name__ == '__main__':
    import tensorflow as tf
    from keras import backend as K
    K.clear_session()
    import random

    random.seed(2025)
    import numpy as np
    np.random.seed(2025)
    tf.random.set_seed(2025)
    import os

    os.environ['PYTHONHASHSEED'] = '2025'

    # 不使用gpu则开启这一行代码
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print(tf.test.is_gpu_available())
    fer_model = FerModel()
    print(tf.test.is_gpu_available())
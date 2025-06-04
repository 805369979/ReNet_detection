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
from keras.applications import vgg16,mobilenet_v3,densenet,nasnet
import tensorflow.keras.layers
from tensorflow.python.keras.applications.densenet import DenseNet201
import cbam
import numpy as np
import uuid
# 用于避免卷积层同名报错
unique_random_number = uuid.uuid4()
import sklearn
import seaborn as sns
class FerModel(object):
    def __init__(self):
        self.x_shape = (224, 224,3)
        self.epoch = 100
        self.batchsize = 32
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

    def channel_shuffle(self, x, groups=1):
        height, width, in_channels = x.shape.as_list()[1:]
        channels_per_group = in_channels // groups
        x = K.reshape(x, [-1, height, width, groups, channels_per_group])
        x = K.permute_dimensions(x, (0, 1, 2, 4, 3))  # transpose
        x = K.reshape(x, [-1, height, width, in_channels])
        print(K.print_tensor(x, 'shuffled'))
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
        x = self.conv_bn(input, k1, (3, 3), padding=padding, strides=strides)
        x = self.conv_bn(x, k2, (5, 5), padding=padding)
        x = self.conv_bn(x, k3, (7, 7), padding=padding)
        x = cbam.cbam_module(x)
        if with_conv_shortcut:
            shortcut = self.conv_bn(input, k3, (1, 1), padding=padding, strides=strides)
            x = add([x, shortcut])
        else:
            x = add([x, input])
        return x

    def build_model(self):
        base_model = tensorflow.keras.applications.mobilenet.MobileNet(include_top=False, weights='imagenet',
                                                                       input_shape=(224, 224, 3))
        base = base_model.output

        x = self.bottle_block(base, (64, 64, 64), strides=(1, 1), with_conv_shortcut=True)
        x = self.bottle_block(x, (128, 128, 128), strides=(1, 1), with_conv_shortcut=True)
        x = self.bottle_block(x, (256, 256, 256), strides=(1, 1), with_conv_shortcut=True)
        # x = self.bottle_block(x, (256, 256, 512), strides=(1, 1), with_conv_shortcut=True)
        #
        x = GlobalAveragePooling2D()(x)
        base = GlobalAveragePooling2D()(base)

        x = tf.keras.layers.concatenate([base, x], name=str(uuid.uuid4()))
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        predictions = Dense(10, activation="softmax", kernel_initializer='he_normal')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.summary()
        return model
    def train(self):
        import numpy as np
        x = tf.constant(np.random.randn(1, 224, 224, 3))
        start = time.time()
        for i in range(500):
            features = self.model.predict(x)
        end = time.time()-start
        print(end/i)
        x = tf.constant(np.random.randn(1, 224, 224, 3))
        print(get_flops(self.model, [x]))



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

    # 测试自己的模型时间
    import tensorflow as tf
    # 使用gpu则开启这一行代码
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # 使用cpu则开启这一行代码
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print(tf.test.is_gpu_available())
    fer_model = FerModel()
    print(tf.test.is_gpu_available())

    # # 测试模型时间
    # import tensorflow as tf
    # # 使用gpu则开启这一行代码
    # # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # # 使用cpu则开启这一行代码
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # print(tf.test.is_gpu_available())
    # import numpy as np
    # base_model = tensorflow.keras.applications.efficientnet.EfficientNetB7(input_shape=(224,224,3),weights=None,include_top=False)
    # # base_model = DenseNet201(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    # # base_model = tensorflow.keras.applications.resnet.ResNet152(include_top=False, weights='imagenet',input_shape=(224, 224, 3))
    # # base_model = tensorflow.keras.applications.resnet.ResNet50(include_top=False, weights='imagenet',input_shape=(224, 224, 3))
    # # base_model = tensorflow.keras.applications.mobilenet.MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # # base_model = tensorflow.keras.applications.NASNetMobile(weights='imagenet', include_top=False,input_shape=(224, 224, 3))
    # # base_model = tensorflow.keras.applications.Xception(weights='imagenet', include_top=False,input_shape=(224, 224, 3))
    # x = base_model.output
    # x = GlobalAveragePooling2D()(x)
    # predictions = Dense(10, activation='softmax')(x)
    # # print(base_model.summary())
    # model = Model(inputs=base_model.input, outputs=predictions)
    # x = tf.constant(np.random.randn(1, 224, 224, 3))
    # start = time.time()
    # # cpu 50 Gpu 500
    # for i in range(18):
    #     features = model.predict(x)
    # end = time.time() - start
    # print(end / i)
    # x = tf.constant(np.random.randn(1, 224, 224, 3))
    # print(get_flops(model, [x]))
    #
    #

import cv2
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
from tensorflow.python.keras.layers import concatenate, Conv2DTranspose, ZeroPadding2D, multiply, UpSampling2D, Add, \
    SyncBatchNormalization
import cbam
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

        x = SyncBatchNormalization(name=bn_name)(x)
        x = Activation('relu')(x)

        return x

    def bottle_block(self, input, nb_filters,name, padding="same", strides=(1, 1), with_conv_shortcut=True):
        k1, k2, k3 = nb_filters
        x1 = self.conv_bn(input, k1, (3, 3), padding=padding, strides=strides)
        x2 = self.conv_bn(x1, k2, (5, 5), padding=padding)
        x3 = self.conv_bn(x2, k3, (7, 7), padding=padding,name=name)
        # x4 = cbam.cbam_module(x3)
        if with_conv_shortcut:
            # shortcut1 = self.conv_bn(x1, k3, (1, 1), padding=padding, strides=strides)
            shortcut1 = self.conv_bn(input, k3, (1, 1), padding=padding, strides=strides)
            # shortcut2 = self.conv_bn(x2, k3, (1, 1), padding=padding, strides=strides)
            # shortcut3 = self.conv_bn(x3, k3, (1, 1), padding=padding, strides=strides)
            x = add([x3, shortcut1])
        else:
            x = add([x1, input])
        return x
    def bottle_block1(self, input, nb_filters,name, padding="same", strides=(1, 1), with_conv_shortcut=True):
        k1, k2, k3 = nb_filters
        x1 = self.conv_bn(input, k1, (3, 3), padding=padding, strides=strides)
        x2 = self.conv_bn(x1, k2, (5, 5), padding=padding)
        x3 = self.conv_bn(x2, k3, (7, 7), padding=padding,name=name)
        # x4 = cbam.cbam_module(x3)
        if with_conv_shortcut:
            shortcut = self.conv_bn(input, k3, (1, 1), padding=padding, strides=strides)
            x = add([x3, shortcut])
        else:
            x = add([x1, input])
        return x


    def build_model(self):
        base_model = tensorflow.keras.applications.efficientnet.EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        # base_model = tensorflow.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        base = base_model.output



        # base_model = keras_cv_attention_models.fastervit.FasterViT(input_shape=(224, 224, 3), pretrained='imagenet')
        # outputs = [layer.output for layer in base_model1.layers if layer.name == 'features_swish']

        x = self.bottle_block(base, (64, 64, 64), strides=(1, 1), name="1", with_conv_shortcut=True)
        x = self.bottle_block(x, (128, 128, 128), strides=(1, 1),name="2", with_conv_shortcut=True)
        x = self.bottle_block(x, (256, 256, 256), strides=(1, 1),name="3", with_conv_shortcut=True)


        # x = self.bottle_block(base_model.layers[-3].output, (32, 32, 64), strides=(1, 1), with_conv_shortcut=True)
        # x = self.bottle_block(x, (64, 64, 128), strides=(1, 1), with_conv_shortcut=True)
        # x = self.bottle_block(x, (128, 128, 256), strides=(1, 1), with_conv_shortcut=True)

        x = GlobalAveragePooling2D()(x)
        # outputs = GlobalAveragePooling2D()(base_model.layers[-3].output)
        base = GlobalAveragePooling2D()(base)

        x = tf.keras.layers.concatenate([base, x], name=str(uuid.uuid4()))
        x = SyncBatchNormalization()(x)
        # x = Dropout(0.2)(x)
        predictions = Dense(10, activation="softmax", kernel_initializer='he_normal')(x)
        # model = Model(inputs=[base_model1.input], outputs=predictions)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.summary()
        return model

    def train(self):
        import numpy as np
        X_train = np.load('driver_feature_RGB_224/train/images.npy')
        X_train = X_train.reshape([-1, 224,224, 3])
        np.random.seed(2025)
        np.random.shuffle(X_train)

        y_train = np.load('driver_feature_RGB_224/train/labels.npy')
        np.random.seed(2025)
        np.random.shuffle(y_train)

        X_valid = np.load('driver_feature_RGB_224/test/images.npy')
        X_valid = X_valid.reshape([-1, 224, 224, 3])
        np.random.seed(2025)
        np.random.shuffle(X_valid)

        y_valid = np.load('driver_feature_RGB_224/test/labels.npy')
        np.random.seed(2025)
        np.random.shuffle(y_valid)

        print(X_train.shape)
        print(y_train.shape)
        print(X_valid.shape)
        print(y_valid.shape)

        # 创建Momentum优化器
        momentum_optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.95)
        self.model.compile(optimizer=momentum_optimizer,
                           loss='categorical_crossentropy', # 损失函数
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

        # yPred = []
        # y = []
        # for i in range(len(X_valid)):
        #     image = np.expand_dims(X_valid[i], axis=0)
        #     predictions = self.model.predict(image)
        #     top_class_index = tf.argmax(predictions, axis=-1)
        #     # top_class_probability = predictions[0][top_class_index]
        #     yPred.append([int(top_class_index.numpy())])
        #     x_real = [k for k, v in enumerate(y_valid[i]) if v == 1]
        #     y.append(x_real)
        # draw_confu(y, yPred, name='test')
        self.model.save_weights("adasd.h5")

        # 靠谱方式
        data_path_abs = 'C:\\Users\\Administrator\\Desktop\\DriverDetection\\auc'
        img_list_all = os.listdir(data_path_abs)
        for key, v in enumerate(img_list_all):
            input_img1 = cv2.imread(data_path_abs + "/" + v)
            input_img1 = cv2.resize(input_img1, (224, 224))
            input_img1 = np.expand_dims(input_img1, axis=0)
            # sobel_image = sobel_image - np.mean(sobel_image, axis=0)
            predictions = self.model.predict(input_img1)
            #     # 获取最后一层卷积层的输出
            last_conv_layer = self.model.get_layer('conv_pw_13_relu')
            grad_model = Model([self.model.inputs], [last_conv_layer.output, self.model.output])

            #     # 计算类别的梯度
            with tf.GradientTape() as tape:
                conv_layer_output, preds = grad_model(input_img1)
                class_channel = preds[0][np.argmax(preds[0])]
            # 计算梯度
            grads = tape.gradient(class_channel, conv_layer_output)
            # 计算权重
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            # pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
            # heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_layer_output), axis=-1)
            heatmap = conv_layer_output @ pooled_grads[..., tf.newaxis]

            # feature_map_sum = sum(ele for ele in feature_map_combination)
            heatmap = tf.maximum(heatmap, 0)
            heatmap /= tf.reduce_max(heatmap)

            # 重塑热力图并将其缩放到与原始图像相同的大小
            heatmap = np.squeeze(heatmap)
            #     heatmap = cv2.resize(heatmap, (224, 224))
            #     heatmap = np.uint8(255 * heatmap)
            #     heatmap = np.clip(heatmap, 0, 1)
            # 颜色映射
            gbkInput = cv2.imread(data_path_abs + "/" + v)
            gbkInput = cv2.resize(gbkInput, (224, 224))

            feature_map_sum1 = cv2.resize(heatmap, (224, 224))
            # 将热力图转换为RGB格式
            feature_map_sum = np.uint8(255 * feature_map_sum1)
            feature_map_sum[feature_map_sum < 80] = 0
            # 将热利用应用于原始图像
            feature_map_sum = cv2.applyColorMap(feature_map_sum, cv2.COLORMAP_JET)
            # 　这里的热力图因子是０.４
            superimposed_img = feature_map_sum * 0.4 + gbkInput
            cv2.imwrite("my2{}".format(v), superimposed_img)


            #     # 获取最后一层卷积层的输出
            last_conv_layer = self.model.get_layer('conv_pw_13_relu')
            grad_model = Model([self.model.inputs], [last_conv_layer.output, self.model.output])

            #     # 计算类别的梯度
            with tf.GradientTape() as tape:
                conv_layer_output, preds = grad_model(input_img1)
                class_channel = preds[0][np.argmax(preds[0])]
            # 计算梯度
            grads = tape.gradient(class_channel, conv_layer_output)
            # 计算权重
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            # pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
            # heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_layer_output), axis=-1)
            heatmap = conv_layer_output @ pooled_grads[..., tf.newaxis]

            # feature_map_sum = sum(ele for ele in feature_map_combination)
            heatmap = tf.maximum(heatmap, 0)
            heatmap /= tf.reduce_max(heatmap)

            # 重塑热力图并将其缩放到与原始图像相同的大小
            heatmap = np.squeeze(heatmap)
            #     heatmap = cv2.resize(heatmap, (224, 224))
            #     heatmap = np.uint8(255 * heatmap)
            #     heatmap = np.clip(heatmap, 0, 1)
            # 颜色映射
            gbkInput = cv2.imread(data_path_abs + "/" + v)
            gbkInput = cv2.resize(gbkInput, (224, 224))

            feature_map_sum1 = cv2.resize(heatmap, (224, 224))
            # 将热力图转换为RGB格式
            feature_map_sum = np.uint8(255 * feature_map_sum1)
            feature_map_sum[feature_map_sum < 80] = 0
            # 将热利用应用于原始图像
            feature_map_sum = cv2.applyColorMap(feature_map_sum, cv2.COLORMAP_JET)
            # 　这里的热力图因子是０.４
            superimposed_img = feature_map_sum * 0.4 + gbkInput
            cv2.imwrite("my3{}".format(v), superimposed_img)

        data_path_abs = 'C:\\Users\\Administrator\\Desktop\\DriverDetection\\auc'
        img_list_all = os.listdir(data_path_abs)
        for key, v in enumerate(img_list_all):
            input_img1 = cv2.imread(data_path_abs + "/" + v)
            input_img1 = cv2.resize(input_img1, (224, 224))

            sobel_image = np.expand_dims(input_img1, axis=0)
            # sobel_image = sobel_image - np.mean(sobel_image, axis=0)
            heatmap = make_gradcam_heatmap(sobel_image, self.model, "conv_pw_13_relu","guofang1")
            save_and_display_gradcam(data_path_abs + "/" + v, heatmap,v,"guanfang1")



        data_path_abs = 'C:\\Users\\Administrator\\Desktop\\DriverDetection\\auc'
        img_list_all = os.listdir(data_path_abs)
        for key, v in enumerate(img_list_all):
            input_img1 = cv2.imread(data_path_abs + "/" + v)
            input_img1 = cv2.resize(input_img1, (224, 224))

            sobel_image = np.expand_dims(input_img1, axis=0)
            # sobel_image = sobel_image - np.mean(sobel_image, axis=0)
            heatmap = make_gradcam_heatmap(sobel_image, self.model, "conv_pw_13_relu","guanfang2")
            save_and_display_gradcam(data_path_abs + "/" + v, heatmap, v,"guanfang2")

        data_path_abs = 'C:\\Users\\Administrator\\Desktop\\DriverDetection\\auc'
        img_list_all = os.listdir(data_path_abs)
        for key, v in enumerate(img_list_all):
            input_img1 = cv2.imread(data_path_abs + "/" + v)
            input_img1 = cv2.resize(input_img1, (224, 224))

            sobel_image = np.expand_dims(input_img1, axis=0)
            sobel_image = sobel_image - np.mean(sobel_image, axis=0)
            predictions = self.model.predict(sobel_image)
            # cv2.imshow('Segmentation', sobel_image)
            # cv2.waitKey(1500)
            # cv2.destroyAllWindows()

            #     # 获取最后一层卷积层的输出
            last_conv_layer = self.model.get_layer('conv_pw_13_relu')
            last_conv_layer_output = last_conv_layer.output
            model_with_last_conv = Model(inputs=self.model.input, outputs=last_conv_layer_output)
            feature_map = model_with_last_conv.predict(sobel_image)
            aa = visualize_feature_map(feature_map)

            input_img1 = cv2.imread(data_path_abs + "/" + v)
            input_img1 = cv2.resize(input_img1, (224, 224))

            feature_map_sum1 = cv2.resize(aa, (224, 224))
            # 将热力图转换为RGB格式
            feature_map_sum = np.uint8(255 * feature_map_sum1)
            # 将热利用应用于原始图像
            feature_map_sum = cv2.applyColorMap(feature_map_sum, cv2.COLORMAP_JET)
            feature_map_sum = cv2.cvtColor(feature_map_sum, cv2.COLOR_BGR2RGB)  # 转换颜色映射

            # feature_map_sum[np.where(feature_map_sum1 < 0.2)] = 0
            # 　这里的热力图因子是０.４
            superimposed_img = feature_map_sum * 0.4 + input_img1
            cv2.imwrite("person1_{}".format(v), superimposed_img)



        data_path_abs = 'C:\\Users\\Administrator\\Desktop\\DriverDetection\\auc'
        img_list_all = os.listdir(data_path_abs)
        for key, v in enumerate(img_list_all):
            input_img1 = cv2.imread(data_path_abs + "/" + v)
            input_img1 = cv2.resize(input_img1, (224, 224))

            sobel_image = np.expand_dims(input_img1, axis=0)
            # sobel_image = sobel_image - np.mean(sobel_image, axis=0)
            predictions = self.model.predict(sobel_image)

            #     # 获取最后一层卷积层的输出
            last_conv_layer = self.model.get_layer('conv_pw_13_relu')
            last_conv_layer_output = last_conv_layer.output
            model_with_last_conv = Model(inputs=self.model.input, outputs=last_conv_layer_output)
            feature_map = model_with_last_conv.predict(sobel_image)
            aa = visualize_feature_map(feature_map)

            input_img1 = cv2.imread(data_path_abs + "/" + v)
            input_img1 = cv2.resize(input_img1, (224, 224))

            feature_map_sum1 = cv2.resize(aa, (224, 224))
            # 将热力图转换为RGB格式
            feature_map_sum = np.uint8(255 * feature_map_sum1)
            # 将热利用应用于原始图像
            feature_map_sum = cv2.applyColorMap(feature_map_sum, cv2.COLORMAP_JET)
            feature_map_sum = cv2.cvtColor(feature_map_sum, cv2.COLOR_BGR2RGB)  # 转换颜色映射

            # feature_map_sum[np.where(feature_map_sum1 < 0.2)] = 0

            # 　这里的热力图因子是０.４
            superimposed_img = feature_map_sum * 0.4 + input_img1
            cv2.imwrite("person2_{}".format(v), superimposed_img)


        # 靠谱方式
        data_path_abs = 'C:\\Users\\Administrator\\Desktop\\DriverDetection\\auc'
        img_list_all = os.listdir(data_path_abs)
        for key, v in enumerate(img_list_all):
            input_img1 = cv2.imread(data_path_abs + "/" + v)
            input_img1 = cv2.resize(input_img1, (224, 224))

            sobel_image = np.expand_dims(input_img1, axis=0)
            # sobel_image = sobel_image - np.mean(sobel_image, axis=0)
            predictions = self.model.predict(sobel_image)

            # cv2.imshow('Segmentation', sobel_image)
            # cv2.waitKey(1500)
            # cv2.destroyAllWindows()

            #     # 获取最后一层卷积层的输出
            last_conv_layer = self.model.get_layer('conv_pw_13_relu')
            grad_model = Model([self.model.inputs], [last_conv_layer.output, self.model.output])

            #     # 计算类别的梯度
            with tf.GradientTape() as tape:
                conv_layer_output, preds = grad_model(sobel_image)
                class_channel = preds[0][np.argmax(preds[0])]
                print(class_channel, end="====")
                print("kaopu1111")
            # 计算梯度
            grads = tape.gradient(class_channel, conv_layer_output)
            # 计算权重
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            # pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
            # heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_layer_output), axis=-1)
            heatmap = conv_layer_output @ pooled_grads[..., tf.newaxis]

            # feature_map_sum = sum(ele for ele in feature_map_combination)
            heatmap = tf.maximum(heatmap, 0)
            heatmap /= tf.reduce_max(heatmap)
            #
            # cv2.imshow('Segmentation', heatmap)
            # cv2.waitKey(1500)
            # cv2.destroyAllWindows()

            # 重塑热力图并将其缩放到与原始图像相同的大小
            heatmap = np.squeeze(heatmap)
            #     heatmap = cv2.resize(heatmap, (224, 224))
            #     heatmap = np.uint8(255 * heatmap)
            #     heatmap = np.clip(heatmap, 0, 1)
            # 颜色映射
            gbkInput = cv2.imread(data_path_abs + "/" + v)
            gbkInput = cv2.resize(gbkInput, (224, 224))

            feature_map_sum1 = cv2.resize(heatmap, (224, 224))
            # 将热力图转换为RGB格式
            feature_map_sum = np.uint8(255 * feature_map_sum1)

            # 将热利用应用于原始图像
            feature_map_sum = cv2.applyColorMap(feature_map_sum, cv2.COLORMAP_JET)
            feature_map_sum = cv2.cvtColor(feature_map_sum, cv2.COLOR_BGR2RGB)  # 转换颜色映射
            # feature_map_sum[np.where(feature_map_sum1 < 0.2)] = 0
            # 　这里的热力图因子是０.４
            superimposed_img = feature_map_sum * 0.4 + gbkInput
            cv2.imwrite("personkaopu22{}".format(v), superimposed_img)




        # 靠谱方式
        data_path_abs = 'C:\\Users\\Administrator\\Desktop\\DriverDetection\\auc'
        img_list_all = os.listdir(data_path_abs)
        for key, v in enumerate(img_list_all):
            input_img1 = cv2.imread(data_path_abs + "/" + v)
            input_img1 = cv2.resize(input_img1, (224, 224))

            sobel_image = np.expand_dims(input_img1, axis=0)
            sobel_image = sobel_image - np.mean(sobel_image, axis=0)
            predictions = self.model.predict(sobel_image)
            # cv2.imshow('Segmentation', sobel_image)
            # cv2.waitKey(1500)
            # cv2.destroyAllWindows()

            #     # 获取最后一层卷积层的输出
            last_conv_layer = self.model.get_layer('conv_pw_13_relu')
            grad_model = Model([self.model.inputs], [last_conv_layer.output, self.model.output])

            #     # 计算类别的梯度
            with tf.GradientTape() as tape:
                conv_layer_output, preds = grad_model(sobel_image)
                class_channel = preds[0][np.argmax(preds[0])]
                print(class_channel, end="====")
                print("kaopu1111")
            # 计算梯度
            grads = tape.gradient(class_channel, conv_layer_output)
            # 计算权重
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            # pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
            # heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_layer_output), axis=-1)
            heatmap = conv_layer_output @ pooled_grads[..., tf.newaxis]

            # feature_map_sum = sum(ele for ele in feature_map_combination)
            heatmap = tf.maximum(heatmap, 0)
            heatmap /= tf.reduce_max(heatmap)
            #
            # cv2.imshow('Segmentation', heatmap)
            # cv2.waitKey(1500)
            # cv2.destroyAllWindows()
            # 重塑热力图并将其缩放到与原始图像相同的大小
            heatmap = np.squeeze(heatmap)
            #     heatmap = cv2.resize(heatmap, (224, 224))
            #     heatmap = np.uint8(255 * heatmap)
            #     heatmap = np.clip(heatmap, 0, 1)
            # 颜色映射
            gbkInput = cv2.imread(data_path_abs + "/" + v)
            gbkInput = cv2.resize(gbkInput, (224, 224))

            feature_map_sum1 = cv2.resize(heatmap, (224, 224))
            # 将热力图转换为RGB格式
            feature_map_sum = np.uint8(255 * feature_map_sum1)
            # 将热利用应用于原始图像
            feature_map_sum = cv2.applyColorMap(feature_map_sum, cv2.COLORMAP_JET)
            feature_map_sum = cv2.cvtColor(feature_map_sum, cv2.COLOR_BGR2RGB)  # 转换颜色映射

            # feature_map_sum[np.where(feature_map_sum1 < 0.2)] = 0
            # 　这里的热力图因子是０.４
            superimposed_img = feature_map_sum * 0.4 + gbkInput
            cv2.imwrite("personkaopu11{}".format(v), superimposed_img)

        self.model.save('my_model.h5')

def make_gradcam_heatmap(img_array, model, last_conv_layer_name,name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:

            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    print(class_channel, end="====")
    print(name)

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col
def save_and_display_gradcam(img_path, heatmap,path,flag, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    input_img1 = cv2.imread(img_path)
    img = cv2.resize(input_img1, (224, 224))
    # hhhh = cv2.resize(heatmap,(224, 224))
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)
    heatmap[heatmap < 80] = 0
    # Use jet colormap to colorize heatmap
    # jet = mpl.colormaps["jet"]
    # # Use RGB values of the colormap
    # jet_colors = jet(np.arange(256))[:, :3]
    # jet_heatmap = jet_colors[heatmap]

    jet_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # jet_heatmap = cv2.cvtColor(jet_heatmap, cv2.COLOR_BGR2RGB)  # 转换颜色映射

    # Create an image with RGB colorized heatmap
    jet_heatmap = tensorflow.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tensorflow.keras.preprocessing.image.img_to_array(jet_heatmap)

    # 将热利用应用于原始图像
    # feature_map_sum = cv2.applyColorMap(feature_map_sum, cv2.COLORMAP_JET)
    # jet_heatmap[np.where(hhhh < 0.2)] = 0

    superimposed_img = jet_heatmap * alpha + img
    # superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    cv2.imwrite(flag+"person_{}".format(path), superimposed_img)




def visualize_feature_map(img_batch):
    feature_map = np.squeeze(img_batch, axis=0)
    # print(feature_map.shape)

    feature_map_combination = []
    plt.figure()

    num_pic = feature_map.shape[2]
    row, col = get_row_col(num_pic)

    for i in range(0, num_pic):
        feature_map_split = feature_map[:, :, i]
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
    feature_map_sum = np.maximum(feature_map_sum, 0)
    feature_map_sum /= np.max(feature_map_sum)

    # print(feature_map_sum)

    # plt.imshow(feature_map_sum)
    # plt.savefig("{}".format("feature_map_sum"+str(uuid.uuid4())+".eps"))
    # plt.savefig("{}".format("feature_map_sum"+str(uuid.uuid4())+".jpg"))
    return feature_map_sum

    # cv2.imshow('Segmentation', feature_map_sum)
    # cv2.waitKey(4000)
    # cv2.destroyAllWindows()

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model

def grad_cam(model, img_array, layer_name, class_idx=None):
    # 创建梯度计算模型
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if class_idx is None:
            class_idx = np.argmax(predictions)
        loss = predictions[:, class_idx]

    # 计算梯度
    grads = tape.gradient(loss, conv_outputs)

    # 计算权重（全局平均池化梯度）
    weights = tf.reduce_mean(grads, axis=(0, 1))

    # 生成热力图
    cam = np.dot(conv_outputs, weights)
    cam = np.maximum(cam, 0)  # ReLU
    cam = cv2.resize(cam, (img_array.shape, img_array.shape))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # 归一化

    return cam



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


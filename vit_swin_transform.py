import matplotlib.pyplot as plt
import numpy as np
import tensorflow
import tensorflow as tf  # For tf.data and preprocessing only.
from keras.layers import Flatten
from keras_cv_attention_models import swin_transformer_v2,vit
import os
from keras_cv_attention_models.caformer import CAFormer
from keras_cv_attention_models import cmt
from keras_cv_attention_models.efficientformer import EfficientFormerV2
from keras_cv_attention_models.efficientvit import EfficientViT_M
from keras_cv_attention_models.halonet import HaloNet
from keras_cv_attention_models.iformer import InceptionTransformer
from keras_cv_attention_models import nat
from keras_cv_attention_models.vanillanet import VanillaNet
from tensorflow.keras.layers import Conv2D, MaxPooling2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, PReLU, Input, \
    BatchNormalization, GlobalMaxPooling2D
from tensorflow.python.keras.layers import Reshape

os.environ['KECAM_BACKEND'] = 'keras_core'
os.environ['KERAS_BACKEND'] = 'jax'

""" Run predict """
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


from keras_cv_attention_models import swin_transformer_v2,repvit,mobilevit,convnext,uniformer,iformer,caformer,davit,\
    maxvit,fastvit,flexivit,efficientformer,gpvit,gcvit,tinyvit,pvt,levit,coat,edgenext,resnext,lcnet,efficientnet,halonet,efficientdet,fbnetv3,meta_transformer,nfnets,maxvit
# model = swin_transformer_v2.SwinTransformerV2(input_shape=(224, 224, 3),pretrained=None)
# model = repvit.RepViT(input_shape=(224, 224, 3),pretrained=None)
model = mobilevit.MobileViT(input_shape=(224, 224, 3),pretrained=None)
# model = uniformer.Uniformer(input_shape=(224, 224, 3),pretrained=None)
# model = InceptionTransformer(input_shape=(224, 224, 3),pretrained=None)
# model = convnext.ConvNeXtV2(input_shape=(224, 224, 3),pretrained='None')
# model = efficientformer.EfficientFormer(input_shape=(224, 224, 3),pretrained=Nonde)
# model = gpvit.GPViT(input_shape=(224, 224, 3),pretrained=None)
# model = caformer.ConvFormerS18(input_shape=(224, 224, 3),pretrained=None)
# model = fastvit.FastViT(input_shape=(224, 224, 3),pretrained=None)
# model = flexivit.FlexiViT(input_shape=(224, 224, 3),pretrained=None)
# model = davit.DaViT(input_shape=(224, 224, 3),pretrained=None)
# model = tinyvit.TinyViT(input_shape=(224, 224, 3),pretrained=None)
# model = cmt.CMTTiny (input_shape=(224, 224, 3),pretrained=None)
# model = lcnet.LCNet(input_shape=(224, 224, 3),pretrained=None)
# model = EfficientFormerV2(input_shape=(224, 224, 3),pretrained=None)
# model = efficientnet.EfficientNetV2(input_shape=(224, 224, 3),pretrained=None)
# model = efficientnet.EfficientNetV1(input_shape=(224, 224, 3),pretrained=None)
# model = mobilevit.MobileViT_V2(input_shape=(224, 224, 3),pretrained=None)
# model = resnext.ResNeXt50(input_shape=(224, 224, 3),pretrained=None)
# model = fbnetv3.FBNetV3(input_shape=(224, 224, 3),pretrained=None)
# model = fbnetv3.FBNetV3D(input_shape=(224, 224, 3),pretrained=None)
# model = fbnetv3.FBNetV3G(input_shape=(224, 224, 3),pretrained=None)
# model = fbnetv3.FBNetV3B(input_shape=(224, 224, 3),pretrained=None)
# model = pvt.PyramidVisionTransformerV2(input_shape=(224, 224, 3),pretrained="imagenet")
# model = mobilevit.MobileViT_S(input_shape=(224, 224, 3),pretrained=None)
# model = efficientformer.EfficientFormerL1(input_shape=(224, 224, 3),pretrained=None)
# model = efficientformer.EfficientFormerV2L(input_shape=(224, 224, 3),pretrained=None)
#
x = model.layers[-2].output

outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
custom_model = tf.keras.Model(model.input, outputs)
custom_model.summary()

momentum_optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.95)
custom_model.compile(optimizer=momentum_optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
history = custom_model.fit(
    X_train,y_train,
    batch_size=32,
    epochs=200,
    validation_data=(X_valid,y_valid),
    # callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)]
)
custom_model.save("custom_model.keras")

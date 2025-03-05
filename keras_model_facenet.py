########################################################################
# import python-library
########################################################################
# from import
import keras.models
from keras import backend as K
from keras import layers
from keras.models import Model
from keras.initializers import orthogonal
from keras import activations
import tensorflow as tf
import numpy as np
from keras.layers import Input, Concatenate, Dense
from keras.layers import Input, Dense, BatchNormalization, Activation, Lambda
from keras.models import Model

from keras.utils import metrics_utils
import os

import numpy as np
from tensorflow.python.util.tf_export import keras_export
from tensorflow.keras.callbacks import Callback


###################################
#         Loss Function_4
###################################
def BinaryCrossEntropy_machine(y_true, y_pred):
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    term_0 = (1 - y_true) * K.log(1 - y_pred + K.epsilon())  
    term_1 = y_true * K.log(y_pred + K.epsilon())
    result = -K.mean(term_0 + term_1, axis=0)
    return result

def binary_crossentropy_test(y_true, y_pred):
    batch_size = K.shape(y_true)[0]
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    term_0 = (1 - y_true[:,:-1]) * K.log(1 - y_pred + K.epsilon())  
    term_1 = y_true[:,:-1] * K.log(y_pred + K.epsilon())
    size = tf.cast(tf.size(term_0 + term_1), dtype=tf.float32) * tf.cast(tf.size(y_true[:,-1]), dtype=tf.float32)
    machine_true = tf.reshape(y_true[:,-1], [batch_size, 1])
    result = -K.mean(K.mean((term_0 + term_1)*machine_true, axis=0))
    return result

def categorical_cross_entropy(y_true, y_pred):
    batch_size = K.shape(y_true)[0]
    # y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    term_1 = y_true[:,:-1] * K.log(y_pred + K.epsilon())
    machine_true = tf.reshape(y_true[:,-1], [batch_size, 1])
    result = -K.mean(K.sum(term_1*machine_true, axis=0))
    return result


def BinaryCrossEntropy_section(y_true, y_pred): 
    batch_size = K.shape(y_true)[0]
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    term_0 = (1 - y_true[:,:-1]) * K.log(1 - y_pred + K.epsilon())  
    term_1 = y_true[:,:-1] * K.log(y_pred + K.epsilon())
    size = tf.cast(tf.size(term_0 + term_1), dtype=tf.float32) * tf.cast(tf.size(y_true[:,-1]), dtype=tf.float32)
    machine_true = tf.reshape(y_true[:,-1], [batch_size, 1])
    result = -tf.math.divide_no_nan(K.sum(K.sum((term_0 + term_1)*machine_true, axis=0)), (size + K.epsilon()))
    return result

def binary_accuracy_section(y_true, y_pred):
    batch_size = K.shape(y_true)[0]
    machine_true = tf.reshape(y_true[:,-1], [batch_size, 1])
    return K.mean(K.equal(y_true[:,:-1]*machine_true, K.round(y_pred)), axis=-1)

def categorical_accuracy_section(y_true, y_pred, class_to_ignore=0):
    batch_size = K.shape(y_true)[0]
    machine_true = tf.reshape(y_true[:,-1], [batch_size, 1])
    return metrics_utils.sparse_categorical_matches(
        tf.math.argmax(y_true*machine_true, axis=-1), y_pred
    )


def l2_norm(x):
    x = x ** 2
    x = K.sum(x, axis=1)
    x = tf.expand_dims(x, axis=1)
    return x

########################################################################
# keras model
########################################################################

def Conv2DLayer(inputs, filters, kernel, strides, linear=False, padding='same'):
    x = layers.Conv2D(filters, kernel_size=kernel, strides=strides, padding=padding)(inputs)
    x = layers.BatchNormalization()(x)
    # x = layers.Dropout(0.5, name=prefix+'drop')((x))
    if not linear:
        x = layers.ReLU()(x)
    return x

def DeConv2DLayer(inputs, filters, kernel, strides, linear=False, padding='same'):
    x = layers.Conv2DTranspose(filters, kernel_size=kernel, strides=strides, padding=padding)(inputs)
    # x = layers.BatchNormalization()(x)
    # x = layers.Dropout(0.5, name=prefix+'drop')((x))
    if not linear:
        x = layers.ReLU()(x)
    return x


def GDConv(inputs, kernel, strides, padding='same'):
    """Global Depthwise Convolution
    This function defines a Global Depthwise Convolution.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        padding: Integer, expansion factor.
            t is always applied to the input size.
    # Returns
        Output tensor.
    """
    x = layers.DepthwiseConv2D(kernel_size=kernel, strides=strides, padding=padding)(inputs)
    x = layers.BatchNormalization()(x)
    return x


def bottleneck(inputs, filters, kernel, t, s, r=False):
    """Bottleneck
    This function defines a basic bottleneck structure.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        r: Boolean, Whether to use the residuals.
    # Returns
        Output tensor.
    """

    tchannel = inputs.shape[-1] * t

    x = Conv2DLayer(inputs, tchannel, (1, 1), (1, 1))

    x = layers.DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)

    if r:
        x = tf.keras.layers.add([x, inputs])
    return x


def InvertedRes(inputs, filters, kernel, t, strides, n):
    """Inverted Residual Block
    This function defines a sequence of 1 or more identical layers.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        n: Integer, layer repeat times.
    # Returns
        Output tensor.
    """
    x = bottleneck(inputs, filters, kernel, t, strides)

    for i in range(1, n):
        x = bottleneck(x, filters, kernel, t, 1, True)

    return x




#########################################################################


def get_model(input_shape, num_class=4):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    x = Conv2DLayer(x, 64, (3, 3), strides=(2, 2))
    x = Conv2DLayer(x, 64, (3, 3), strides=(1, 1))
    x = InvertedRes(x, 64, (3, 3), t=2, strides=2, n=5)
    x = InvertedRes(x, 128, (3, 3), t=4, strides=2, n=1)
    x = InvertedRes(x, 128, (3, 3), t=2, strides=2, n=6)
    x = InvertedRes(x, 128, (3, 3), t=4, strides=2, n=1)
    x = InvertedRes(x, 128, (3, 3), t=2, strides=1, n=2)
    x = Conv2DLayer(x, 512, (1, 1), strides=(1, 1))
    x = GDConv(x, kernel=(4, 4), strides=1)
    x1 = layers.Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x1 = layers.Dropout(0.1, name='Dropout1')(x1)
    # x2 = layers.Dropout(0.5, name='Dropout2')(x2)
    x1 = layers.GlobalAveragePooling2D()(x1)

    x1 = layers.Dense(128, activation = 'relu')(x1)
    x1 = layers.Dense(128, activation = 'relu')(x1)
    g_linear = layers.Dense(num_class, activation = 'sigmoid', use_bias=False, name='linear')(x1)
    
    x2 = Lambda(lambda x: l2_norm(x))(x1)
    g_affine = layers.Dense(1, activation = 'sigmoid', use_bias=True, name='affine')(x2)
    
    
    model = Model(inputs=inputs, outputs=[g_affine, g_linear]) 
    return model

# def get_model(input_shape, num_class=4):
#     inputs = layers.Input(shape=input_shape)
#     x = inputs
#     x = Conv2DLayer(x, 64, (3, 3), strides=(2, 2))
#     x = Conv2DLayer(x, 64, (3, 3), strides=(1, 1))
#     x = InvertedRes(x, 128, (3, 3), t=2, strides=2, n=2)
#     x = InvertedRes(x, 128, (3, 3), t=4, strides=2, n=2)
#     x = InvertedRes(x, 128, (3, 3), t=4, strides=2, n=2)
#     x = Conv2DLayer(x, 512, (1, 1), strides=(1, 1))
#     x = GDConv(x, kernel=(7, 7), strides=2)
#     x = Conv2DLayer(x, 128, (1, 1), strides=(1, 1), linear=True)
#     x = layers.Dropout(0.2, name='Dropout')(x)
#     x = layers.GlobalAveragePooling2D()(x)
#     x = layers.ReLU()(x)
#     g_linear = layers.Dense(num_class, activation = 'sigmoid', use_bias=False)(x)
#     x = Lambda(lambda x: l2_norm(x))(x)
#     g_affine = layers.Dense(1, activation = 'sigmoid', use_bias=True)(x)
#     model = Model(inputs=inputs, outputs=[g_affine, g_linear]) 
#     # TEST ADAMW
# #     optimizer = AdamW(learning_rate=lr_schedule(0), weight_decay=wd_schedule(0))
# #     tb_callback = tf.keras.callbacks.TensorBoard(os.path.join('logs', 'adamw'),
# #                                                  profile_batch=0)
# #     lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
# #     wd_callback = WeightDecayScheduler(wd_schedule)
    
# #     model.compile(optimizer = optimizer, loss = {"dense_1" :"binary_crossentropy",
# #                                                                                       "dense":binary_crossentropy_test},
# #                   metrics= {"dense_1"  :'binary_accuracy', "dense": binary_accuracy_section},
# #                   loss_weights=[1, 0.1])
#     # model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001), loss = 
#     #               {"dense_1":"binary_crossentropy", "dense":binary_crossentropy_test},
#     #               metrics= {"dense_1"  :'binary_accuracy', "dense": binary_accuracy_section},
#     #               loss_weights=[10, 1])
#     # model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001), loss = 
#     #           {"dense_1":"binary_crossentropy", "dense":categorical_cross_entropy},
#     #           metrics= {"dense_1"  :'binary_accuracy', "dense": categorical_accuracy_section},
#     #           loss_weights=[1, 1])
#     return model


def load_model(file_path):
    return keras.models.load_model(file_path, compile=False)


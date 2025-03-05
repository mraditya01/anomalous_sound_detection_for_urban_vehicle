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
import math


######################################################################

    
########################################################################
# keras model
########################################################################
def tanh_custom(x):
    return tf.tanh(x)*1.2

def Conv2DLayer_crnn(x, filters, kernel, pool_size, strides, padding, block_id):
    prefix = f'block_{block_id}_'
    x = layers.Conv2D(filters, kernel_size=kernel, strides=strides, padding=padding,
                       name=prefix+'conv')(x)
    x = layers.BatchNormalization(name=prefix+'conv_bn')(x)
    x = layers.ReLU(name=prefix+'lrelu')(x)
    x = layers.MaxPooling2D(pool_size=pool_size, strides=pool_size, padding="valid")(x)
    return x


def Conv2DLayer(x, filters, kernel, strides, padding, block_id, last=False):
    prefix = f'block_{block_id}_'
    x = layers.Conv2D(filters, kernel_size=kernel, strides=strides, padding=padding,
                       name=prefix+'conv')(x)
    x = layers.BatchNormalization(name=prefix+'conv_bn')(x)
    x = layers.LeakyReLU(name=prefix+'lrelu')(x)
    # x = layers.ELU(name=prefix+'elu')(x)
    return x


########################################################################
# FACENET

def Conv2DLayer_facenet(inputs, filters, kernel, strides, linear=False, padding='same'):
    x = layers.Conv2D(filters, kernel_size=kernel, strides=strides, padding=padding)(inputs)
    x = layers.BatchNormalization()(x)
    # x = layers.Dropout(0.5, name=prefix+'drop')(x)
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

    x = Conv2DLayer_facenet(inputs, tchannel, (1, 1), (1, 1))

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


def masked_mse(y_gt, model_out):
    # SED mask: Use only the predicted DOAs when gt SED > 0.5
    sed_out = y_gt[:, :, :14] >= 0.5 #TODO fix this hardcoded value of number of classes
    sed_out = keras.backend.repeat_elements(sed_out, 3, -1)
    sed_out = keras.backend.cast(sed_out, 'float32')

    # Use the mask to computed mse now. Normalize with the mask weights #TODO fix this hardcoded value of number of classes
    return keras.backend.sqrt(keras.backend.sum(keras.backend.square(y_gt[:, :, 14:] - model_out[:, :, 14:]) * sed_out))/keras.backend.sum(sed_out)


def ResnetBlockV2(input_tensor, training=True, kernel_size=(3,3), filter_size=16, stride=1, dif_fsize=False):
    use_identity_shortcut = (stride==1) and not dif_fsize
    if not use_identity_shortcut:
        conv2_sc = tf.keras.layers.Conv2D(filter_size, (1,1), strides=stride, padding='same')

    x = BatchNormalization()(input_tensor, training=training)
    x1 = tf.nn.relu(x) # shortcutがidentityではない場合ここから分岐させる
    x = Conv2D(filter_size, kernel_size, strides=stride, padding='same')(x1) # こちらは残差ブロック側

    x = BatchNormalization()(x, training=training)    
    x = tf.nn.relu(x)
    x = Conv2D(filter_size, kernel_size, strides=(1,1), padding='same')(x)
    use_identity_shortcut = (stride==1) and not dif_fsize
    if use_identity_shortcut:
        skip = input_tensor
    else:
        skip = conv2_sc(x1)
    x += skip
    return x
    
    

#########################################################################
from keras.layers.core import  Activation
from keras.layers import Dense, GRU, Bidirectional, Conv2D, MaxPooling2D, Input, Concatenate, Dropout, Reshape, Permute, BatchNormalization, TimeDistributed, AveragePooling2D, LSTM, ELU, add, multiply,GlobalAveragePooling2D, Conv2DTranspose
from keras.models import Model
from keras.optimizers import Adam
from keras.models import load_model
import keras
from IPython import embed
import numpy as np




def squeeze_excite_block(input_tensor, ratio=16):
    """ Create a channel-wise squeeze-excite block
    Args:
        input_tensor: input Keras tensor
        ratio: number of output filters
    Returns: a Keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    """
    init = input_tensor
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = tf.shape(init)[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


def spatial_squeeze_excite_block(input_tensor):
    """ Create a spatial squeeze-excite block
    Args:
        input_tensor: input Keras tensor
    Returns: a Keras tensor
    References
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    """

    se = Conv2D(1, (1, 1), activation='sigmoid', use_bias=False,
                kernel_initializer='he_normal')(input_tensor)

    x = multiply([input_tensor, se])
    return x


def channel_spatial_squeeze_excite(input_tensor, ratio=16):
    """ Create a spatial squeeze-excite block
    Args:
        input_tensor: input Keras tensor
        ratio: number of output filters
    Returns: a Keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    """

    cse = squeeze_excite_block(input_tensor, ratio)
    sse = spatial_squeeze_excite_block(input_tensor)

    x = add([cse, sse])
    return x


def _tensor_shape(tensor):
    return getattr(tensor, '_keras_shape')


        

def get_model_newest(input_shape, dropout_rate=0, nb_cnn2d_filt=128, f_pool_size=[4, 4, 2], t_pool_size=[2, 2, 1], rnn_size=[128, 128], fnn_size=[128, 128, 128], output=37, activation='softmax'):
    # model definition
    spec_start = Input(shape=(input_shape))
    spec_cnn = spec_start
    for i, convCnt in enumerate(f_pool_size):
        spec_cnn = Conv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding='same')(spec_cnn)
        spec_cnn = BatchNormalization()(spec_cnn)
        spec_cnn = Activation('relu')(spec_cnn)
        spec_cnn = MaxPooling2D(pool_size=(f_pool_size[i], t_pool_size[i]))(spec_cnn)
        spec_cnn = Dropout(0.05)(spec_cnn)
    spec_cnn = Permute((2, 1, 3))(spec_cnn)

    
    # RNN
    spec_rnn = Reshape((128, -1))(spec_cnn)
    for nb_rnn_filt in rnn_size:
        spec_rnn = Bidirectional(
            GRU(nb_rnn_filt, activation='tanh', dropout=0.05, recurrent_dropout=0.05,
                return_sequences=True),
            merge_mode='mul'
        )(spec_rnn)
        

    # FC - DOA
    doa = spec_rnn
    for nb_fnn_filt in fnn_size:
        doa = TimeDistributed(Dense(nb_fnn_filt))(doa)
        doa = Dropout(0.05)(doa)
    doa = TimeDistributed(Dense(output))(doa)
    doa = Activation(activation, name='doa_out')(doa)
    
    # SED ==================================================================
    sed = spec_rnn
    for nb_fnn_filt in fnn_size:
        sed = TimeDistributed(Dense(nb_fnn_filt))(sed)
        sed = Dropout(0.05)(sed)
    sed = TimeDistributed(Dense(4))(sed)
    sed = Activation('sigmoid', name='sed_out')(sed)

    
    model = Model(inputs=spec_start, outputs=[doa, sed])
    return model
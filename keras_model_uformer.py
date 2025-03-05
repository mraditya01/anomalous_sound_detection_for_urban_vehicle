import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer
from einops.layers.tensorflow import Rearrange
from einops import rearrange
from keras import backend as K
from keras import layers
from keras.models import Model
from keras.initializers import orthogonal
from keras import activations
import tensorflow as tf
import numpy as np
import math
from keras.layers.core import  Activation
from keras.layers import Dense, GRU, Bidirectional, Conv2D, MaxPooling2D, Input, Concatenate, Dropout, Reshape, Permute, BatchNormalization, TimeDistributed, AveragePooling2D, LSTM, ELU, add, multiply, GlobalAveragePooling2D, Conv2DTranspose, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.models import load_model
import keras
from IPython import embed
import numpy as np
from keras.layers import Bidirectional, Conv2D, MaxPooling2D, Input
from keras.layers.core import Dense, Activation, Dropout, Reshape, Permute
from keras.models import Model
from keras.optimizers import Adam
import keras
# keras.backend.set_image_data_format('channels_first')
from IPython import embed

# from tensorflow.python.framework.ops import disable_eager_execution

# disable_eager_execution()


"""# Custom layers"""

class AudioClipLayer(Layer):

    def __init__(self, **kwargs):
        '''Initializes the instance attributes'''
        super(AudioClipLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        '''Create the state of the layer (weights)'''
        # initialize the weights
        pass
        
    def call(self, inputs, training):
        '''Defines the computation from inputs to outputs'''
        if training:
            return inputs
        else:
            return tf.maximum(tf.minimum(inputs, 1.0), -1.0)

class MelSpectrogramLayer(Layer):
    def __init__(self, **kwargs):
        '''Initializes the instance attributes'''
        super(AudioClipLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        '''Create the state of the layer (weights)'''
        # initialize the weights
        pass
        
    def call(self, inputs):
        '''Defines the computation from inputs to outputs'''
        y_true = K.mean(inputs, axis=-1)
        spectrogram_true = tfio.audio.spectrogram(y_true, nfft=2823, window=2823, stride=347)
        mel_spectrogram_true = tfio.audio.melscale(spectrogram_true, 22050, mels=128, fmin=0, fmax=11025, name=None)
        dbscale_mel_spectrogram_true = tfio.audio.dbscale(mel_spectrogram_true, top_db=80)
        return dbscale_mel_spectrogram_true
    
# Learned Interpolation layer

class InterpolationLayer(Layer):

    def __init__(self, padding = "valid", **kwargs):
        '''Initializes the instance attributes'''
        super(InterpolationLayer, self).__init__(**kwargs)
        self.padding = padding
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'padding': self.padding,
        })
        return config
    
    def build(self, input_shape):
        '''Create the state of the layer (weights)'''
        self.features = input_shape.as_list()[3]

        # initialize the weights
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(name="kernel",
            initial_value=w_init(shape=(self.features, ),
                                 dtype='float32'),
            trainable=True)

    def call(self, inputs):
        '''Defines the computation from inputs to outputs'''

        w_scaled = tf.math.sigmoid(self.w)

        counter_w = 1 - w_scaled

        conv_weights = tf.expand_dims(tf.concat([tf.expand_dims(tf.linalg.diag(w_scaled), axis=0), tf.expand_dims(tf.linalg.diag(counter_w), axis=0)], axis=0), axis=0)

        intermediate_vals = tf.nn.conv2d(inputs, conv_weights, strides=[1,1,1,1], padding=self.padding.upper())

        intermediate_vals = tf.transpose(intermediate_vals, [2, 0, 1, 3])
        out = tf.transpose(inputs, [2, 0, 1, 3])
        
        num_entries = out.shape.as_list()[0]
        out = tf.concat([out, intermediate_vals], axis=0)

        indices = list()

        # num_outputs = 2*num_entries - 1
        num_outputs = (2*num_entries - 1) if self.padding == "valid" else 2*num_entries

        for idx in range(num_outputs):
            if idx % 2 == 0:
                indices.append(idx // 2)
            else:
                indices.append(num_entries + idx//2)
        out = tf.gather(out, indices)
        current_layer = tf.transpose(out, [1, 2, 0, 3])

        return current_layer

class CropLayer(Layer):
    def __init__(self, shape, match_feature_dim=True,  **kwargs):
        '''Initializes the instance attributes'''
        super(CropLayer, self).__init__(**kwargs)
        self.match_feature_dim = match_feature_dim
        self.shape = shape

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'match_feature_dim' : self.match_feature_dim,
            'shape' : self.shape
        })
        return config
    
    def build(self, input_shape):
        '''Create the state of the layer (weights)'''
        # initialize the weights
        pass
        
    def call(self, inputs):
        '''Defines the computation from inputs to outputs'''
        if self.shape is None:
            return inputs
        inputs = self.crop(inputs, self.shape, self.match_feature_dim)
        return inputs

    def crop(self, tensor, target_shape, match_feature_dim=True):
        '''
        Crops a 3D tensor [batch_size, width, channels] along the width axes to a target shape.
        Performs a centre crop. If the dimension difference is uneven, crop last dimensions first.
        :param tensor: 4D tensor [batch_size, width, height, channels] that should be cropped. 
        :param target_shape: Target shape (4D tensor) that the tensor should be cropped to
        :return: Cropped tensor
        '''
        shape = np.array(tensor.shape.as_list())

        ddif = shape[1] - target_shape[1]
        # if (ddif % 2 != 0):
        #     # print("WARNING: Cropping with uneven number of extra entries on one side")
        # assert ddiff[1] >= 0 # Only positive difference allowed
        if ddif == 0:
            return tensor
        crop_start = ddif // 2
        crop_end = ddif - crop_start

        return tensor[:,crop_start:-crop_end,:]

class IndependentOutputLayer(Layer):

    def __init__(self, source_names, num_channels, filter_width, padding="valid", **kwargs):
        '''Initializes the instance attributes'''
        super(IndependentOutputLayer, self).__init__(**kwargs)
        self.source_names = source_names
        self.num_channels = num_channels
        self.filter_width = filter_width
        self.padding = padding

        self.conv1a = tf.keras.layers.Conv1D(self.num_channels, self.filter_width, padding= self.padding, activation='tanh')
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'source_names' : self.source_names,
            'num_channels' : self.num_channels,
            'filter_width' : self.filter_width,
            'padding' : self.padding
        })
        return config
    

    def build(self, input_shape):
        '''Create the state of the layer (weights)'''
        pass
        
    def call(self, inputs, training):
        '''Defines the computation from inputs to outputs'''
        return self.conv1a(inputs)

class CropLayer2(Layer):
    def __init__(self, x2, match_feature_dim=True, **kwargs):
        '''Initializes the instance attributes'''
        super(CropLayer2, self).__init__(**kwargs)
        self.match_feature_dim = match_feature_dim
        self.x2 = x2

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'match_feature_dim' : self.match_feature_dim,
        })
        return config
    
    def build(self, input_shape):
        '''Create the state of the layer (weights)'''
        # initialize the weights
        pass
        
    def call(self, inputs):
        '''Defines the computation from inputs to outputs'''
        if self.x2 is None:
            return inputs

        inputs = self.crop(inputs, self.x2.shape.as_list(), self.match_feature_dim)
        return inputs

    def crop(self, tensor, target_shape, match_feature_dim=True):
        '''
        Crops a 3D tensor [batch_size, width, channels] along the width axes to a target shape.
        Performs a centre crop. If the dimension difference is uneven, crop last dimensions first.
        :param tensor: 4D tensor [batch_size, width, height, channels] that should be cropped. 
        :param target_shape: Target shape (4D tensor) that the tensor should be cropped to
        :return: Cropped tensor
        '''
        shape = np.array(tensor.shape.as_list())

        ddif = shape[1] - target_shape[1]

        if (ddif % 2 != 0):
            print("WARNING: Cropping with uneven number of extra entries on one side")
        # assert diff[1] >= 0 # Only positive difference allowed
        if ddif == 0:
            return tensor
        crop_start = ddif // 2
        crop_end = ddif - crop_start

        return tensor[:,crop_start:-crop_end,:]

class ConformerConvModule(tf.keras.layers.Layer):
    def __init__(
        self,
        dim,
        causal=False,
        expansion_factor=2,
        kernel_size=31,
        dropout=0.0,
        **kwargs
    ):
        super(ConformerConvModule, self).__init__(**kwargs)

        inner_dim = dim * expansion_factor
        if not causal:
            padding = (kernel_size // 2, kernel_size // 2 - (kernel_size + 1) % 2)
        else:
            padding = (kernel_size - 1, 0)

        self.net = tf.keras.Sequential(
            [
                tf.keras.layers.LayerNormalization(axis=-1),
                Rearrange("b n c -> b c n"),
                tf.keras.layers.Conv1D(filters=inner_dim * 2, kernel_size=1),
                GLU(dim=1),
                DepthwiseLayer(
                    inner_dim, inner_dim, kernel_size=kernel_size, padding=padding
                ),
                layers.LayerNormalization(axis=-1),
                Swish(),
                tf.keras.layers.Conv1D(filters=dim, kernel_size=1),
                # Rearrange("b c n -> b n c"),
                tf.keras.layers.Dropout(dropout),
            ]
        )

    def call(self, inputs):
        return self.net(inputs)

"""# Define the Network"""
class GLU(tf.keras.layers.Layer):
    def __init__(self, bias=True, dim=-1, **kwargs):
        super(GLU, self).__init__(**kwargs)
        self.bias = bias
        self.dim = dim
        self.dense = tf.keras.layers.Dense(2, use_bias=bias)
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'bias' : self.bias,
            "dim" : self.dim,
        })
        return config
    
    def call(self, x):
        out, gate = tf.split(x, num_or_size_splits=2, axis=self.dim)
        gate = tf.sigmoid(gate)
        x = tf.multiply(out, gate)
        return x
    
class ConformerBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        attn_dropout=0.1,
        ff_dropout=0.1,
        conv_dropout=0.1,
        **kwargs
    ):
        super(ConformerBlock, self).__init__(**kwargs)
        self.dim=dim,
        self.dim_head=64,
        self.heads=8,
        self.ff_mult=4,
        self.conv_expansion_factor=2,
        self.conv_kernel_size=31,
        self.attn_dropout=0.1,
        self.ff_dropout=0.1,
        self.conv_dropout=0.1,
        self.ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.attn = Attention(
            dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout
        )
        self.conv = ConformerConvModule(
            dim=dim,
            causal=False,
            expansion_factor=conv_expansion_factor,
            kernel_size=conv_kernel_size,
            dropout=conv_dropout,
        )
        self.ff2 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        
        self.attn = PreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = tf.keras.layers.LayerNormalization(axis=-1)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'dim' : self.dim,
            'dim_head' : self.dim_head,
            'heads' : self.heads,
            'ff_mult' : self.ff_mult,
            'conv_expansion_factor' : self.conv_expansion_factor,
            'conv_kernel_size' : self.conv_kernel_size,
            'attn_dropout' : self.attn_dropout,
            'ff_dropout' : self.ff_dropout,
            'conv_dropout': self.conv_dropout
        })
        return config
    
    def call(self, inputs, mask=None):
        inputs = self.ff1(inputs) + inputs
        inputs = self.attn(inputs, mask=mask) + inputs
        inputs = self.conv(inputs) + inputs
        inputs = self.ff2(inputs) + inputs
        inputs = self.post_norm(inputs)
        return inputs

class DiffOutputLayer(Layer):

    def __init__(self, source_names, num_channels, filter_width, padding="valid", **kwargs):
        '''Initializes the instance attributes'''
        super(DiffOutputLayer, self).__init__(**kwargs)
        self.source_names = source_names
        self.num_channels = num_channels
        self.filter_width = filter_width
        self.padding = padding

        self.conv1a = tf.keras.layers.Conv1D(self.num_channels, self.filter_width, padding= self.padding)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'source_names' : self.source_names,
            'num_channels' : self.num_channels,
            'filter_width' : self.filter_width,
            'padding' : self.padding
        })
        return config
    
    def build(self, input_shape):
        '''Create the state of the layer (weights)'''
        pass
        
    def call(self, inputs, training):
        '''Defines the computation from inputs to outputs'''
        outputs = {}
        sum_source = 0
        for name in self.source_names[:-1]:
            out = self.conv1a(inputs[0])
            out = AudioClipLayer()(out)
            outputs[name] = out
            sum_source = sum_source + out
        
        last_source = CropLayer2(sum_source)(inputs[1]) - sum_source
        last_source = AudioClipLayer()(last_source)

        outputs[self.source_names[-1]] = last_source

        return outputs


    
class Attention(tf.keras.layers.Layer):
    def __init__(
        self, dim, heads=8, dim_head=64, dropout=0.0, max_pos_emb=512, **kwargs
    ):
        super(Attention, self).__init__(**kwargs)
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = tf.keras.layers.Dense(inner_dim, use_bias=False)
        self.to_kv = tf.keras.layers.Dense(inner_dim * 2, use_bias=False)
        self.to_out = tf.keras.layers.Dense(dim)

        self.max_pos_emb = max_pos_emb
        self.rel_pos_emb = tf.keras.layers.Embedding(2 * max_pos_emb + 1, dim_head)

        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, context=None, mask=None, context_mask=None):
        n = inputs.shape[-2]
        heads = self.heads
        max_pos_emb = self.max_pos_emb
        if context is None:
            has_context = False
            context = inputs
        else:
            has_context = True

        kv = tf.split(self.to_kv(context), num_or_size_splits=2, axis=-1)
        q, k, v = (self.to_q(inputs), *kv)

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=heads), (q, k, v)
        )
        dots = tf.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        seq = tf.range(n)
        dist = rearrange(seq, "i -> i ()") - rearrange(seq, "j -> () j")
        dist = (
            tf.clip_by_value(
                dist, clip_value_min=-max_pos_emb, clip_value_max=max_pos_emb
            )
            + max_pos_emb
        )
        rel_pos_emb = self.rel_pos_emb(dist)
        pos_attn = tf.einsum("b h n d, n r d -> b h n r", q, rel_pos_emb) * self.scale
        dots = dots + pos_attn

        if mask is not None or context_mask is not None:
            if mask is not None:
                mask = tf.ones(*inputs.shape[:2])
            if not has_context:
                if context_mask is None:
                    context_mask = mask
            else:
                if context_mask is None:
                    context_mask = tf.ones(*context.shape[:2])
            mask_value = -tf.experimental.numpy.finfo(dots.dtype).max
            mask = rearrange(mask, "b i -> b () i ()") * rearrange(
                context_mask, "b j -> b () () j"
            )
            dots = tf.where(mask, mask_value, dots)

        attn = tf.nn.softmax(dots, axis=-1)

        out = tf.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return self.dropout(out)

class Swish(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Swish, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs * tf.sigmoid(inputs)




class DepthwiseLayer(tf.keras.layers.Layer):
    def __init__(self, chan_in, chan_out, kernel_size, padding, **kwargs):
        super(DepthwiseLayer, self).__init__(**kwargs)
        self.padding = padding
        self.chan_in = chan_in
        self.conv = tf.keras.layers.Conv1D(chan_out, 1, groups=chan_in)

    def call(self, inputs):
        inputs = tf.reshape(inputs, [-1])
        padded = tf.zeros(
            [self.chan_in * self.chan_in * 1] - tf.shape(inputs), dtype=inputs.dtype
        )
        inputs = tf.concat([inputs, padded], 0)
        inputs = tf.reshape(inputs, [-1, self.chan_in, self.chan_in])

        return self.conv(inputs)


class Scale(tf.keras.layers.Layer):
    def __init__(self, scale, fn, **kwargs):
        super(Scale, self).__init__(**kwargs)
        self.scale = scale
        self.fn = fn

    def call(self, inputs, **kwargs):
        return self.fn(inputs, **kwargs) * self.scale


class PreNorm(tf.keras.layers.Layer):
    def __init__(self, dim, fn, **kwargs):
        super(PreNorm, self).__init__(**kwargs)
        self.norm = tf.keras.layers.LayerNormalization(axis=-1)
        self.fn = fn

    def call(self, inputs, **kwargs):
        inputs = self.norm(inputs)
        return self.fn(inputs, **kwargs)


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, dim, mult=4, dropout=0.0, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.net = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dim * mult, activation=Swish()),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(dim, input_dim=dim * mult),
                tf.keras.layers.Dropout(dropout),
            ]
        )

    def call(self, inputs):
        return self.net(inputs)


class BatchNorm(tf.keras.layers.Layer):
    def __init__(self, causal, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)
        self.causal = causal
        self.batchnorm = tf.keras.layers.BatchNormalization(axis=-1)

    def call(self, inputs):
        if not self.causal:
            return self.batchnorm(inputs)
        return tf.identity(inputs)
    
def wave_u_net(num_initial_filters = 24, num_layers = 12,
               source_names = ["bass", "drums", "other", "vocals"], num_channels = 1, output_filter_size = 1,
               padding = "same", input_size = 16384 * 4, context = False, upsampling_type = "learned",
               output_activation = "linear", output_type = "difference"):
  
  # `enc_outputs` stores the downsampled outputs to re-use during upsampling.
    enc_outputs = []

    # `raw_input` is the input to the network
    raw_input = tf.keras.layers.Input(shape=(input_size, num_channels),name="raw_input")
    X = raw_input
    X = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1), name="exp_dims_"+str(0))(X)
    X = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, [1, 32768]), name="bilinear_interpol_"+str(0))(X)
    X = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, [1, x.shape.as_list()[2] * 2]), name="bilinear_interpol_"+str(1))(X)
    X = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, [1, x.shape.as_list()[2] * 2]), name="bilinear_interpol_"+str(2))(X)
    # X = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, [1, x.shape.as_list()[2] * 2]), name="bilinear_interpol_"+str(3))(X)

    X = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=1), name="sq_dims_"+str(0))(X)
    inp = X

  # Down sampling
    for i in range(num_layers):
        # X = tf.keras.layers.Conv1D(filters=num_initial_filters + (num_initial_filters * i),
        #                       kernel_size=8,strides=4,
        #                       padding=padding, name=f"Down_Conv_{i}")(X)
        X = tf.keras.layers.Conv1D(filters=num_initial_filters*2**i,
                          kernel_size=8,strides=4,
                          padding=padding, name=f"Down_Conv_{i}")(X)
        X = tf.keras.layers.ReLU(name=f"Down_Conv_Activ_{i}")(X)
        # X = tf.keras.layers.Conv1D(filters=num_initial_filters + (num_initial_filters * (i+1)),
        #                   kernel_size=1,strides=1,
        #       padding=padding, name=f"Down_Conv_2_{i}")(X)
        X = tf.keras.layers.Conv1D(filters=num_initial_filters*2**i,
                  kernel_size=1,strides=1,
              padding=padding, name=f"Down_Conv_2_{i}")(X)

        X = GLU(dim=-1, name='glu_down'+str(i))(X)
        enc_outputs.append(X)


  
    # X = tf.keras.layers.Lambda(lambda x: x[:,::2,:], name="Decimate_"+str(i))(X)

#   X = tf.keras.layers.Conv1D(filters=num_initial_filters + (num_initial_filters * i),
#                           kernel_size=kernel_size,strides=1,
#                           padding=padding, name="Down_Conv_"+str(num_layers), dilation_rate=2**i)(X)

#   # X = tf.keras.layers.Conv1D(filters=num_initial_filters*2**num_layers,
#   #                         kernel_size=kernel_size,strides=1,
#   #                         padding=padding, name="Down_Conv_"+str(num_layers), dilation_rate=2**num_layers)(X)
  # X = tf.keras.layers.LeakyReLU(name="Down_Conv_Activ_"+str(num_layers))(X)
    X = ConformerBlock(dim = 256)(X)
    X = ConformerBlock(dim = 256)(X)

  # Up sampling
    for i in range(num_layers):
        c_layer = CropLayer(enc_outputs[-i-1].shape.as_list(), False, name="crop_layer_"+str(i))(X)
        X = tf.keras.layers.Concatenate(axis=2, name="concatenate_"+str(i))([c_layer, enc_outputs[-i-1]]) 
        X = tf.keras.layers.Conv1D(filters=num_initial_filters*2**(num_layers - i - 1),
                  kernel_size=1,strides=1,
                  padding=padding, name=f"Up_Conv_{i}")(X)
        # X = tf.keras.layers.Conv1D(filters=num_initial_filters + (num_initial_filters * (num_layers - i - 1)),
        #                         kernel_size=1,strides=1,
        #                         padding=padding, name="Up_Conv_"+str(i))(X)
        X = GLU(dim=-1, name='glu_up'+str(i))(X)
        X = tf.keras.layers.ReLU(name="Up_Conv_Activ_"+str(i))(X)
        X = tf.keras.layers.Conv1DTranspose(filters=num_initial_filters*2**(num_layers - i - 1),
                            strides=4, kernel_size=8,
                            padding=padding, name="Up_Conv_Trans"+str(i))(X)



# ######################################

    c_layer = CropLayer(X.shape.as_list(), False, name="crop_layer_"+str(num_layers))(inp)
    X = tf.keras.layers.Concatenate(axis=2, name="concatenate_"+str(num_layers))([X, c_layer]) 
    X = AudioClipLayer(name="audio_clip_"+str(0))(X, training = True)

    if output_type == "direct":
        X = IndependentOutputLayer(source_names, num_channels, output_filter_size, padding=padding, name="independent_out")(X)

#     else:
#         # Difference Output
#         cropped_input = CropLayer(X.shape.as_list(), False, name="crop_layer_"+str(num_layers+1))(inp)
#         X = DiffOutputLayer(source_names, num_channels, output_filter_size, padding=padding, name="diff_out")([X, cropped_input], training=True)
    X = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1), name="exp_dims_"+str(1))(X)
    X = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, [1, x.shape.as_list()[2]//2]), name="bilinear_interpol_down"+str(0))(X)
    X = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, [1, x.shape.as_list()[2]//2]), name="bilinear_interpol_down"+str(1))(X)
    X = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, [1, 44100]), name="bilinear_interpol_down"+str(2))(X)
    X = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=1), name="sq_dims_"+str(1))(X)

    o = X
    model = tf.keras.Model(inputs=raw_input, outputs=o)
    return model

"""# Other utility functions"""
def weighted_crossentropy(weight): 
    def compute_loss(y_true, y_pred):
        loss_object = tf.losses.CategoricalCrossentropy()
        loss_ = loss_object(y_true, y_pred, sample_weight=weight)
        return loss_
    return compute_loss

def load_model(file_path):
    return tf.keras.models.load_model(file_path, compile=False, custom_objects={'CropLayer' : CropLayer, 'IndependentOutputLayer' : IndependentOutputLayer,
                                                                                'InterpolationLayer' : InterpolationLayer, 'weighted_crossentropy' : weighted_crossentropy,
                                                                               'AudioClipLayer' : AudioClipLayer, 'DiffOutputLayer' : DiffOutputLayer, 'GLU': GLU, "ConformerBlock":ConformerBlock} )

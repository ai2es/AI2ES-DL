import tensorflow as tf
import numpy as np
from time import time

from tensorflow.keras.layers import Flatten, Conv2D, Dense, Input, Concatenate, Dropout, \
    BatchNormalization, DepthwiseConv2D, GlobalAveragePooling2D, Multiply, LayerNormalization, Add, \
    UpSampling2D, Lambda

from trainable.activations import hardswish


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = tf.stack((tf.sin(sin_inp), tf.cos(sin_inp)), -1)
    emb = tf.reshape(emb, (*emb.shape[:-2], -1))
    return emb


class TFPositionalEncoding2D(tf.keras.layers.Layer):
    def __init__(self, channels: int, dtype=tf.float32):
        """
        Args:
            channels int: The last dimension of the tensor you want to apply pos emb to.
        Keyword Args:
            dtype: output type of the encodings. Default is "tf.float32".
        """
        super(TFPositionalEncoding2D, self).__init__()

        self.channels = int(2 * np.ceil(channels / 4))
        self.inv_freq = np.float32(
            1
            / np.power(
                10000, np.arange(0, self.channels, 2) / np.float32(self.channels)
            )
        )
        self.cached_penc = None

    @tf.function
    def call(self, inputs):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(inputs.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == inputs.shape:
            return self.cached_penc

        self.cached_penc = None
        _, x, y, org_channels = inputs.shape

        dtype = self.inv_freq.dtype

        pos_x = tf.range(x, dtype=dtype)
        pos_y = tf.range(y, dtype=dtype)

        sin_inp_x = tf.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = tf.einsum("i,j->ij", pos_y, self.inv_freq)

        emb_x = tf.expand_dims(get_emb(sin_inp_x), 1)
        emb_y = tf.expand_dims(get_emb(sin_inp_y), 0)

        emb_x = tf.tile(emb_x, (1, y, 1))
        emb_y = tf.tile(emb_y, (x, 1, 1))
        emb = tf.concat((emb_x, emb_y), -1)
        self.cached_penc = tf.repeat(
            emb[None, :, :, :org_channels], tf.shape(inputs)[0], axis=0
        )
        return self.cached_penc


class PCA(tf.keras.layers.Layer):
    """
    Layer that iteratively updates a PCA estimate and transforms the output
    into the PC space

    inputs: a batch of n-D vectors
    outputs: a batch of 1-D vectors in PC coordinate space
    """

    def Q_init(self, shape, dtype=tf.float32, **kwargs):
        H = tf.random_normal_initializer(0.0, 1.0, 42)(shape, dtype=dtype)
        Q, _ = tf.linalg.qr(H, full_matrices=False)
        return Q

    def streaming_tf_PCA(self, X, p, k):
        # the examples are vectors of size p, want to deconstruct them into features k
        # this means H is p x k
        X = tf.cast(X, tf.float32)
        X, _ = tf.linalg.normalize(X, axis=-1)
        B = tf.shape(X)[0]
        B = tf.convert_to_tensor(float(B), dtype=tf.float32)
        S = tf.zeros_like(self.Q, dtype=tf.float32)
        XXT = tf.linalg.matmul(tf.transpose(X), X) / B
        S = S + tf.linalg.matmul(XXT, self.Q)

        Q, R = tf.linalg.qr(S, full_matrices=False, name='name')

        self.Q.assign(Q)

    def __init__(self, num_outputs):
        super(PCA, self).__init__()
        self.num_outputs = num_outputs
        self.Q = None

    def build(self, input_shape):
        self.Q = tf.Variable(self.Q_init((tf.reduce_prod(input_shape)[1:], self.num_outputs)), trainable=False)

    def call(self, inputs, training=None):
        input_shape = tf.shape(inputs)
        inputs = tf.reshape(inputs, (input_shape[0], tf.reduce_prod(input_shape[1:])))
        if training:
            self.streaming_tf_PCA(inputs, inputs.shape[-1], self.num_outputs)

        return tf.matmul(inputs, self.Q)

    def get_config(self):
        return {"Q": self.Q.numpy()}


class PCACompress(tf.keras.layers.Layer):
    """
    Layer that iteratively updates a PCA estimate and transforms the output
    into the PC space

    inputs: a batch of n-D vectors
    outputs: a batch of n-D vectors reconstructed from PCs
    """

    def Q_init(self, shape, dtype=tf.float32, **kwargs):
        H = tf.random_normal_initializer(0.0, 1.0, 42)(shape, dtype=dtype)
        Q, _ = tf.linalg.qr(H, full_matrices=False)
        return Q

    def streaming_tf_PCA(self, X, p, k):
        # the examples are vectors of size p, want to deconstruct them into features k
        # this means H is p x k
        X = tf.cast(X, tf.float32)
        X, _ = tf.linalg.normalize(X, axis=-1)
        B = tf.shape(X)[0]
        B = tf.convert_to_tensor(float(B), dtype=tf.float32)
        S = tf.zeros_like(self.Q, dtype=tf.float32)
        XXT = tf.linalg.matmul(tf.transpose(X), X) / B
        S = S + tf.linalg.matmul(XXT, self.Q)

        Q, R = tf.linalg.qr(S, full_matrices=False, name='name')

        self.Q.assign(Q)

    def __init__(self, num_outputs):
        super(PCA_Compress, self).__init__()
        self.num_outputs = num_outputs
        self.Q = None

    def build(self, input_shape):
        input_shape = tf.reduce_prod(input_shape[1:])
        self.Q = tf.Variable(self.Q_init((input_shape, self.num_outputs)), trainable=False)

    def call(self, inputs, training=None):
        input_shape = tf.shape(inputs)
        inputs = tf.reshape(inputs, (input_shape[0], tf.reduce_prod(input_shape[1:])))
        if training:
            self.streaming_tf_PCA(inputs, inputs.shape[-1], self.num_outputs)

        return tf.reshape(tf.linalg.matmul(tf.linalg.matmul(inputs, self.Q), tf.transpose(self.Q)), input_shape)

    def get_config(self):
        return {"Q": self.Q.numpy()}


def focal_module(units, focal_depth):
    """
    focal modulation block proposed by https://arxiv.org/pdf/2203.11926.pdf

    :param units: channel size of the output, number of channels in the convolutions
    :param focal_depth: number of convolutional layers in the hierarchal contextualzation
    :return: a keras layer
    """

    def depthwise_stack(x):
        outputs = []
        for i in range(focal_depth):
            x = DepthwiseConv2D(kernel_size=3, activation=hardswish, padding='same')(x)
            x = BatchNormalization()(x)
            outputs.append(x)
        x = GlobalAveragePooling2D(keepdims=True)(x)
        x = UpSampling2D(size=(outputs[-1].shape[1], outputs[-1].shape[2]))(x)
        outputs.append(x)

        return outputs

    def module(x):
        # layer normalization is essential for performance
        x = LayerNormalization()(x)
        q = Dense(units)(x)
        k = Dense(units)(x)
        # hard swish is almost GELU
        G = Dense(focal_depth + 1, activation=hardswish)(k)
        G = tf.expand_dims(G, -1)
        context = depthwise_stack(k)
        x = None
        for i, c in enumerate(context):
            y = Multiply()([G[:, :, :, i], c])
            if x is not None:
                x = Add()([x, y])
            else:
                x = y

        return tf.math.multiply(x, q, name=f"chkpt_{time()}")

    return module


def custom_focal_module(input_shape, units, focal_depth):
    """
    focal modulation block inspired by https://arxiv.org/pdf/2203.11926.pdf
    except in this version I make whatever changes I want to

    :param units: channel size of the output, number of channels in the convolutions
    :param focal_depth: number of convolutional layers in the hierarchal contextualzation
    :return: a keras layer
    """

    def depthwise_stack(x):
        outputs = []
        for i in range(focal_depth):
            # x = Conv2D(units, 1)(x)
            x = Conv2D(units, kernel_size=3, activation=hardswish, padding='same')(x)
            x = BatchNormalization()(x)
            outputs.append(x)
        x = GlobalAveragePooling2D(keepdims=True)(x)
        x = UpSampling2D(size=(outputs[-1].shape[1], outputs[-1].shape[2]))(x)
        outputs.append(x)

        return outputs

    def module(x):
        x = LayerNormalization()(x)
        q = Dense(units)(x)
        k = Dense(units)(x)
        G = Dense(focal_depth + 1, activation='sigmoid')(k)
        G = tf.expand_dims(G, -1)
        context = depthwise_stack(k)
        x = None
        for i, c in enumerate(context):
            y = Multiply()([G[:, :, :, i], c])
            if x is not None:
                x = Add()([x, y])
            else:
                x = y

        return Multiply(name=f"chkpt_{time()}")([x, q])

    inputs = Input(input_shape)
    outputs = module(inputs)

    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])

    model.compile()
    return model


def ConvNeXt_block(filters, l1=None, l2=None):
    """
    ConvNeXt block from https://arxiv.org/pdf/2201.03545.pdf

    :param filters: channels in the input and output of the convolution
    :param l1: l1 regularization weight
    :param l2: l2 regularization weight
    :return: a keras layer
    """

    def module(x):
        conv_params = {
            'use_bias': False,
            'kernel_initializer': tf.keras.initializers.GlorotUniform(),
            'bias_initializer': 'zeros',
            'kernel_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
            'bias_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
            'padding': 'same'
        }

        inputs = x

        x = DepthwiseConv2D(7, **conv_params)(inputs)

        x = LayerNormalization()(x)

        x = Conv2D(x.shape[-1] * 4, 1, activation=hardswish, **conv_params)(x)

        x = Conv2D(filters, 1, **conv_params)(x)

        outputs = inputs + x

        return outputs

    return module


def GRN(X, gamma, beta, p=2):
    gx = tf.norm(X, ord=p, axis=1, keepdims=True)
    gx = tf.norm(gx, ord=p, axis=2, keepdims=True)
    nx = gx / (tf.reduce_mean(gx, axis=-1, keepdims=True) + 1e-6)
    return gamma * (X * nx) + beta + X


class GlobalResponseNormalization(tf.keras.layers.Layer):
    """
    Global Response Normalization defined in https://arxiv.org/pdf/2301.00808.pdf

    inputs: a batch of n-D vectors
    outputs: a batch of 1-D vectors in PC coordinate space
    """

    def __init__(self, p=2):
        super(GlobalResponseNormalization, self).__init__()
        self.p = p
        self.gamma = None
        self.beta = None

    def build(self, input_shape):
        self.gamma = tf.Variable(.5, trainable=True)
        self.beta = tf.Variable(.5, trainable=True)

    def call(self, inputs, training=None):
        return GRN(inputs, self.gamma, self.beta, self.p)

    def get_config(self):
        return {"gamma": self.gamma.numpy(), "beta": self.beta.numpy()}


def ConvNeXtV2_block(filters, l1=None, l2=None):
    """
    ConvNeXtV2 block from https://arxiv.org/pdf/2301.00808.pdf

    :param filters: channels in the input and output of the convolution
    :param l1: l1 regularization weight
    :param l2: l2 regularization weight
    :return: a keras layer
    """

    def module(x):
        conv_params = {
            'use_bias': False,
            'kernel_initializer': tf.keras.initializers.GlorotUniform(),
            'bias_initializer': 'zeros',
            'kernel_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
            'bias_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
            'padding': 'same'
        }

        inputs = x

        x = DepthwiseConv2D(7, **conv_params)(x)

        x = LayerNormalization()(x)

        x = Conv2D(x.shape[-1] * 4, 1, activation=hardswish, **conv_params)(x)

        x = GlobalResponseNormalization()(x)

        x = Conv2D(filters, 1, **conv_params)(x)

        outputs = inputs + x

        return outputs

    return module


class LunchboxMHSA(tf.keras.layers.Layer):
    def __init__(self,
                 dim,
                 num_heads,
                 lunchbox_dim,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 prefix=''):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.lunchbox_dim = lunchbox_dim

        self.scale = qk_scale or max(dim, lunchbox_dim) ** -0.5
        self.prefix = prefix

        self.qkv = Dense(dim * 2 * num_heads, use_bias=qkv_bias,
                         name=f'{self.prefix}/attn/qkv')
        self.attn_drop = Dropout(attn_drop)
        self.proj = Dense(dim, name=f'{self.prefix}/attn/proj')
        self.proj_drop = Dropout(proj_drop)
        self.k = None
        self.built = False

    def build(self, input_shape):
        self.k = self.add_weight(f'{self.prefix}/attn/lunchbox',
                                 shape=(self.dim, self.lunchbox_dim),
                                 initializer=tf.initializers.GlorotUniform(), trainable=True)
        self.built = True

    def call(self, x):
        B_, N, C = x.get_shape().as_list()

        # x = tf.reshape(x, (B_, N, C))

        x = self.qkv(x)
        x = tf.transpose(x, perm=[0, 2, 1])

        x = tf.reshape(x, (-1, 2, self.num_heads, self.dim, N))
        qv = tf.transpose(x, perm=[1, 0, 2, 3, 4])

        q, v = qv[0], qv[1]

        attn = tf.einsum('bikj,kr->birk', q, self.k)
        attn = attn * self.scale
        tf.nn.softmax(attn, -1)

        res = tf.einsum('birk,bikn->bink', attn, v)

        x = tf.reshape(res, (-1, N, self.num_heads * self.dim))

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def get_config(self):
        return {"k": self.k.numpy()}

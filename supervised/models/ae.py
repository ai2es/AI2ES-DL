"""
model building functions should accept only float, int, or string arguments and must return only a compiled keras model
"""
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, Dense, Input, Concatenate, Dropout, SpatialDropout2D, \
    MultiHeadAttention, Add, BatchNormalization, LayerNormalization, Conv1D, Reshape, Cropping2D, ZeroPadding3D, \
    GlobalMaxPooling2D, GlobalAveragePooling2D, Lambda, Average, SeparableConv2D, DepthwiseConv2D, UpSampling2D, \
    Conv2DTranspose, AveragePooling2D, Multiply
from supervised.models.custom_layers import TFPositionalEncoding2D
HARDSWISH = lambda x: x * tf.nn.relu6(x + 3) / 6
from time import time
from supervised.models.custom_layers import int_act_no_slip


def clam_unet(conv_filters,
              image_size,
              l1=None,
              l2=None,
              activation=lambda x: x * tf.nn.relu6(x + 3) / 6,
              n_classes=10,
              depth=3,
              **kwargs):
    inputs = Input(image_size)

    conv_params = {
        'use_bias': False,
        'kernel_initializer': tf.keras.initializers.GlorotUniform(),
        'bias_initializer': 'zeros',
        'kernel_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        'bias_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        'padding': 'same'
    }

    x = inputs

    def thrifty_imb(z, down=True):
        inp = z
        if down:
            z = Conv2D(filters=max(inp.shape[-1] * 2, conv_filters), kernel_size=3, activation=activation,
                       **conv_params)(z)
            z = BatchNormalization()(z)
            return z
        else:
            z = Conv2D(filters=max(inp.shape[-1] // 2, conv_filters), kernel_size=3, activation=activation,
                       **conv_params)(z)
            z = BatchNormalization()(z)
            return z

    skips = [x]
    for i in range(depth):
        x = thrifty_imb(x)
        skips.append(x)
        x = MaxPooling2D(2)(x)

    for i in range(depth):
        x = thrifty_imb(x, down=False)
        x = UpSampling2D(interpolation="bilinear")(x)
        x = Concatenate()([x, skips.pop(-1)])
    # semantic segmentation output with extra (irrelevant) channel
    cam = Dense(n_classes + 1, activation='softmax')(x)
    # reduce sum over width / height
    x = Lambda(lambda z: tf.reduce_mean(z, axis=(1, 2)))(cam)
    # want to re-normalize without destroying the gradient
    # ideally would just divide by sum
    idk = x[:, -1]

    x = x[:, :-1] + 2 ** (-16)  # * (inputs.shape[1] * inputs.shape[2])

    out, _ = tf.linalg.normalize(x, 1, -1)

    # outputs shape is (batch, n_classes)

    model = tf.keras.Model(inputs=[inputs], outputs=[out, idk, cam],
                           name=f'clam')

    return model


def unet(conv_filters,
         image_size,
         l1=None,
         l2=None,
         activation=lambda x: x * tf.nn.relu6(x + 3) / 6,
         n_classes=10,
         depth=3,
         **kwargs):
    inputs = Input(image_size)

    conv_params = {
        'use_bias': False,
        'kernel_initializer': tf.keras.initializers.GlorotUniform(),
        'bias_initializer': 'zeros',
        'kernel_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        'bias_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        'padding': 'same'
    }

    x = inputs

    def thrifty_imb(z, down=True):
        inp = z
        if down:
            z = Conv2D(filters=max(inp.shape[-1] * 2, conv_filters), kernel_size=3, activation=activation,
                       **conv_params)(z)
            z = BatchNormalization()(z)
            return z
        else:
            z = Conv2D(filters=max(inp.shape[-1] // 2, conv_filters), kernel_size=3, activation=activation,
                       **conv_params)(z)
            z = BatchNormalization()(z)
            return z

    skips = [x]
    for i in range(depth):
        x = thrifty_imb(x)
        skips.append(x)
        x = MaxPooling2D(2)(x)

    x = thrifty_imb(x)

    for i in range(depth):
        x = thrifty_imb(x, down=False)
        x = UpSampling2D(interpolation="bilinear")(x)
        x = Add()([x, skips.pop(-1)])
    # semantic segmentation output with extra (irrelevant) channel
    cam = Dense(n_classes + 1, activation='softmax')(x)
    # reduce sum over width / height
    x = Lambda(lambda z: tf.reduce_mean(z, axis=(1, 2)))(cam)
    # want to re-normalize without destroying the gradient
    # ideally would just divide by sum
    idk = x[:, -1]

    x = x[:, :-1] + 2 ** (-16)  # * (inputs.shape[1] * inputs.shape[2])

    out, _ = tf.linalg.normalize(x, 1, -1)

    # outputs shape is (batch, n_classes)

    model = tf.keras.Model(inputs=[inputs], outputs=[out, idk, cam],
                           name=f'clam')

    return model


def transformer_unet(conv_filters,
                     image_size,
                     l1=None,
                     l2=None,
                     activation=lambda x: x * tf.nn.relu6(x + 3) / 6,
                     n_classes=10,
                     depth=3,
                     **kwargs):
    inputs = Input(image_size)

    conv_params = {
        'use_bias': False,
        'kernel_initializer': tf.keras.initializers.GlorotUniform(),
        'bias_initializer': 'zeros',
        'kernel_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        'bias_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        'padding': 'same'
    }

    x = inputs

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
                x = Conv2D(units, kernel_size=3, activation=HARDSWISH, padding='same')(x)
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
        
    def attention_block(z, heads=8):
        z = LayerNormalization()(z)
        input_shape = z.shape
        key_dim = value_dim = z.shape[-1]
        fft = tf.signal.fft(tf.cast(z, tf.complex64))

        real = tf.math.real(fft)
        imag = tf.math.imag(fft)

        z = Reshape((input_shape[1] * input_shape[2], input_shape[-1]))(z)
        real = Reshape((input_shape[1] * input_shape[2], input_shape[-1]))(real)
        imag = Reshape((input_shape[1] * input_shape[2], input_shape[-1]))(imag)

        real = LayerNormalization()(real)
        imag = LayerNormalization()(imag)
        # at the last layer of attention set the output to be a vector instead of a matrix
        z = MultiHeadAttention(heads,
                               key_dim,
                               value_dim,
                               kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2)
                               )(real, imag, z)

        z = Reshape((input_shape[1], input_shape[2], input_shape[-1]))(z)
        
        return z

    skips = [x]
    for i in range(depth):
        if x.shape[1] < 64:
            x = attention_block(x)
        else:
            skip = x
            x = custom_focal_module(x.shape[1:], x.shape[-1]*2, depth)(x)
            skip = Conv2D(x.shape[-1], 1)(skip)
            x = Add()([x, skip])
        skips.append(x)
        x = MaxPooling2D(2)(x)

    x = attention_block(x)

    for i in range(depth):
        if x.shape[1] < 64:
            x = attention_block(x)
        else:
            x = custom_focal_module(x.shape[1:], x.shape[-1] // 2, depth)(x)
        x = UpSampling2D(interpolation="bilinear")(x)
        x = Add()([x, skips.pop(-1)])
    # semantic segmentation output with extra (irrelevant) channel
    cam = Dense(n_classes + 1, activation='softmax')(x)
    # reduce sum over width / height
    x = Lambda(lambda z: tf.reduce_mean(z, axis=(1, 2)))(cam)
    # want to re-normalize without destroying the gradient
    # ideally would just divide by sum
    idk = x[:, -1]

    x = x[:, :-1] + 2 ** (-16)  # * (inputs.shape[1] * inputs.shape[2])

    out, _ = tf.linalg.normalize(x, 1, -1)

    # outputs shape is (batch, n_classes)

    model = tf.keras.Model(inputs=[inputs], outputs=[out, idk, cam],
                           name=f'clam')

    return model

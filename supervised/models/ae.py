"""
model building functions should accept only float, int, or string arguments and must return only a compiled keras model
"""
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, Dense, Input, Concatenate, Dropout, SpatialDropout2D, \
    MultiHeadAttention, Add, BatchNormalization, LayerNormalization, Conv1D, Reshape, Cropping2D, ZeroPadding3D, \
    GlobalMaxPooling2D, GlobalAveragePooling2D, Lambda, Average, SeparableConv2D, DepthwiseConv2D, UpSampling2D, \
    Conv2DTranspose, AveragePooling2D, Multiply


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
            z = Conv2D(filters=max(inp.shape[-1] * 2, conv_filters), kernel_size=3, activation=activation, **conv_params)(z)
            z = BatchNormalization()(z)
            return z
        else:
            z = Conv2D(filters=max(inp.shape[-1] // 2, conv_filters), kernel_size=3, activation=activation, **conv_params)(z)
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

    x = x[:, :-1] + 2**(-16)  # * (inputs.shape[1] * inputs.shape[2])

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
            z = Conv2D(filters=max(inp.shape[-1] * 2, conv_filters), kernel_size=3, activation=activation, **conv_params)(z)
            z = BatchNormalization()(z)
            return z
        else:
            z = Conv2D(filters=max(inp.shape[-1] // 2, conv_filters), kernel_size=3, activation=activation, **conv_params)(z)
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

    x = x[:, :-1] + 2**(-16)  # * (inputs.shape[1] * inputs.shape[2])

    out, _ = tf.linalg.normalize(x, 1, -1)

    # outputs shape is (batch, n_classes)

    model = tf.keras.Model(inputs=[inputs], outputs=[out, idk, cam],
                           name=f'clam')

    return model

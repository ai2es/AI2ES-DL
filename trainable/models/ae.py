"""Image to image models

model building functions should accept only float, int, or string arguments
and must return only a compiled keras model
"""
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, Dense, Input, Concatenate, Dropout, SpatialDropout2D, \
    MultiHeadAttention, Add, BatchNormalization, LayerNormalization, Conv1D, Reshape, Cropping2D, ZeroPadding3D, \
    GlobalMaxPooling2D, GlobalAveragePooling2D, Lambda, Average, SeparableConv2D, DepthwiseConv2D, UpSampling2D, \
    Conv2DTranspose, AveragePooling2D, Multiply, Activation
from trainable.custom_layers import TFPositionalEncoding2D

from trainable.custom_layers import VLunchboxMHSA, QLunchboxMHSA, DarkLunchboxMHSA

from time import time
from trainable.activations import hardswish


def clam_unet(conv_filters,
              image_size,
              l1=None,
              l2=None,
              activation=lambda x: x * tf.nn.relu6(x + 3) / 6,
              n_classes=10,
              depth=3,
              **kwargs):
    """
    LAX fully convolutional U-net.

    :param image_size: input image dimensions
    :param l1: l1 regularization coefficient
    :param l2: l2 regularization coefficient
    :param activation: activation function
    :param n_classes: number of channels in the output occlusion masks
    :param depth: depth of the 'U'
    :return: a keras model
    """
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
    """
    LAX fully convolutional U-net.

    :param conv_filters: convolutional filters in the first convolution
    :param image_size: input image dimensions
    :param l1: l1 regularization coefficient
    :param l2: l2 regularization coefficient
    :param activation: activation function
    :param n_classes: number of channels in the output occlusion masks
    :param depth: depth of the 'U'
    :return: a keras model
    """
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
                     n_classes=10,
                     depth=3,
                     **kwargs):
    """
    LAX fully transformer U-net.  Higher spatial resolution layers use Focal Modulation blocks, lower resolutions
    use Multi-headed self-attention

    :param conv_filters: convolutional filters in the first focal modulation block
    :param image_size: input image dimensions
    :param l1: l1 regularization coefficient
    :param l2: l2 regularization coefficient
    :param n_classes: number of channels in the output occlusion masks
    :param depth: depth of the 'U'
    :return: a keras model
    """

    inputs = Input(image_size)

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

    def attention_block(z, heads=8):
        z = LayerNormalization()(z)
        input_shape = z.shape
        key_dim = value_dim = z.shape[-1]

        z = Reshape((input_shape[1] * input_shape[2], input_shape[-1]))(z)

        # at the last layer of attention set the output to be a vector instead of a matrix
        z = MultiHeadAttention(heads,
                               key_dim,
                               value_dim,
                               kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2)
                               )(z, z, z)

        z = Reshape((input_shape[1], input_shape[2], input_shape[-1]))(z)

        return z

    inp = x
    skips = [x]
    for i in range(depth):
        if x.shape[1] < 32:
            x = attention_block(x)
        else:
            skip = x
            x = custom_focal_module(x.shape[1:], max(int(inp.shape[-1] * 1.5), conv_filters), depth)(x)
            skip = Conv2D(x.shape[-1], 1)(skip)
            x = Add()([x, skip])
        skips.append(x)
        x = MaxPooling2D(2)(x)

    x = attention_block(x)

    for i in range(depth):
        if x.shape[1] < 32:
            x = attention_block(x)
        else:
            x = custom_focal_module(x.shape[1:], max(inp.shape[-1] // 1.5, conv_filters), depth)(x)
        x = UpSampling2D(interpolation="bilinear")(x)
        x = Add()([x, skips.pop(-1)])
    x = SpatialDropout2D(.1)(x)
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


def vit_unet(conv_filters,
             image_size,
             l1=None,
             l2=None,
             n_classes=10,
             depth=3,
             **kwargs):
    """
    LAX transformer U-net.  Higher spatial resolution layers use convolution blocks, lower resolutions
    use Multi-headed self-attention

    :param conv_filters: convolutional filters in the first convolution block
    :param image_size: input image dimensions
    :param l1: l1 regularization coefficient
    :param l2: l2 regularization coefficient
    :param n_classes: number of channels in the output occlusion masks
    :param depth: depth of the 'U'
    :return: a keras model
    """

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

    def attention_block(z, heads=8):
        z = LayerNormalization()(z)
        input_shape = z.shape
        key_dim = value_dim = z.shape[-1]

        z = Reshape((input_shape[1] * input_shape[2], input_shape[-1]))(z)

        # at the last layer of attention set the output to be a vector instead of a matrix
        z = MultiHeadAttention(heads,
                               key_dim,
                               value_dim,
                               kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2)
                               )(z, z, z)

        z = Reshape((input_shape[1], input_shape[2], input_shape[-1]))(z)

        return z

    inp = x
    skips = [x]
    for i in range(depth):
        if x.shape[1] < 64:
            x = attention_block(x)
        else:
            skip = x
            x = Conv2D(max(int(inp.shape[-1] * 1.5), conv_filters), 3, **conv_params)(x)
            x = BatchNormalization()(x)
            x = Activation(hardswish)(x)

            skip = Conv2D(x.shape[-1], 1)(skip)
            x = Add()([x, skip])
        skips.append(x)
        x = MaxPooling2D(2)(x)

    x = attention_block(x)

    for i in range(depth):
        x = UpSampling2D(interpolation="bilinear")(x)
        if x.shape[1] < 64:
            x = attention_block(x)
        else:
            x = Conv2D(max(int(inp.shape[-1] * 1.5), conv_filters), 3, **conv_params)(x)
        x = Add()([x, skips.pop(-1)])
    x = SpatialDropout2D(.1)(x)
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


def lunchbox_packer(conv_filters,
                    image_size,
                    depth=3,
                    learning_rate=1e-3,
                    **kwargs):
    inputs = Input(image_size)

    x = inputs

    for i in range(depth):
        w, h, ch = x.shape[1], x.shape[2], x.shape[-1]
        x = LayerNormalization()(x)

        x = Reshape((w * h, ch))(x)
        x = QLunchboxMHSA(conv_filters, 4, (w * h) // 4)(x)
        x = Reshape((w // 2, h // 2, conv_filters))(x)

    w, h, ch = x.shape[1], x.shape[2], x.shape[-1]
    x = LayerNormalization()(x)
    x = Reshape((w * h, ch))(x)
    x = QLunchboxMHSA(conv_filters, 4, (w * h))(x)
    x = Reshape((w, h, conv_filters))(x)

    for i in range(depth)[::-1]:
        w, h, ch = x.shape[1], x.shape[2], x.shape[-1]
        x = LayerNormalization()(x)
        x = Reshape((w * h, ch))(x)
        x = QLunchboxMHSA(conv_filters, 4, (w * h) * 4)(x)
        x = Reshape((w * 2, h * 2, conv_filters))(x)

    outputs = Dense(3)(x)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs],
                           name=f'lunchbox_ae')

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)

    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    model.compile(loss='mae',
                  optimizer=opt)
    return model


def lunchbox_packerv2(conv_filters,
                      image_size,
                      depth=3,
                      learning_rate=1e-3,
                      **kwargs):
    inputs = Input(image_size)

    x = inputs

    # x = Conv2D(24, 2, 2, **conv_params)(x)
    w, h, ch = x.shape[1], x.shape[2], x.shape[-1]
    x = Reshape((w * h, ch))(x)

    for i in range(depth):
        x = MultiHeadAttention(4,
                               conv_filters,
                               conv_filters,
                               )(x, x, x)
        x = QLunchboxMHSA(conv_filters, 8, x.shape[1] // 2)(x)
        x = LayerNormalization()(x)

    x = MultiHeadAttention(4,
                           conv_filters,
                           conv_filters,
                           )(x, x, x)

    for i in range(depth)[::-1]:
        x = LayerNormalization()(x)
        x = MultiHeadAttention(4,
                               conv_filters,
                               conv_filters,
                               )(x, x, x)
        x = QLunchboxMHSA(conv_filters, 8, x.shape[1] * 2)(x)

    x = Dense(ch)(x)
    outputs = Reshape((w, h, ch))(x)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs],
                           name=f'lunchbox_ae')

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)

    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    model.compile(loss='mae',
                  optimizer=opt)
    return model


def ConvNeXt_unet(conv_filters,
                  image_size,
                  l1=None,
                  l2=None,
                  n_classes=10,
                  depth=3,
                  **kwargs):

    from trainable.custom_layers import ConvNeXt_block
    inputs = Input(image_size)
    x = inputs
    skips = []
    for i in range(depth):
        x = ConvNeXt_block(conv_filters, l1, l2)(x)
        skips.append(x)
        x = MaxPooling2D(2)(x)
        conv_filters *= 4 / 3
        conv_filters = int(conv_filters)

    x = ConvNeXt_block(conv_filters)(x)

    for i in range(depth):
        conv_filters *= 3 / 4
        conv_filters = int(conv_filters + .5)

        x = ConvNeXt_block(conv_filters, l1, l2)(x)
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

    model = tf.keras.Model(inputs=[inputs], outputs=[out, idk, cam],
                           name=f'clam')

    return model

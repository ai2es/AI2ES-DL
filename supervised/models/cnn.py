import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, Dense, Input, Concatenate, Dropout, SpatialDropout2D, \
    MultiHeadAttention, Add, BatchNormalization, LayerNormalization, Conv1D, Reshape, Cropping2D, ZeroPadding3D, \
    GlobalMaxPooling2D, GlobalAveragePooling2D, Lambda, Average, SeparableConv2D, DepthwiseConv2D, UpSampling2D, \
    Conv2DTranspose, AveragePooling2D, Multiply

from tensorflow.keras.applications import EfficientNetB0, ResNet50V2, VGG16, MobileNetV3Small
from time import time

"""
model building functions should accept only float, int, or string arguments and must return only a compiled keras model
"""

HARDSWISH = lambda x: x * tf.nn.relu6(x + 3) / 6
MISH = lambda x: x * tf.nn.tanh(tf.nn.softplus(x))


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
            x = DepthwiseConv2D(kernel_size=3, activation=HARDSWISH, padding='same')(x)
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
        G = Dense(focal_depth + 1, activation=HARDSWISH)(k)
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


def build_focal_modulator(image_size, n_classes, e_dim=24, learning_rate=1e-3, blocks=5, depth=3,
                          loss='categorical_crossentropy', **kwargs):
    """
    simple image classification network built with focal modules
    """
    inputs = Input(image_size)
    x = inputs
    # patch partitioning and embedding
    x = Conv2D(e_dim, kernel_size=4, strides=4)(x)
    for i in range(blocks):
        skip = x
        x = focal_module(x.shape[1:], 24 * (i + 1), depth)(x)
        skip = Conv2D(x.shape[-1], 1)(skip)
        x = Add()([x, skip])
        x = MaxPooling2D(2, 2)(x)

    x = Flatten()(x)

    outputs = Dense(n_classes, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)
    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    accuracy = 'sparse_categorical_accuracy' if loss == 'sparse_categorical_crossentropy' else 'categorical_accuracy'
    # compile the model
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[accuracy])

    return model


def build_focal_camnet(conv_filters,
                       conv_size,
                       dense_layers,
                       learning_rate,
                       image_size,
                       iterations=24,
                       loss='categorical_crossentropy',
                       l1=None, l2=None,
                       activation=lambda x: x * tf.nn.relu6(x + 3) / 6,
                       n_classes=10,
                       skips=2,
                       depth=4,
                       **kwargs):
    if isinstance(conv_filters, str):
        conv_filters = [int(i) for i in conv_filters.strip('[]').split(', ')]
    if isinstance(conv_size, str):
        conv_size = [int(i) for i in conv_size.strip('[]').split(', ')]
    if isinstance(dense_layers, str):
        dense_layers = [int(i) for i in dense_layers.strip('[]').split(', ')]

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

    x = Lambda(lambda z: tf.pad(z, ((0, 0), (0, 0), (0, 0), (0, conv_filters[0] - z.shape[-1]))))(x)

    for block in range(len(conv_filters)):

        thrifty_focus = custom_focal_module(x.shape[1:], conv_filters[block], depth)

        prev_layers = [thrifty_focus(x)]
        for i in range(iterations - 1):
            x = thrifty_focus(x)
            prev_layers.append(x)
            x = Add()(prev_layers[-skips:])
            x = BatchNormalization()(x)
        x = Add()(prev_layers)
    # this dense operation is over an input tensor of size (batch, width, height, channels)
    # semantic segmentation output with extra (irrelevant) channel
    x = Dense(n_classes + 1, activation='softmax', use_bias=False, name='cam')(x)
    # reduce sum over width / height
    y = Lambda(
        lambda z: tf.reduce_sum(
            tf.stack([z[:, :, :, -1] / (n_classes * 2) for i in range(n_classes)], -1), axis=(1, 2)
        )
    )(x)

    x = Lambda(lambda z: tf.reduce_sum(z[:, :, :, :-1], axis=(1, 2)))(x)
    x = Add()([x, y, tf.ones_like(x) * (2 ** (-10))])
    # want to re-normalize without destroying the gradient
    outputs = Lambda(lambda z: tf.linalg.normalize(z, 1, axis=-1)[0])(x)
    # outputs shape is (batch, n_classes)

    accuracy = 'sparse_categorical_accuracy' if loss == 'sparse_categorical_crossentropy' else 'categorical_accuracy'

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)

    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs],
                           name=f'thrifty_model_{"%02d" % time()}')

    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[accuracy])

    return model


def build_focal_camnetv2(conv_filters,
                         conv_size,
                         dense_layers,
                         learning_rate,
                         image_size,
                         iterations=24,
                         loss='categorical_crossentropy',
                         l1=None, l2=None,
                         activation=lambda x: x * tf.nn.relu6(x + 3) / 6,
                         n_classes=10,
                         skips=2,
                         depth=4,
                         **kwargs):
    if isinstance(conv_filters, str):
        conv_filters = [int(i) for i in conv_filters.strip('[]').split(', ')]
    if isinstance(conv_size, str):
        conv_size = [int(i) for i in conv_size.strip('[]').split(', ')]
    if isinstance(dense_layers, str):
        dense_layers = [int(i) for i in dense_layers.strip('[]').split(', ')]

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

    x = Lambda(lambda z: tf.pad(z, ((0, 0), (0, 0), (0, 0), (0, conv_filters[0] - z.shape[-1]))))(x)

    for block in range(len(conv_filters)):

        thrifty_focus = custom_focal_module(x.shape[1:], conv_filters[block], depth)

        prev_layers = [thrifty_focus(x)]
        for i in range(iterations - 1):
            x = thrifty_focus(x)
            prev_layers.append(x)
            x = Add()(prev_layers[-skips:])
            x = BatchNormalization()(x)
        x = Add()(prev_layers)
    # this dense operation is over an input tensor of size (batch, width, height, channels)
    # semantic segmentation output with extra (irrelevant) channel
    x = Dense(n_classes + 1, activation='softmax', use_bias=False, name='cam')(x)
    # reduce sum over width / height
    y = Lambda(
        lambda z: tf.reduce_sum(
            tf.stack([z[:, :, :, -1] for i in range(n_classes)], -1),
            axis=(1, 2)
        )
    )(x)

    y = tf.math.log(y + tf.ones_like(y))

    x = Lambda(lambda z: tf.reduce_sum(z[:, :, :, :-1], axis=(1, 2)))(x)
    x = Add()([x, y])
    # want to re-normalize without destroying the gradient
    outputs = Lambda(lambda z: tf.linalg.normalize(z, 1, axis=-1)[0])(x)
    # outputs shape is (batch, n_classes)

    accuracy = 'sparse_categorical_accuracy' if loss == 'sparse_categorical_crossentropy' else 'categorical_accuracy'

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)

    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs],
                           name=f'thrifty_model_{"%02d" % time()}')

    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[accuracy])

    return model


def build_keras_application(application, image_size=(256, 256, 3), learning_rate=1e-4, loss='categorical_crossentropy',
                            n_classes=10, dropout=0, **kwargs):
    inputs = Input(image_size)

    try:
        model = application(input_tensor=inputs, include_top=False, weights='imagenet', pooling='avg',
                            include_preprocessing=False)
    except TypeError as t:
        # no preprocessing for ResNet
        model = application(input_tensor=inputs, include_top=False, weights='imagenet', pooling='avg')

    outputs = Dense(n_classes, activation='softmax')(Dropout(dropout)(Flatten()(model.output)))

    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)
    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    accuracy = 'sparse_categorical_accuracy' if loss == 'sparse_categorical_crossentropy' else 'categorical_accuracy'
    # compile the model
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[accuracy])

    return model


def build_special_VGG(image_size, classes=(3, 4, 5), learning_rate=1e-3, loss='categorical_crossentropy'):
    inputs = Input(image_size)
    # load VGG16 with imagenet weights
    model = VGG16(input_tensor=inputs, include_top=False, weights='imagenet', pooling='avg',
                  include_preprocessing=False)

    outputs = []
    for n_classes in classes:
        outputs.append(Dense(n_classes, activation='softmax')(Flatten()(model.output)))
    # 'outputs' can take a list of output tensors
    model = tf.keras.models.Model(inputs=[inputs], outputs=outputs)
    # you can ignore these next two statements
    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)
    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
    # select the correct kind of accuracy
    accuracy = 'sparse_categorical_accuracy' if loss == 'sparse_categorical_crossentropy' else 'categorical_accuracy'
    # compile the model
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[accuracy])
    # return the compiled model
    return model


def build_EfficientNetB0(**kwargs):
    return build_keras_application(EfficientNetB0, **kwargs)


def build_ResNet50V2(**kwargs):
    return build_keras_application(ResNet50V2, **kwargs)


def build_MobileNetV3Small(**kwargs):
    return build_keras_application(MobileNetV3Small, **kwargs)


def build_camnet(conv_filters,
                 conv_size,
                 dense_layers,
                 learning_rate,
                 image_size,
                 iterations=24,
                 loss='categorical_crossentropy',
                 l1=None, l2=None,
                 activation=lambda x: x * tf.nn.relu6(x + 3) / 6,
                 n_classes=10,
                 skips=2,
                 **kwargs):
    if isinstance(conv_filters, str):
        conv_filters = [int(i) for i in conv_filters.strip('[]').split(', ')]
    if isinstance(conv_size, str):
        conv_size = [int(i) for i in conv_size.strip('[]').split(', ')]
    if isinstance(dense_layers, str):
        dense_layers = [int(i) for i in dense_layers.strip('[]').split(', ')]

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

    x = Lambda(lambda z: tf.pad(z, ((0, 0), (0, 0), (0, 0), (0, conv_filters[0] - z.shape[-1]))))(x)

    for block in range(len(conv_filters)):
        thrifty_exp = Conv2D(filters=conv_filters[block], kernel_size=1, activation=None, **conv_params)

        thrifty_convs = [
            Conv2D(conv_filters[block], kernel_size=3, activation=activation, **conv_params),
            Conv2D(conv_filters[block], kernel_size=3, activation=None, **conv_params)
        ]

        thrifty_inv3 = UpSampling2D(2)

        thrifty_squeeze = Conv2D(filters=conv_filters[block], kernel_size=1, activation=activation, **conv_params)

        def thrifty_imb(z, output_dim):
            exp, exc, inv, sqzn = thrifty_exp, thrifty_convs, thrifty_inv3, thrifty_squeeze
            inp = z
            z = MaxPooling2D(2, 2, 'same')(z)
            z = Concatenate()([inv(e(z)) for e in exc])
            z = Conv2D(filters=output_dim // 2, kernel_size=1, activation=None, **conv_params)(z)
            z = activation(z)
            z = BatchNormalization()(z)
            z0 = Concatenate()([e(inp) for e in exc] + [inp])
            z0 = Conv2D(filters=output_dim // 2, kernel_size=1, activation=None, **conv_params)(z0)
            z0 = activation(z0)
            z = Concatenate()([z0, z])
            z = BatchNormalization(name=f"chkpt_{time()}")(z)
            return z

        prev_layers = [thrifty_imb(x, conv_filters[block])]
        for i in range(iterations):
            x = thrifty_imb(x, conv_filters[block])
            prev_layers.append(x)
            x = Add()(prev_layers[-skips:])
            x = BatchNormalization()(x)
        x = Add()(prev_layers)
    # this dense operation is over an input tensor of size (batch, width, height, channels)
    # semantic segmentation output with extra (irrelevant) channel
    x = Dense(n_classes + 1, activation='softmax', use_bias=False)(x)
    # reduce sum over width / height
    y = Lambda(
        lambda z: tf.reduce_sum(
            tf.stack([z[:, :, :, -1] / (n_classes * 2) for i in range(n_classes)], -1), axis=(1, 2)
        )
    )(x)

    x = Lambda(lambda z: tf.reduce_sum(z[:, :, :, :-1], axis=(1, 2)))(x)
    x = Add()([x, y, tf.ones_like(x) * (2 ** (-10))])
    # want to re-normalize without destroying the gradient
    outputs = Lambda(lambda z: tf.linalg.normalize(z, 1, axis=-1)[0])(x)
    # outputs shape is (batch, n_classes)

    accuracy = 'sparse_categorical_accuracy' if loss == 'sparse_categorical_crossentropy' else 'categorical_accuracy'

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)

    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs],
                           name=f'thrifty_model_{"%02d" % time()}')

    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[accuracy])

    return model


def build_camnetv2(conv_filters,
                   conv_size,
                   dense_layers,
                   learning_rate,
                   image_size,
                   iterations=24,
                   loss='categorical_crossentropy',
                   l1=None, l2=None,
                   activation=lambda x: x * tf.nn.relu6(x + 3) / 6,
                   n_classes=10,
                   skips=2,
                   **kwargs):
    if isinstance(conv_filters, str):
        conv_filters = [int(i) for i in conv_filters.strip('[]').split(', ')]
    if isinstance(conv_size, str):
        conv_size = [int(i) for i in conv_size.strip('[]').split(', ')]
    if isinstance(dense_layers, str):
        dense_layers = [int(i) for i in dense_layers.strip('[]').split(', ')]

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

    x = Lambda(lambda z: tf.pad(z, ((0, 0), (0, 0), (0, 0), (0, conv_filters[0] - z.shape[-1]))))(x)

    for block in range(len(conv_filters)):
        thrifty_exp = Conv2D(filters=conv_filters[block], kernel_size=1, activation=None, **conv_params)

        thrifty_convs = [
            Conv2D(conv_filters[block] // 4, kernel_size=3, activation=activation, **conv_params),
            Conv2D(conv_filters[block] // 4, kernel_size=3, activation=None, **conv_params)
        ]

        thrifty_inv3 = UpSampling2D(2)

        # thrifty_squeeze = Conv2D(filters=conv_filters[block], kernel_size=1, activation=activation, **conv_params)

        def thrifty_imb(z):
            exp, exc, inv = thrifty_exp, thrifty_convs, thrifty_inv3
            inp = z
            z = MaxPooling2D(2, 2, 'same')(z)
            z = Concatenate()([inv(e(z)) for e in exc])
            z0 = Concatenate()([e(inp) for e in exc])
            z = Concatenate()([z0, z, tf.reduce_max(x, axis=-1, keepdims=True),
                               tf.reduce_min(x, axis=-1, keepdims=True)])
            z = BatchNormalization(name=f"chkpt_{time()}")(z)
            return z

        x = Concatenate(axis=-1)([x, tf.reduce_max(x, axis=-1, keepdims=True),
                                  tf.reduce_min(x, axis=-1, keepdims=True)])

        prev_layers = [thrifty_imb(x)]
        for i in range(iterations):
            x = thrifty_imb(x)
            prev_layers.append(x)
            x = Add()(prev_layers[-skips:])
            x = BatchNormalization()(x)
        x = Add()(prev_layers)
    # this dense operation is over an input tensor of size (batch, width, height, channels)
    # semantic segmentation output with extra (irrelevant) channel
    x = Dense(n_classes + 1, activation='softmax', use_bias=False)(x)
    # reduce sum over width / height
    y = Lambda(
        lambda z: tf.reduce_sum(
            tf.stack([z[:, :, :, -1] / (n_classes * 2) for i in range(n_classes)], -1), axis=(1, 2)
        )
    )(x)

    x = Lambda(lambda z: tf.reduce_sum(z[:, :, :, :-1], axis=(1, 2)))(x)
    x = Add()([x, y])

    outputs = tf.nn.softmax(x)
    # want to re-normalize without destroying the gradient
    # outputs = Lambda(lambda z: tf.linalg.normalize(z, 1, axis=-1)[0])(x)
    # outputs shape is (batch, n_classes)

    accuracy = 'sparse_categorical_accuracy' if loss == 'sparse_categorical_crossentropy' else 'categorical_accuracy'

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)

    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs],
                           name=f'thrifty_model_{"%02d" % time()}')

    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[accuracy])

    return model


def build_camnet_reordered(conv_filters,
                           conv_size,
                           dense_layers,
                           learning_rate,
                           image_size,
                           iterations=24,
                           loss='categorical_crossentropy',
                           l1=None, l2=None,
                           activation=lambda x: x * tf.nn.relu6(x + 3) / 6,
                           n_classes=10,
                           skips=2,
                           **kwargs):
    if isinstance(conv_filters, str):
        conv_filters = [int(i) for i in conv_filters.strip('[]').split(', ')]
    if isinstance(conv_size, str):
        conv_size = [int(i) for i in conv_size.strip('[]').split(', ')]
    if isinstance(dense_layers, str):
        dense_layers = [int(i) for i in dense_layers.strip('[]').split(', ')]

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

    x = Lambda(lambda z: tf.pad(z, ((0, 0), (0, 0), (0, 0), (0, conv_filters[0] - z.shape[-1]))))(x)

    for block in range(len(conv_filters)):
        thrifty_exp = Conv2D(filters=conv_filters[block], kernel_size=1, activation=None, **conv_params)

        thrifty_convs = [
            Conv2D(conv_filters[block], kernel_size=3, activation=activation, **conv_params),
            Conv2D(conv_filters[block], kernel_size=3, activation=None, **conv_params)
        ]

        thrifty_inv3 = UpSampling2D(2)

        def thrifty_imb(z, output_dim):
            exp, exc, inv = thrifty_exp, thrifty_convs, thrifty_inv3
            inp = z
            z = MaxPooling2D(2, 2, 'same')(z)
            z = Concatenate()([inv(e(z)) for e in exc])
            z = Conv2D(filters=output_dim // 2, kernel_size=1, activation=None, **conv_params)(z)
            z0 = Concatenate()([e(inp) for e in exc] + [inp])
            z0 = Conv2D(filters=output_dim // 2, kernel_size=1, activation=None, **conv_params)(z0)
            z = BatchNormalization()(z)
            z = activation(z)
            z = Concatenate(name=f"chkpt_{time()}")([z0, z])
            return z

        prev_layers = [thrifty_imb(x, conv_filters[block])]
        for i in range(iterations):
            x = thrifty_imb(x, conv_filters[block])
            prev_layers.append(x)
            x = Add()(prev_layers[-skips:])
            x = BatchNormalization()(x)
        x = Add()(prev_layers)
    # this dense operation is over an input tensor of size (batch, width, height, channels)
    # semantic segmentation output with extra (irrelevant) channel
    x = Dense(n_classes + 1, activation='softmax', use_bias=False)(x)
    # reduce sum over width / height
    y = Lambda(
        lambda z: tf.reduce_sum(
            tf.stack([z[:, :, :, -1] / (n_classes * 2) for i in range(n_classes)], -1), axis=(1, 2)
        )
    )(x)

    x = Lambda(lambda z: tf.reduce_sum(z[:, :, :, :-1], axis=(1, 2)))(x)
    x = Add()([x, y, tf.ones_like(x) * (2 ** (-10))])
    # want to re-normalize without destroying the gradient
    outputs = Lambda(lambda z: tf.linalg.normalize(z, 1, axis=-1)[0])(x)
    # outputs shape is (batch, n_classes)

    accuracy = 'sparse_categorical_accuracy' if loss == 'sparse_categorical_crossentropy' else 'categorical_accuracy'

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)

    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs],
                           name=f'thrifty_model_{"%02d" % time()}')

    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[accuracy])

    return model


def build_camnet_reorderedv2(conv_filters,
                           conv_size,
                           dense_layers,
                           learning_rate,
                           image_size,
                           iterations=24,
                           loss='categorical_crossentropy',
                           l1=None, l2=None,
                           activation=lambda x: x * tf.nn.relu6(x + 3) / 6,
                           n_classes=10,
                           skips=2,
                           **kwargs):
    if isinstance(conv_filters, str):
        conv_filters = [int(i) for i in conv_filters.strip('[]').split(', ')]
    if isinstance(conv_size, str):
        conv_size = [int(i) for i in conv_size.strip('[]').split(', ')]
    if isinstance(dense_layers, str):
        dense_layers = [int(i) for i in dense_layers.strip('[]').split(', ')]

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

    x = Lambda(lambda z: tf.pad(z, ((0, 0), (0, 0), (0, 0), (0, conv_filters[0] - z.shape[-1]))))(x)

    for block in range(len(conv_filters)):
        thrifty_exp = Conv2D(filters=conv_filters[block], kernel_size=1, activation=None, **conv_params)

        thrifty_convs = [
            Conv2D(conv_filters[block], kernel_size=3, activation=activation, **conv_params),
            Conv2D(conv_filters[block], kernel_size=3, activation=None, **conv_params)
        ]

        thrifty_inv3 = UpSampling2D(2)

        def thrifty_imb(z, output_dim):
            exp, exc, inv = thrifty_exp, thrifty_convs, thrifty_inv3
            inp = z
            z = MaxPooling2D(2, 2, 'same')(z)
            z = Concatenate()([inv(e(z)) for e in exc])
            z = Conv2D(filters=output_dim // 2, kernel_size=1, activation=None, **conv_params)(z)
            z0 = Concatenate()([e(inp) for e in exc] + [inp])
            z0 = Conv2D(filters=output_dim // 2, kernel_size=1, activation=None, **conv_params)(z0)
            z = BatchNormalization()(z)
            z = activation(z)
            z = Concatenate(name=f"chkpt_{time()}")([z0, z])
            return z

        prev_layers = [thrifty_imb(x, conv_filters[block])]
        for i in range(iterations):
            x = thrifty_imb(x, conv_filters[block])
            prev_layers.append(x)
            x = Add()(prev_layers[-skips:])
            x = BatchNormalization()(x)
        x = Add()(prev_layers)
    # this dense operation is over an input tensor of size (batch, width, height, channels)
    # semantic segmentation output with extra (irrelevant) channel
    x = Dense(n_classes + 1, activation='softmax', use_bias=False, name='cam')(x)
    # reduce sum over width / height
    y = Lambda(
        lambda z: tf.reduce_sum(
            tf.stack([z[:, :, :, -1] for i in range(n_classes)], -1),
            axis=(1, 2)
        )
    )(x)

    x = Lambda(lambda z: tf.reduce_sum(z[:, :, :, :-1], axis=(1, 2)))(x)
    x = x + (tf.math.log(y + tf.ones_like(y)) / n_classes)
    # want to re-normalize without destroying the gradient
    outputs = Lambda(lambda z: tf.linalg.normalize(z, 1, axis=-1)[0])(x)
    # outputs shape is (batch, n_classes)

    accuracy = 'sparse_categorical_accuracy' if loss == 'sparse_categorical_crossentropy' else 'categorical_accuracy'

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)

    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs],
                           name=f'thrifty_model_{"%02d" % time()}')

    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[accuracy])

    return model


def build_camnet_reorderedv3(conv_filters,
                           conv_size,
                           dense_layers,
                           learning_rate,
                           image_size,
                           iterations=24,
                           loss='categorical_crossentropy',
                           l1=None, l2=None,
                           activation=lambda x: x * tf.nn.relu6(x + 3) / 6,
                           n_classes=10,
                           skips=2,
                           **kwargs):
    if isinstance(conv_filters, str):
        conv_filters = [int(i) for i in conv_filters.strip('[]').split(', ')]
    if isinstance(conv_size, str):
        conv_size = [int(i) for i in conv_size.strip('[]').split(', ')]
    if isinstance(dense_layers, str):
        dense_layers = [int(i) for i in dense_layers.strip('[]').split(', ')]

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

    x = Lambda(lambda z: tf.pad(z, ((0, 0), (0, 0), (0, 0), (0, conv_filters[0] - z.shape[-1]))))(x)

    for block in range(len(conv_filters)):
        thrifty_exp = Conv2D(filters=conv_filters[block], kernel_size=1, activation=None, **conv_params)

        thrifty_convs = [
            Conv2D(conv_filters[block], kernel_size=3, activation=activation, **conv_params),
            Conv2D(conv_filters[block], kernel_size=3, activation=None, **conv_params)
        ]

        thrifty_inv3 = UpSampling2D(2)

        def thrifty_imb(z, output_dim):
            exp, exc, inv = thrifty_exp, thrifty_convs, thrifty_inv3
            inp = z
            z = MaxPooling2D(2, 2, 'same')(z)
            z = Concatenate()([inv(e(z)) for e in exc])
            z = Conv2D(filters=output_dim // 2, kernel_size=1, activation=None, **conv_params)(z)
            z0 = Concatenate()([e(inp) for e in exc] + [inp])
            z0 = Conv2D(filters=output_dim // 2, kernel_size=1, activation=None, **conv_params)(z0)
            z = BatchNormalization()(z)
            z = activation(z)
            z = Concatenate(name=f"chkpt_{time()}")([z0, z])
            return z

        prev_layers = [thrifty_imb(x, conv_filters[block])]
        for i in range(iterations):
            x = thrifty_imb(x, conv_filters[block])
            prev_layers.append(x)
            x = Add()(prev_layers[-skips:])
            x = BatchNormalization()(x)
        x = Add()(prev_layers)
    # this dense operation is over an input tensor of size (batch, width, height, channels)
    # semantic segmentation output with extra (irrelevant) channel
    x = Dense(n_classes + 1, activation='softmax', use_bias=False, name='cam')(x)
    # reduce sum over width / height
    y = Lambda(
        lambda z: tf.reduce_mean(
            tf.stack([z[:, :, :, -1] for i in range(n_classes)], -1),
            axis=(1, 2)
        )
    )(x)

    x = Lambda(lambda z: tf.reduce_mean(z[:, :, :, :-1], axis=(1, 2)))(x)
    eps = 2**(-64)
    x = y / tf.math.negative(tf.math.log(((1 + eps)*tf.ones_like(x) - x)))
    # want to re-normalize without destroying the gradient
    outputs = Lambda(lambda z: tf.linalg.normalize(z, 1, axis=-1)[0])(x)
    # outputs shape is (batch, n_classes)

    accuracy = 'sparse_categorical_accuracy' if loss == 'sparse_categorical_crossentropy' else 'categorical_accuracy'

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)

    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs],
                           name=f'thrifty_model_{"%02d" % time()}')

    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[accuracy])

    return model


def build_basic_cnn(conv_filters,
                    conv_size,
                    dense_layers,
                    learning_rate,
                    image_size,
                    iterations=24,
                    loss='categorical_crossentropy',
                    l1=None, l2=None,
                    activation=lambda x: x * tf.nn.relu6(x + 3) / 6,
                    n_classes=10,
                    skips=2,
                    **kwargs):
    if isinstance(conv_filters, str):
        conv_filters = [int(i) for i in conv_filters.strip('[]').split(', ')]
    if isinstance(conv_size, str):
        conv_size = [int(i) for i in conv_size.strip('[]').split(', ')]
    if isinstance(dense_layers, str):
        dense_layers = [int(i) for i in dense_layers.strip('[]').split(', ')]

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

    for block in range(len(conv_filters)):
        def basic_cnn_block(z):
            inp = z
            z = Conv2D(conv_filters[block], kernel_size=3, activation=None, **conv_params)(z)
            z = Conv2D(conv_filters[block] * 2, kernel_size=5, activation=None, **conv_params)(z)
            z = BatchNormalization()(z)
            z = activation(z)
            z = MaxPooling2D(2, 2)(z)
            z = Concatenate()([z, MaxPooling2D(2, 2)(inp)])
            return z

        x = basic_cnn_block(x)

    # this dense operation is over an input tensor of size (batch, width, height, channels)
    # semantic segmentation output with extra (irrelevant) channel
    x = GlobalMaxPooling2D()(x)

    for units in dense_layers:
        x = Dense(units, activation=activation, use_bias=False)(x)

    outputs = Dense(n_classes, activation='softmax', use_bias=False)(x)

    accuracy = 'sparse_categorical_accuracy' if loss == 'sparse_categorical_crossentropy' else 'categorical_accuracy'

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)

    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs],
                           name=f'thrifty_model_{"%02d" % time()}')

    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[accuracy])

    return model

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, Dense, Input, Concatenate, Dropout, SpatialDropout2D, \
    MultiHeadAttention, Add, BatchNormalization, LayerNormalization, Conv1D, Reshape, Cropping2D, ZeroPadding3D, \
    GlobalMaxPooling2D, GlobalAveragePooling2D, Lambda, Average, SeparableConv2D, DepthwiseConv2D, UpSampling2D, \
    Conv2DTranspose, AveragePooling2D, Multiply

import keras.backend as K

from tensorflow.keras.applications import EfficientNetB0, ResNet50V2, VGG16, MobileNetV3Small
from time import time

from supervised.models.ae import clam_unet, unet, transformer_unet, vit_unet
from supervised.models.custom_layers import TFPositionalEncoding2D

"""
model building functions should accept only float, int, or string arguments and must return only a compiled keras model
"""


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
        x = thrifty_focus(x)
        x = Lambda(lambda z: z, name=f"chkpt_{time()}")(x)
        prev_layers = [x]
        for i in range(iterations - 1):
            x = thrifty_focus(x)
            x = Lambda(lambda z: z, name=f"chkpt_{time()}")(x)
            prev_layers.append(x)
            x = Add()(prev_layers[-skips:])
            x = BatchNormalization()(x)
        if len(prev_layers) > 1:
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
    x = x + (tf.math.sqrt(y) / n_classes)
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
    def jay_loss(y_true, y_pred):
        indicies = tf.argmax(y_true, axis=-1)
        nd_indicies = tf.stack([tf.range(tf.shape(indicies)[0], dtype=indicies.dtype), indicies], -1)

        # optionally we can relu the following
        return y_pred[:, -1] ** 2 / tf.gather_nd(y_pred[:, :-1], nd_indicies)

    def CE(y_true, y_pred):
        return tf.math.negative(tf.reduce_sum(tf.math.log(y_pred) * y_true, axis=-1))

    def NCE(y_true, y_pred):
        return tf.math.negative(tf.math.log(1 - tf.reduce_max(y_pred, axis=-1)))

    def mask_loss(y_true, y_pred):
        return 0 * tf.math.negative(tf.math.log(1 - tf.reduce_max(y_pred, axis=-1)))

    if isinstance(conv_filters, str):
        conv_filters = [int(i) for i in conv_filters.strip('[]').split(', ')]
    if isinstance(conv_size, str):
        conv_size = [int(i) for i in conv_size.strip('[]').split(', ')]
    if isinstance(dense_layers, str):
        dense_layers = [int(i) for i in dense_layers.strip('[]').split(', ')]

    conv_params = {
        'use_bias': False,
        'kernel_initializer': tf.keras.initializers.GlorotUniform(),
        'bias_initializer': 'zeros',
        'kernel_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        'bias_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        'padding': 'same'
    }

    def base_model():
        inputs = Input(image_size)

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
        cam = Dense(n_classes + 1, activation='softmax', use_bias=False, name='cam')(x)

        x = Lambda(lambda z: tf.reduce_sum(z, axis=(1, 2)))(cam)

        idk = x[:, -1]

        x = x[:, :-1] + 2 ** (-16)

        out, _ = tf.linalg.normalize(x, 1, -1)

        # outputs shape is (batch, n_classes)

        model = tf.keras.Model(inputs=[inputs], outputs=[out, idk, cam],
                               name=f'clam')

        return model

    # in the masker we replace the masked pixels with the mean of the input tensor plus some noise

    def masker(image_size, cam_size, pred_size):
        image_inputs = Input(image_size)
        cam_inputs = Input(cam_size)
        pred_inputs = Input(pred_size)

        i = tf.argmax(pred_inputs[:, :-1], axis=-1)
        print(cam_inputs.shape)
        m = tf.gather(cam_inputs, i, axis=-1, batch_dims=1)
        # m = tf.slice(layer.output, i, axis=-1)
        m = tf.expand_dims(m, axis=-1)

        # mask out the class relevant pixels
        x = image_inputs * (tf.ones_like(m) - m)
        # replace with noise that preserves the mean

        masker_outputs = x + (m * tf.reduce_mean(image_inputs, axis=0))

        model = tf.keras.Model(inputs=[image_inputs, cam_inputs, pred_inputs], outputs=[masker_outputs],
                               name=f'masker')

        return model

    model_inputs = Input(image_size)

    base = base_model()
    # base = clam_unet(12, image_size, n_classes=n_classes, depth=5)

    mask = masker(image_size, base.outputs[-1].shape[1:], base.outputs[0].shape[1:])

    base_out, idk, cam = base(model_inputs)

    masked_inputs = mask([model_inputs, cam, base_out])

    masked_inputs = Lambda(lambda z: K.stop_gradient(z))(masked_inputs)

    masked_out, _, _ = base(masked_inputs)

    mask_size = tf.reduce_max(base_out, axis=-1, keepdims=True) / (
            tf.reduce_sum(base_out, axis=-1, keepdims=True) + idk)

    outputs = [base_out, masked_out, mask_size]

    model = tf.keras.Model(inputs=[model_inputs], outputs=outputs,
                           name=f'clam_masker')

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)

    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    model.compile(loss=[CE, NCE, mask_loss],
                  loss_weights=[1, 1, 1],
                  optimizer=opt,
                  metrics=['categorical_accuracy'])

    return model


def build_camnet_reorderedv4(conv_filters,
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
                             alpha=1e-3,
                             beta=1e-3,
                             **kwargs):
    def CE(y_true, y_pred):
        return tf.math.negative(tf.reduce_sum(tf.math.log(y_pred) * y_true, axis=-1))

    def NCE(y_true, y_pred):
        return tf.math.negative(
            tf.math.log(tf.reduce_mean(y_pred, axis=-1)) - tf.math.log(tf.reduce_max(y_pred, axis=-1)))

    def mask_loss(y_true, y_pred):
        return tf.math.negative(tf.math.log(1 + 2 ** (-16) - tf.reduce_sum(y_pred, keepdims=True, axis=-1)))

    if isinstance(conv_filters, str):
        conv_filters = [int(i) for i in conv_filters.strip('[]').split(', ')]
    if isinstance(conv_size, str):
        conv_size = [int(i) for i in conv_size.strip('[]').split(', ')]
    if isinstance(dense_layers, str):
        dense_layers = [int(i) for i in dense_layers.strip('[]').split(', ')]

    conv_params = {
        'use_bias': False,
        'kernel_initializer': tf.keras.initializers.GlorotUniform(),
        'bias_initializer': 'zeros',
        'kernel_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        'bias_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        'padding': 'same'
    }

    # in the masker we replace the masked pixels with the mean of the input tensor plus some noise

    def mask_plurality(image_size, cam_size, pred_size):
        image_inputs = Input(image_size)
        cam_inputs = Input(cam_size)
        pred_inputs = Input(pred_size)

        i = tf.argmax(pred_inputs[:, :-1], axis=-1)
        print(cam_inputs.shape)
        m = tf.gather(cam_inputs, i, axis=-1, batch_dims=1)
        # m = tf.slice(layer.output, i, axis=-1)
        m = tf.expand_dims(m, axis=-1)

        # mask out the class relevant pixels
        z = image_inputs * (tf.ones_like(m) - m)
        # replace with noise that preserves the mean

        masker_outputs = z + (m * tf.reduce_mean(image_inputs, axis=0))

        model = tf.keras.Model(inputs=[image_inputs, cam_inputs, pred_inputs], outputs=[masker_outputs],
                               name=f'plurality_masker')

        return model

    def mask_total(image_size, cam_size, pred_size):
        image_inputs = Input(image_size)
        cam_inputs = Input(cam_size)
        pred_inputs = Input(pred_size)

        m = tf.reduce_sum(cam_inputs[:, :, :, :-1], keepdims=True, axis=-1)

        # mask out the class relevant pixels
        z = image_inputs * (tf.ones_like(m) - m)
        # replace with noise that preserves the mean

        masker_outputs = z + (m * tf.reduce_mean(image_inputs, axis=0))

        model = tf.keras.Model(inputs=[image_inputs, cam_inputs, pred_inputs], outputs=[masker_outputs],
                               name=f'all_masker')

        return model

    model_inputs = Input(image_size)

    base = unet(12, image_size, n_classes=n_classes, depth=4)

    masked_pred = mask_plurality(image_size, base.outputs[-1].shape[1:], base.outputs[0].shape[1:])
    masked_all = mask_total(image_size, base.outputs[-1].shape[1:], base.outputs[0].shape[1:])

    base_out, idk, cam = base(model_inputs)

    masked_inputs_pred = masked_pred([model_inputs, cam, base_out])
    masked_inputs_all = masked_all([model_inputs, cam, base_out])

    # masked_inputs_pred = Lambda(lambda z: K.stop_gradient(z))(masked_inputs_pred)
    # masked_inputs_all = Lambda(lambda z: K.stop_gradient(z))(masked_inputs_all)

    masked_pred_out, _, _ = base(masked_inputs_pred)
    masked_all_out, _, _ = base(masked_inputs_all)

    normed_base_out, _ = tf.linalg.normalize(base_out, axis=-1)
    normed_masked_pred_out, _ = tf.linalg.normalize(masked_pred_out, axis=-1)

    masked_pred_out = normed_base_out * normed_masked_pred_out

    outputs = {'crossentropy': base_out, 'cosine': masked_pred_out, 'all_masked': masked_all_out}

    model = tf.keras.Model(inputs=[model_inputs], outputs=outputs,
                           name=f'clam_masker')

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)

    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    model.compile(loss=[CE, mask_loss, NCE],
                  loss_weights=[1, alpha, beta],
                  optimizer=opt,
                  metrics={'crossentropy': ['categorical_accuracy'], 'all_masked': ['categorical_accuracy']})

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


def fast_fourier_transformer(
                             learning_rate,
                             image_size,
                             attention_heads,
                             loss='categorical_crossentropy',
                             l1=None, l2=None,
                             n_classes=10,
                             dropout=0.0,
                             **kwargs
                            ):

    conv_params = {
        'use_bias': True,
        'kernel_initializer': tf.keras.initializers.LecunNormal(),
        'bias_initializer': tf.keras.initializers.LecunNormal(),
        'kernel_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        'bias_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
    }
    if isinstance(attention_heads, str):
        attention_heads = [int(i) for i in attention_heads.strip('[]').split(', ')]
    # define the input layer (required)
    inputs = Input(image_size)
    x = inputs
    # set reference x separately to keep track of the input layer

    x = Conv2D(filters=32, kernel_size=(4, 4), strides=(4, 4), **conv_params, padding='same')(x)

    for i, heads in enumerate(attention_heads):
        x = Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), **conv_params, padding='same')(x)
        x = LayerNormalization()(x)
        # for all layers except the last one, we return sequences
        x = Concatenate()([x, TFPositionalEncoding2D(2)(x)])
        skip = x
        input_shape = x.shape
        key_dim = value_dim = x.shape[-1]
        if i == len(attention_heads) - 1:
            fft = tf.signal.fft(tf.cast(x, tf.complex64))

            real = tf.math.real(fft)
            imag = tf.math.imag(fft)

            mag = tf.cast(tf.math.sqrt(real ** 2 + imag ** 2), tf.float32)
            phase = tf.cast(tf.math.atan(imag / real), tf.float32)

            x = Reshape((input_shape[1] * input_shape[2], input_shape[-1]))(x)
            mag = Reshape((input_shape[1] * input_shape[2], input_shape[-1]))(real)
            phase = Reshape((input_shape[1] * input_shape[2], input_shape[-1]))(imag)

            mag = LayerNormalization()(mag)
            phase = LayerNormalization()(phase)
            # at the last layer of attention set the output to be a vector instead of a matrix
            x = MultiHeadAttention(heads,
                                   key_dim,
                                   value_dim,
                                   kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2),
                                   dropout=dropout)(mag, phase, x)

            x = Reshape((input_shape[1], input_shape[2], input_shape[-1]))(x)
        else:
            fft = tf.signal.fft(tf.cast(x, tf.complex64))

            real = tf.math.real(fft)
            imag = tf.math.imag(fft)

            mag = tf.cast(tf.math.sqrt(real ** 2 + imag ** 2), tf.float32)
            phase = tf.cast(tf.math.atan(imag / real), tf.float32)

            x = Reshape((input_shape[1] * input_shape[2], input_shape[-1]))(x)
            mag = Reshape((input_shape[1] * input_shape[2], input_shape[-1]))(real)
            phase = Reshape((input_shape[1] * input_shape[2], input_shape[-1]))(imag)

            mag = LayerNormalization()(mag)
            phase = LayerNormalization()(phase)

            x = MultiHeadAttention(heads,
                                   key_dim,
                                   value_dim,
                                   kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2),
                                   dropout=dropout)(mag, phase, x)

            x = Reshape((input_shape[1], input_shape[2], input_shape[-1]))(x)

        x = Add()([x, skip])
        x = LayerNormalization()(x)

    def replacenan(t):
        return tf.where(tf.math.is_nan(t), tf.zeros_like(t), t)

    x = replacenan(x)
    x = GlobalMaxPooling2D()(x)

    outputs = Dense(n_classes, activation=tf.keras.activations.softmax)(x)
    # build the model
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs],
                           name=f'vit_model_{"%02d" % time()}')

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)
    accuracy = 'sparse_categorical_accuracy' if loss == 'sparse_categorical_crossentropy' else 'categorical_accuracy'
    # compile the model
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[accuracy])

    return model


def build_camnet_reorderedv5(conv_filters,
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
                             alpha=1e-3,
                             beta=1e-3,
                             noise_level=0.05,
                             depth=5,
                             **kwargs):
    def CE(y_true, y_pred):
        return tf.math.negative(tf.reduce_sum(tf.math.log(y_pred) * y_true, axis=-1))

    def NCE(y_true, y_pred):
        if tf.shape(y_pred)[-1] > 0:
            return tf.math.negative(tf.math.log(tf.reduce_mean(y_pred, axis=-1)) - tf.math.log(tf.reduce_max(y_pred, axis=-1)))
        
        else:
            return tf.expand_dims(tf.math.negative(tf.math.log(0.0)), axis=0)

    def mask_loss(y_true, y_pred):
        return tf.math.negative(tf.math.log(1 + 2 ** (-16) - tf.reduce_sum(y_pred, keepdims=True, axis=-1)))

    if isinstance(conv_filters, str):
        conv_filters = [int(i) for i in conv_filters.strip('[]').split(', ')]
    if isinstance(conv_size, str):
        conv_size = [int(i) for i in conv_size.strip('[]').split(', ')]
    if isinstance(dense_layers, str):
        dense_layers = [int(i) for i in dense_layers.strip('[]').split(', ')]

    conv_params = {
        'use_bias': False,
        'kernel_initializer': tf.keras.initializers.GlorotUniform(),
        'bias_initializer': 'zeros',
        'kernel_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        'bias_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        'padding': 'same'
    }

    # in the masker we replace the masked pixels with the mean of the input tensor plus some noise

    def mask_plurality(image_size, cam_size, pred_size):
        image_inputs = Input(image_size)
        cam_inputs = Input(cam_size)
        pred_inputs = Input(pred_size)

        i = tf.argmax(pred_inputs[:, :-1], axis=-1)
        print(cam_inputs.shape)
        m = tf.gather(cam_inputs, i, axis=-1, batch_dims=1)
        # m = tf.slice(layer.output, i, axis=-1)
        m = tf.expand_dims(m, axis=-1)

        # mask out the class relevant pixels
        z = image_inputs * (tf.ones_like(m) - m)
        # replace with noise that preserves the mean

        batch_mean = tf.reduce_mean(image_inputs, axis=0)

        masker_outputs = z + (m * (batch_mean + tf.random.normal(tf.shape(batch_mean), stddev=noise_level)))
        
        model = tf.keras.Model(inputs=[image_inputs, cam_inputs, pred_inputs], outputs=[masker_outputs],
                               name=f'plurality_masker')

        return model

    def mask_total(image_size, cam_size, pred_size):
        image_inputs = Input(image_size)
        cam_inputs = Input(cam_size)
        pred_inputs = Input(pred_size)

        m = tf.reduce_sum(cam_inputs[:, :, :, :-1], keepdims=True, axis=-1)

        # mask out the class relevant pixels
        z = image_inputs * (tf.ones_like(m) - m)
        # replace with noise that preserves the mean
        
        batch_mean = tf.reduce_mean(image_inputs, axis=0)

        masker_outputs = z + (m * (batch_mean + tf.random.normal(tf.shape(batch_mean), stddev=noise_level)))

        model = tf.keras.Model(inputs=[image_inputs, cam_inputs, pred_inputs], outputs=[masker_outputs],
                               name=f'all_masker')

        return model

    model_inputs = Input(image_size)

    base = transformer_unet(24, image_size, n_classes=n_classes, depth=depth)

    masked_pred = mask_plurality(image_size, base.outputs[-1].shape[1:], base.outputs[0].shape[1:])
    masked_all = mask_total(image_size, base.outputs[-1].shape[1:], base.outputs[0].shape[1:])

    base_out, idk, cam = base(model_inputs)

    masked_inputs_pred = masked_pred([model_inputs, cam, base_out])
    masked_inputs_all = masked_all([model_inputs, cam, base_out])

    # masked_inputs_pred = Lambda(lambda z: K.stop_gradient(z))(masked_inputs_pred)
    # masked_inputs_all = Lambda(lambda z: K.stop_gradient(z))(masked_inputs_all)

    masked_pred_out, _, _ = base(masked_inputs_pred)
    masked_all_out, _, _ = base(masked_inputs_all)

    normed_base_out, _ = tf.linalg.normalize(base_out, axis=-1)
    normed_masked_pred_out, _ = tf.linalg.normalize(masked_pred_out, axis=-1)

    masked_pred_out = normed_base_out * normed_masked_pred_out

    outputs = {'crossentropy': base_out, 'cosine': masked_pred_out, 'all_masked': masked_all_out}

    model = tf.keras.Model(inputs=[model_inputs], outputs=outputs,
                           name=f'clam_masker')

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)

    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    model.compile(loss=[CE, mask_loss, NCE],
                  loss_weights=[1, alpha, beta],
                  optimizer=opt,
                  metrics={'crossentropy': ['categorical_accuracy'], 'all_masked': ['categorical_accuracy']})

    return model


def build_camnet_reorderedv6(conv_filters,
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
                             alpha=1e-3,
                             beta=1e-3,
                             noise_level=0.05,
                             depth=5,
                             **kwargs):
    def CE(y_true, y_pred):
        return tf.math.negative(tf.reduce_sum(tf.math.log(y_pred) * y_true, axis=-1))

    def NCE(y_true, y_pred):
        if tf.shape(y_pred)[-1] > 0:
            return tf.math.negative(tf.math.log(tf.reduce_mean(y_pred, axis=-1)) - tf.math.log(tf.reduce_max(y_pred, axis=-1)))
        
        else:
            return tf.expand_dims(tf.math.negative(tf.math.log(0.0)), axis=0)

    def mask_loss(y_true, y_pred):
        return tf.math.negative(tf.math.log(1 + 2 ** (-16) - tf.reduce_sum(y_pred, keepdims=True, axis=-1)))

    if isinstance(conv_filters, str):
        conv_filters = [int(i) for i in conv_filters.strip('[]').split(', ')]
    if isinstance(conv_size, str):
        conv_size = [int(i) for i in conv_size.strip('[]').split(', ')]
    if isinstance(dense_layers, str):
        dense_layers = [int(i) for i in dense_layers.strip('[]').split(', ')]

    conv_params = {
        'use_bias': False,
        'kernel_initializer': tf.keras.initializers.GlorotUniform(),
        'bias_initializer': 'zeros',
        'kernel_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        'bias_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        'padding': 'same'
    }

    # in the masker we replace the masked pixels with the mean of the input tensor plus some noise

    def mask_plurality(image_size, cam_size, pred_size):
        image_inputs = Input(image_size)
        cam_inputs = Input(cam_size)
        pred_inputs = Input(pred_size)

        i = tf.argmax(pred_inputs[:, :-1], axis=-1)
        print(cam_inputs.shape)
        m = tf.gather(cam_inputs, i, axis=-1, batch_dims=1)
        # m = tf.slice(layer.output, i, axis=-1)
        m = tf.expand_dims(m, axis=-1)

        # mask out the class relevant pixels
        z = image_inputs * (tf.ones_like(m) - m)
        # replace with noise that preserves the mean

        batch_mean = tf.reduce_mean(image_inputs, axis=0)

        masker_outputs = z + (m * (batch_mean + tf.random.normal(tf.shape(batch_mean), stddev=noise_level)))
        
        model = tf.keras.Model(inputs=[image_inputs, cam_inputs, pred_inputs], outputs=[masker_outputs],
                               name=f'plurality_masker')

        return model

    def mask_total(image_size, cam_size, pred_size):
        image_inputs = Input(image_size)
        cam_inputs = Input(cam_size)
        pred_inputs = Input(pred_size)

        m = tf.reduce_sum(cam_inputs[:, :, :, :-1], keepdims=True, axis=-1)

        # mask out the class relevant pixels
        z = image_inputs * (tf.ones_like(m) - m)
        # replace with noise that preserves the mean
        
        batch_mean = tf.reduce_mean(image_inputs, axis=0)

        masker_outputs = z + (m * (batch_mean + tf.random.normal(tf.shape(batch_mean), stddev=noise_level)))

        model = tf.keras.Model(inputs=[image_inputs, cam_inputs, pred_inputs], outputs=[masker_outputs],
                               name=f'all_masker')

        return model

    model_inputs = Input(image_size)

    base = transformer_unet(24, image_size, n_classes=n_classes, depth=depth)

    masked_pred = mask_plurality(image_size, base.outputs[-1].shape[1:], base.outputs[0].shape[1:])
    masked_all = mask_total(image_size, base.outputs[-1].shape[1:], base.outputs[0].shape[1:])

    base_out, idk, cam = base(model_inputs)

    masked_inputs_pred = masked_pred([model_inputs, cam, base_out])
    masked_inputs_all = masked_all([model_inputs, cam, base_out])

    masked_inputs_pred = Lambda(lambda z: K.stop_gradient(z))(masked_inputs_pred)
    masked_inputs_all = Lambda(lambda z: K.stop_gradient(z))(masked_inputs_all)

    masked_pred_out, _, _ = base(masked_inputs_pred)
    masked_all_out, _, _ = base(masked_inputs_all)

    normed_base_out, _ = tf.linalg.normalize(base_out, axis=-1)
    normed_masked_pred_out, _ = tf.linalg.normalize(masked_pred_out, axis=-1)

    masked_pred_out = normed_base_out * normed_masked_pred_out

    outputs = {'crossentropy': base_out, 'cosine': masked_pred_out, 'all_masked': masked_all_out}

    model = tf.keras.Model(inputs=[model_inputs], outputs=outputs,
                           name=f'clam_masker')

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)

    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    model.compile(loss=[CE, mask_loss, NCE],
                  loss_weights=[1, alpha, beta],
                  optimizer=opt,
                  metrics={'crossentropy': ['categorical_accuracy'], 'all_masked': ['categorical_accuracy']})

    return model


def build_camnet_reorderedv7(conv_filters,
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
                             alpha=1e-3,
                             beta=1e-3,
                             noise_level=0.05,
                             depth=5,
                             **kwargs):
    def CE(y_true, y_pred):
        return tf.math.negative(tf.reduce_sum(tf.math.log(y_pred) * y_true, axis=-1))

    def NCE(y_true, y_pred):
        if tf.shape(y_pred)[-1] > 0:
            return tf.math.negative(tf.math.log(tf.reduce_mean(y_pred, axis=-1)) - tf.math.log(tf.reduce_max(y_pred, axis=-1)))
        
        else:
            return tf.expand_dims(tf.math.negative(tf.math.log(0.0)), axis=0)

    def mask_loss(y_true, y_pred):
        return tf.math.negative(tf.math.log(1 + 2 ** (-16) - tf.reduce_sum(y_pred, keepdims=True, axis=-1)))

    if isinstance(conv_filters, str):
        conv_filters = [int(i) for i in conv_filters.strip('[]').split(', ')]
    if isinstance(conv_size, str):
        conv_size = [int(i) for i in conv_size.strip('[]').split(', ')]
    if isinstance(dense_layers, str):
        dense_layers = [int(i) for i in dense_layers.strip('[]').split(', ')]

    conv_params = {
        'use_bias': False,
        'kernel_initializer': tf.keras.initializers.GlorotUniform(),
        'bias_initializer': 'zeros',
        'kernel_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        'bias_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        'padding': 'same'
    }

    # in the masker we replace the masked pixels with the mean of the input tensor plus some noise

    def mask_plurality(image_size, cam_size, pred_size):
        image_inputs = Input(image_size)
        cam_inputs = Input(cam_size)
        pred_inputs = Input(pred_size)

        i = tf.argmax(pred_inputs[:, :-1], axis=-1)
        print(cam_inputs.shape)
        m = tf.gather(cam_inputs, i, axis=-1, batch_dims=1)
        # m = tf.slice(layer.output, i, axis=-1)
        m = tf.expand_dims(m, axis=-1)

        # mask out the class relevant pixels
        z = image_inputs * (tf.ones_like(m) - m)
        # replace with noise that preserves the mean

        batch_mean = tf.reduce_mean(image_inputs, axis=0)

        masker_outputs = z + (m * (batch_mean + tf.random.normal(tf.shape(batch_mean), stddev=noise_level)))
        
        model = tf.keras.Model(inputs=[image_inputs, cam_inputs, pred_inputs], outputs=[masker_outputs],
                               name=f'plurality_masker')

        return model

    def mask_total(image_size, cam_size, pred_size):
        image_inputs = Input(image_size)
        cam_inputs = Input(cam_size)
        pred_inputs = Input(pred_size)

        m = tf.reduce_sum(cam_inputs[:, :, :, :-1], keepdims=True, axis=-1)

        # mask out the class relevant pixels
        z = image_inputs * (tf.ones_like(m) - m)
        # replace with noise that preserves the mean
        
        batch_mean = tf.reduce_mean(image_inputs, axis=0)

        masker_outputs = z + (m * (batch_mean + tf.random.normal(tf.shape(batch_mean), stddev=noise_level)))

        model = tf.keras.Model(inputs=[image_inputs, cam_inputs, pred_inputs], outputs=[masker_outputs],
                               name=f'all_masker')

        return model

    model_inputs = Input(image_size)

    base = transformer_unet(conv_filters, image_size, n_classes=n_classes, depth=depth)

    masked_pred = mask_plurality(image_size, base.outputs[-1].shape[1:], base.outputs[0].shape[1:])
    masked_all = mask_total(image_size, base.outputs[-1].shape[1:], base.outputs[0].shape[1:])

    base_out, idk, cam = base(model_inputs)

    masked_inputs_pred = masked_pred([model_inputs, cam, base_out])
    masked_inputs_all = masked_all([model_inputs, cam, base_out])

    # masked_inputs_pred = Lambda(lambda z: K.stop_gradient(z))(masked_inputs_pred)
    # masked_inputs_all = Lambda(lambda z: K.stop_gradient(z))(masked_inputs_all)

    masked_pred_out, _, _ = base(masked_inputs_pred)
    masked_all_out, _, _ = base(masked_inputs_all)

    normed_base_out, _ = tf.linalg.normalize(base_out, axis=-1)
    normed_masked_pred_out, _ = tf.linalg.normalize(masked_pred_out, axis=-1)

    masked_pred_out = normed_base_out * normed_masked_pred_out

    outputs = {'crossentropy': base_out, 'cosine': masked_pred_out, 'all_masked': masked_all_out}

    model = tf.keras.Model(inputs=[model_inputs], outputs=outputs,
                           name=f'clam_masker')

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)

    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    model.compile(loss=[CE, mask_loss, NCE],
                  loss_weights=[1, alpha, beta],
                  optimizer=opt,
                  metrics={'crossentropy': ['categorical_accuracy'], 'all_masked': ['categorical_accuracy']})

    return model
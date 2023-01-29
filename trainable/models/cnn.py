import tensorflow as tf
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, Dense, Input, Concatenate, Dropout,\
    BatchNormalization, GlobalMaxPooling2D

from tensorflow.keras.applications import EfficientNetB0, ResNet50V2, VGG16, MobileNetV3Small
from time import time

from trainable.models.ae import transformer_unet, vit_unet
from trainable.losses import CE, NCE, logsum, identity


"""
model building functions should accept only float, int, or string arguments and must return only a compiled keras model
"""


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


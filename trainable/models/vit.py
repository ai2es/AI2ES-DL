"""
Transformer architectures for images

model building functions should accept only float, int, or string arguments
and must return only a compiled keras model
"""
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, Dense, Input, Concatenate, Dropout, \
    BatchNormalization, GlobalMaxPooling2D, Multiply, LayerNormalization, MultiHeadAttention, Reshape, Add

from trainable.custom_layers import TFPositionalEncoding2D, focal_module, GlobalResponseNormalization
from trainable.custom_layers import VLunchboxMHSA as LunchboxMHSA
from trainable.losses import identity, logsum, CE, NCE

from trainable.models.ae import transformer_unet, vit_unet

from time import time


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
    """
    The fast fourier transformer.  A linear embedding is used for 4x4 patches, and then subsequent layers are
    multi-headed self-attention where the V is the input, Q is the real component of the FFT of the input, and
    K is the imaginary component.

    :param learning_rate: learning rate
    :param image_size: input image dimensions
    :param attention_heads: number of attention heads in each transformer layer
    :param loss: loss function
    :param l1: l1 regularization term weight
    :param l2: l2 regularization term weight
    :param n_classes: number of output classes
    :param dropout: dropout rate
    :return: a compiled keras model
    """
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


def build_focal_modulator(image_size,
                          n_classes,
                          e_dim=24,
                          learning_rate=1e-3,
                          blocks=5,
                          depth=3,
                          loss='categorical_crossentropy',
                          **kwargs):

    """
    simple image classification network built with focal modules

    :param image_size: input image dimensions
    :param n_classes: number of output classes
    :param e_dim: number of filters in the first convolutional block\
    :param learning_rate: learning rate
    :param blocks: number of modulation blocks
    :param depth: depth of each modulation block
    :param loss: loss function
    :return: a compiled keras model
    """
    inputs = Input(image_size)
    x = inputs
    # patch partitioning and embedding
    x = Conv2D(e_dim, kernel_size=4, strides=4)(x)
    for i in range(blocks):
        skip = x
        x = focal_module(24 * (i + 1), depth)(x)
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


def build_focal_LAXNet(conv_filters,
                      learning_rate,
                      image_size,
                      n_classes=10,
                      alpha=1e-3,
                      beta=1e-3,
                      gamma=0.0,
                      noise_level=0.05,
                      depth=5,
                      **kwargs):
    """
    LAX image classification network built with focal modules

    :param conv_filters: Convolutional filters in the first convolutional block
    :param learning_rate: learning rate
    :param image_size: input image dimensions
    :param n_classes: number of output classes
    :param alpha: Masking loss weight
    :param beta: NCE loss weight
    :param gamma: Penalty for masking a pixel loss weight
    :param noise_level: standard deviation of noise to add to masked pixels
    :param depth: depth of 'U' in the image-to-image model
    :return: a compiled keras model

    """

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

    outputs = {'crossentropy': base_out, 'cosine': masked_pred_out, 'all_masked': masked_all_out,
               'idk': tf.reduce_sum(tf.reduce_mean(cam, axis=(1, 2))[:, :-1], -1, keepdims=True)}

    model = tf.keras.Model(inputs=[model_inputs], outputs=outputs,
                           name=f'clam_masker')

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)

    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    model.compile(loss=[CE, logsum, NCE, identity],
                  loss_weights=[1, alpha, beta, gamma],
                  optimizer=opt,
                  metrics={'crossentropy': ['categorical_accuracy'], 'all_masked': ['categorical_accuracy']})

    return model


def build_conv_LAXNet(conv_filters,
                       learning_rate,
                       image_size,
                       n_classes=10,
                       alpha=1e-3,
                       beta=1e-3,
                       gamma=1e-4,
                       noise_level=0.05,
                       depth=5,
                       **kwargs):
    """
    LAX image classification network built with convolution and attention modules

    :param conv_filters: Convolutional filters in the first convolutional block
    :param learning_rate: learning rate
    :param image_size: input image dimensions
    :param n_classes: number of output classes
    :param alpha: Masking loss weight
    :param beta: NCE loss weight
    :param gamma: Penalty for masking a pixel loss weight
    :param noise_level: standard deviation of noise to add to masked pixels
    :param depth: depth of 'U' in the image-to-image model
    :return: a compiled keras model

    """
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

    base = vit_unet(conv_filters, image_size, n_classes=n_classes, depth=depth)

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

    outputs = {'crossentropy': base_out, 'cosine': masked_pred_out, 'all_masked': masked_all_out,
               'idk': tf.reduce_sum(tf.reduce_mean(cam, axis=(1, 2))[:, :-1], -1, keepdims=True)}

    model = tf.keras.Model(inputs=[model_inputs], outputs=outputs,
                           name=f'clam_masker')

    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                    beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.99)

    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    model.compile(loss=[CE, logsum, NCE, identity],
                  loss_weights=[1, alpha, beta, gamma],
                  optimizer=opt,
                  metrics={'crossentropy': ['categorical_accuracy'], 'all_masked': ['categorical_accuracy']})

    return model


def build_basic_lunchbox(conv_filters,
                         dense_layers,
                         learning_rate,
                         image_size,
                         loss='categorical_crossentropy',
                         l1=None,
                         l2=None,
                         activation=lambda x: x * tf.nn.relu6(x + 3) / 6,
                         n_classes=10,
                         **kwargs):
    conv_params = {
        'use_bias': False,
        'kernel_initializer': tf.keras.initializers.GlorotUniform(),
        'bias_initializer': 'zeros',
        'kernel_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        'bias_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2),
        'padding': 'same'
    }

    if isinstance(conv_filters, str):
        conv_filters = [int(i) for i in conv_filters.strip('[]').split(', ')]

    if isinstance(dense_layers, str):
        dense_layers = [int(i) for i in dense_layers.strip('[]').split(', ')]

    inputs = Input(image_size)

    x = inputs

    x = Conv2D(32, 4, 4, **conv_params)(x)

    for block in conv_filters:
        w, h, ch = x.shape[1], x.shape[2], x.shape[-1]
        x = LayerNormalization()(x)

        x = Reshape((w * h, ch))(x)
        x = LunchboxMHSA(block, 4, (w * h) // 4)(x)
        x = Reshape((w // 2, h // 2, block))(x)

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

import tensorflow as tf


def CE(y_true, y_pred):
    return tf.math.negative(tf.reduce_sum(tf.math.log(y_pred) * y_true, axis=-1))


def NCE(y_true, y_pred):
    return tf.math.negative(
        tf.math.log(tf.reduce_mean(y_pred, axis=-1)) - tf.math.log(tf.reduce_max(y_pred, axis=-1)))


def logsum(y_true, y_pred):
    return tf.math.negative(tf.math.log(1 + 2 ** (-16) - tf.reduce_sum(y_pred, keepdims=True, axis=-1)))


def identity(y_true, y_pred):
    return y_pred